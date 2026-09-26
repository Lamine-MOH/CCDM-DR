#!/usr/bin/env python3
"""
generate_from_ckpt.py

Generate grade-conditioned fundus images from a trained CCDM-DR checkpoint
WITHOUT re-training and WITHOUT loading the full training set.

The diffusion weights live in `results/model-{step}.pt` (written by trainer.py).
That checkpoint stores the EMA model (the one used for sampling) but NOT the
label-embedding networks or the sigma_data setting, so this script also needs:

  - {root}/output/DRGrading_{image_size}/model_y2h/ckpt_mlp_y2h_epoch_500.pth
  - {root}/output/DRGrading_{image_size}/model_y2cov/ckpt_cnn_y2cov_epoch_500.pth
  - the exact config/model_cfg/*.yaml used for training

All CLI defaults match config/DR128/run_train.sh. Only --grades, --nfake_per_grade
and --cond_scale normally need changing; --out_dir defaults to
`output/generated_cs{cond_scale}` and the h5 stores its generation attrs.

Output: `{out_dir}/generated.h5` with the SAME schema as the training h5
(images: uint8, N x 3 x H x W; labels: float64, raw grades 0-4), so it feeds
directly into downstream_eval/train_dr_classifier.py --synthetic_h5.
Per-grade preview grids are saved alongside it as sample_grade_{g}.png.

Example:
  python generate_from_ckpt.py \
      --model_ckpt output/DRGrading_128/setup1_dr/results/model-150000.pt \
      --model_config config/model_cfg/unet_edm_128_v1.yaml \
      --root_path . --image_size 128 \
      --grades 0 1 2 3 4 --nfake_per_grade 1000 --cond_scale 4
  (default out_dir is output/generated_cs4; pass --out_dir to override)
"""

import argparse
import os
import json

import numpy as np
import torch
import torchvision
import h5py
import yaml

from ema_pytorch import EMA

from models import UNet_EDM
from models.resnet_y2h import model_y2h
from models.resnet_y2cov import model_cnn_y2cov
from diffusion import ElucidatedDiffusion


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--model_ckpt", type=str, required=True,
                   help="path to results/model-{step}.pt")
    p.add_argument("--model_config", type=str, required=True,
                   help="the config/model_cfg/*.yaml used for training")
    p.add_argument("--root_path", type=str, default="./",
                   help="repo root; embedding ckpts are looked up under "
                        "{root_path}/output/DRGrading_{image_size}/{...}")

    # label embedding
    p.add_argument("--path_y2h", type=str, default=None,
                   help="dir containing the y2h ckpt (default: {root}/output/DRGrading_{size}/model_y2h)")
    p.add_argument("--path_y2cov", type=str, default=None,
                   help="dir containing the y2cov ckpt (default: {root}/output/DRGrading_{size}/model_y2cov)")
    p.add_argument("--y2h_ckpt_name", type=str, default="ckpt_mlp_y2h_epoch_500.pth")
    p.add_argument("--y2cov_ckpt_name", type=str, default="ckpt_cnn_y2cov_epoch_500.pth")
    p.add_argument("--dim_embed", type=int, default=128)
    p.add_argument("--use_y2cov", action=argparse.BooleanOptionalAction, default=True,
                   help="must match training; DR configs train with y2cov (--no-use_y2cov to disable)")
    p.add_argument("--y2cov_hy_weight_train", type=float, default=0.05)
    p.add_argument("--y2cov_hy_weight_test", type=float, default=0.05)

    # data / model
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--num_channels", type=int, default=3)
    p.add_argument("--max_label", type=float, default=4.0,
                   help="max raw label; labels are normalized to grade/max_label")

    # EDM hyperparameters (must match training; DR128 uses these defaults)
    p.add_argument("--edm_sigma_data_type", type=str, default="default",
                   choices=["default", "global", "local"],
                   help="sigma_data scheme; authoritative value is read from "
                        "edm_sigma_data.json next to the checkpoint. This flag is "
                        "the fallback only when that file is missing.")
    p.add_argument("--edm_sigma_data_default", type=float, default=0.5)
    p.add_argument("--edm_sigma_min", type=float, default=0.002)
    p.add_argument("--edm_sigma_max", type=float, default=80)
    p.add_argument("--edm_rho", type=float, default=7)
    p.add_argument("--edm_P_mean", type=float, default=-1.2)
    p.add_argument("--edm_P_std", type=float, default=1.2)
    p.add_argument("--edm_S_churn", type=float, default=80)
    p.add_argument("--edm_S_tmin", type=float, default=0.05)
    p.add_argument("--edm_S_tmax", type=float, default=50)
    p.add_argument("--edm_S_noise", type=float, default=1.003)

    # EMA construction (must match trainer.py)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_update_after_step", type=int, default=0)
    p.add_argument("--ema_update_every", type=int, default=10)

    # sampling
    p.add_argument("--sampler", type=str, default="sde", choices=["sde", "ode", "dpmpp"])
    p.add_argument("--num_sample_steps", type=int, default=32)
    p.add_argument("--cond_scale", type=float, default=1.5)
    p.add_argument("--rescaled_phi", type=float, default=0.7)

    # what to generate
    p.add_argument("--grades", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--nfake_per_grade", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--out_dir", type=str, default=None,
                   help="output dir (default: output/generated_cs{cond_scale})")
    p.add_argument("--out_name", type=str, default="generated.h5")

    # reproducibility. Default seed 111 reproduces every pre-existing generated
    # set; --reseed_per_grade is opt-in (it changes the noise draws).
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--reseed_per_grade", action="store_true",
                   help="seed once per grade (seed + grade) so two samplers start "
                        "every grade from the same init noise. Off by default: without "
                        "it the RNG carries across grades, as it always has.")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = "output/generated_cs{}".format(args.cond_scale)
    return args


def load_sigma_data(args):
    """Reconcile the persisted edm_sigma_data.json with CLI fallbacks.

    Returns (sigma_type, default_val, y_unique, sigma_unique). The metadata JSON
    lives next to the checkpoint (results/edm_sigma_data.json, written by main.py).
    - metadata present -> authoritative (reconstructs global/local exactly).
    - metadata missing + non-default request -> hard error (no silent 0.5 fallback).
    - metadata missing + 'default' -> legacy CLI fallback with a warning.
    """
    meta_path = os.path.join(os.path.dirname(os.path.abspath(args.model_ckpt)), "edm_sigma_data.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(" Loaded sigma_data metadata from {}".format(meta_path))
        return (meta.get("type", args.edm_sigma_data_type),
                meta.get("default_val", args.edm_sigma_data_default),
                meta.get("y_unique"), meta.get("sigma_unique"))
    if args.edm_sigma_data_type != "default":
        raise RuntimeError(
            "{} not found and --edm_sigma_data_type is '{}' (non-default). "
            "No silent 0.5 fallback -- copy the metadata from the training machine "
            "or reprocess with edm_sigma_data_type=default.".format(meta_path, args.edm_sigma_data_type))
    print(" WARNING: {} not found; falling back to CLI --edm_sigma_data_default {} "
          "(legacy, verify it matches training).".format(meta_path, args.edm_sigma_data_default))
    return ("default", float(args.edm_sigma_data_default), None, None)


def _check_label_type(y):
    if isinstance(y, torch.Tensor):
        return "tensor"
    elif isinstance(y, (int, float)):
        return "scalar"
    raise TypeError("labels `y` must be a scalar or a torch.Tensor.")


def build_sigma_data_fn(sigma_type, default_val, y_unique=None, sigma_unique=None):
    """Replicates main.py's fn_y2sigma_data for all three edm_sigma_data_type
    branches, so generation uses the exact sigma_data scheme training used."""

    if sigma_type in ("default", "global"):
        const = float(default_val)

        def fn_y2sigma_data(y):
            kind = _check_label_type(y)
            if kind == "scalar":
                return const
            return torch.full_like(y, fill_value=const)

        return fn_y2sigma_data

    if sigma_type == "local":
        y_unique = np.asarray(y_unique, dtype=float)
        sigma_unique = np.asarray(sigma_unique, dtype=float)

        def fn_y2sigma_data(y, y_unique=y_unique, sigma_unique=sigma_unique):
            kind = _check_label_type(y)
            if kind == "scalar":
                y_np = np.array([float(y)], dtype=float)
                orig_shape = y_np.shape
            else:
                y_np = y.detach().cpu().numpy().astype(float)
                orig_shape = y_np.shape
            y_flat = y_np.ravel()
            idx = np.searchsorted(y_unique, y_flat, side="left")
            idx_clipped = np.minimum(idx, len(y_unique) - 1)
            res = np.empty_like(y_flat, dtype=float)
            mask_equal = (idx < len(y_unique)) & (y_flat == y_unique[idx_clipped])
            res[mask_equal] = sigma_unique[idx_clipped[mask_equal]]
            mask_not_equal = ~mask_equal
            mask_left = mask_not_equal & (idx == 0)
            res[mask_left] = sigma_unique[0]
            mask_right = mask_not_equal & (idx == len(y_unique))
            res[mask_right] = sigma_unique[-1]
            mask_between = mask_not_equal & (idx > 0) & (idx < len(y_unique))
            idx_between = idx[mask_between]
            res[mask_between] = 0.5 * (sigma_unique[idx_between - 1] + sigma_unique[idx_between])
            res = res.reshape(orig_shape)
            if kind == "scalar":
                return float(res.reshape(-1)[0])
            return torch.from_numpy(res).to(device=y.device, dtype=y.dtype)

        return fn_y2sigma_data

    raise ValueError("Invalid sigma_data type: {}".format(sigma_type))


def strip_module_prefix(sd):
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


def load_embed_net(module, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["net_state_dict"]
    sd = strip_module_prefix(sd)
    missing, unexpected = module.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError("Embedding net {}: missing keys {}".format(ckpt_path, missing))
    if unexpected:
        raise RuntimeError("Embedding net {}: unexpected keys {}".format(ckpt_path, unexpected))
    return module.to(device).eval()


def make_embed_fns(args, device):
    """Returns (fn_y2h, fn_y2cov) replicating LabelEmbed.fn_y2h / fn_y2cov
    for the 'resnet' embedding type, WITHOUT needing the training dataset."""

    if args.path_y2h is None:
        args.path_y2h = os.path.join(args.root_path, "output/DRGrading_{}/model_y2h".format(args.image_size))
    if args.path_y2cov is None:
        args.path_y2cov = os.path.join(args.root_path, "output/DRGrading_{}/model_y2cov".format(args.image_size))

    y2h_ckpt = os.path.join(args.path_y2h, args.y2h_ckpt_name)
    if not os.path.isfile(y2h_ckpt):
        raise FileNotFoundError(
            "Embedding checkpoint missing: {}\nIt is NOT stored in model-*.pt. "
            "Copy the model_y2h/ and model_y2cov/ folders from the training machine.".format(y2h_ckpt))

    net_y2h = load_embed_net(model_y2h(dim_embed=args.dim_embed), y2h_ckpt, device)

    def fn_y2h(labels):
        with torch.no_grad():
            return net_y2h(labels.to(device))

    if args.use_y2cov:
        y2cov_ckpt = os.path.join(args.path_y2cov, args.y2cov_ckpt_name)
        if not os.path.isfile(y2cov_ckpt):
            raise FileNotFoundError(
                "Embedding checkpoint missing: {}\nIt is NOT stored in model-*.pt. "
                "Copy the model_y2h/ and model_y2cov/ folders from the training machine.".format(y2cov_ckpt))
        net_y2cov = load_embed_net(model_cnn_y2cov(img_size=args.image_size, nc=args.num_channels), y2cov_ckpt, device)

        def fn_y2cov(labels):
            with torch.no_grad():
                return net_y2cov(labels.to(device))
    else:
        fn_y2cov = None

    return fn_y2h, fn_y2cov


def build_diffusion(args, fn_y2sigma_data, fn_y2cov):
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)

    if model_config["name"] != "UNet_EDM":
        raise ValueError("generate_from_ckpt.py currently supports UNet_EDM only "
                         "(got model_config name: {}).".format(model_config["name"]))
    if model_config["data"]["image_size"] != args.image_size:
        raise ValueError("yaml image_size {} != --image_size {}".format(
            model_config["data"]["image_size"], args.image_size))
    if model_config["data"]["num_channels"] != args.num_channels:
        raise ValueError("yaml num_channels {} != --num_channels {}".format(
            model_config["data"]["num_channels"], args.num_channels))

    unet = UNet_EDM(**model_config["model"])

    aux_loss_params = {
        "use_aux_reg_loss": False,
        "aux_reg_loss_type": "mse",
        "aux_reg_loss_weight": 0.0,
        "aux_reg_loss_epsilon": -1.0,
        "aux_reg_net": None,
    }

    diffusion = ElucidatedDiffusion(
        net=unet,
        fn_y2sigma_data=fn_y2sigma_data,
        aux_loss_params=aux_loss_params,
        image_size=args.image_size,
        channels=args.num_channels,
        num_sample_steps=args.num_sample_steps,
        sigma_data_default=args.edm_sigma_data_default,
        sigma_min=args.edm_sigma_min,
        sigma_max=args.edm_sigma_max,
        rho=args.edm_rho,
        P_mean=args.edm_P_mean,
        P_std=args.edm_P_std,
        S_churn=args.edm_S_churn,
        S_tmin=args.edm_S_tmin,
        S_tmax=args.edm_S_tmax,
        S_noise=args.edm_S_noise,
        cond_drop_prob=model_config["model"]["cond_drop_prob"],
        use_y2cov=args.use_y2cov,
        fn_y2cov=fn_y2cov,
        y2cov_hy_weight_train=args.y2cov_hy_weight_train,
        y2cov_hy_weight_test=args.y2cov_hy_weight_test,
    )
    return diffusion


def load_ema_model(diffusion, args):
    ckpt = torch.load(args.model_ckpt, map_location="cpu", weights_only=True)
    step = ckpt.get("step", None)
    print("\n Checkpoint step: {}".format(step))

    ema_state = ckpt.get("ema")
    if ema_state is None:
        raise RuntimeError("Checkpoint {} has no 'ema' key. It is not a trainer.py checkpoint.".format(args.model_ckpt))

    # Same construction/load as trainer.py, so the format is symmetric with
    # whatever ema-pytorch version produced the checkpoint.
    ema = EMA(
        diffusion,
        update_after_step=args.ema_update_after_step,
        beta=args.ema_decay,
        update_every=args.ema_update_every,
        coerce_dtype=True,
    ).to(args.device)

    try:
        ema.load_state_dict(ema_state)
        gen_model = ema.ema_model
        print(" Restored EMA weights from {}".format(args.model_ckpt))
    except Exception as e:  # fallback: extract ema_model.* sub-state directly
        print(" EMA symmetric load failed ({}); falling back to manual ema_model extraction.".format(e))
        if "ema_model" in ema_state:
            sd = {k[len("ema_model."):]: v for k, v in ema_state.items() if k.startswith("ema_model.")}
        else:
            sd = ema_state.get("model", ema_state)
        missing, unexpected = diffusion.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError("Manual EMA extraction failed. missing={} unexpected={}".format(missing, unexpected))
        gen_model = diffusion

    return gen_model.to(args.device).eval()


def sample_for_grade(gen_model, fn_y2h, grade, nfake, args):
    device = args.device
    max_label = float(args.max_label)
    fake_images = []
    remaining = nfake
    with torch.inference_mode():
        while remaining > 0:
            bs = min(args.batch_size, remaining)
            labels = torch.full((bs,), grade / max_label, dtype=torch.float32, device=device)
            labels_emb = fn_y2h(labels)

            if args.sampler == "sde":
                imgs = gen_model.sample_using_sde(
                    labels=labels, labels_emb=labels_emb, cond_scale=args.cond_scale,
                    rescaled_phi=args.rescaled_phi, num_sample_steps=args.num_sample_steps, clamp=False)
            elif args.sampler == "ode":
                imgs = gen_model.sample_using_ode(
                    labels=labels, labels_emb=labels_emb, cond_scale=args.cond_scale,
                    rescaled_phi=args.rescaled_phi, num_sample_steps=args.num_sample_steps, clamp=False)
            elif args.sampler == "dpmpp":
                imgs = gen_model.sample_using_dpmpp(
                    labels=labels, labels_emb=labels_emb, cond_scale=args.cond_scale,
                    rescaled_phi=args.rescaled_phi, num_sample_steps=args.num_sample_steps)
            else:
                raise ValueError("Unsupported sampler: {}".format(args.sampler))

            # sampler returns [0,1] float tensors -> cast to uint8 (same as trainer.sample_given_labels)
            fake_images.append((imgs * 255.0).type(torch.uint8).cpu())
            remaining -= bs

    return torch.cat(fake_images, dim=0)[:nfake]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def seed_everything(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    seed_everything(args.seed)

    fn_y2h, fn_y2cov = make_embed_fns(args, args.device)
    sigma_type, sigma_default_val, y_unique, sigma_unique = load_sigma_data(args)
    fn_y2sigma_data = build_sigma_data_fn(sigma_type, sigma_default_val, y_unique, sigma_unique)
    diffusion = build_diffusion(args, fn_y2sigma_data, fn_y2cov if args.use_y2cov else None)
    total_params = sum(p.numel() for p in diffusion.parameters())
    print(" Reconstructed {} ({} params).".format(type(diffusion.net).__name__, total_params))

    gen_model = load_ema_model(diffusion, args)

    images_all, labels_all = [], []
    for grade in args.grades:
        if args.reseed_per_grade:
            seed_everything(args.seed + grade)
        print("\n Generating {} images for grade {} (sampler={}, steps={})...".format(
            args.nfake_per_grade, grade, args.sampler, args.num_sample_steps))
        imgs = sample_for_grade(gen_model, fn_y2h, grade, args.nfake_per_grade, args)
        images_all.append(imgs)
        labels_all.append(np.full(imgs.shape[0], grade, dtype=np.float64))

        preview = imgs[:36].float() / 255.0
        torchvision.utils.save_image(preview, os.path.join(args.out_dir, "sample_grade_{}.png".format(grade)),
                                     nrow=6, normalize=False, padding=1)

    images = torch.cat(images_all, dim=0).numpy()
    labels = np.concatenate(labels_all)
    print("\n Generated {} images (shape {}).".format(len(images), images.shape))

    out_h5 = os.path.join(args.out_dir, args.out_name)
    with h5py.File(out_h5, "w") as f:
        f.attrs["cond_scale"] = args.cond_scale
        f.attrs["model_ckpt"] = args.model_ckpt
        f.attrs["max_label"] = args.max_label
        f.attrs["sampler"] = args.sampler
        f.attrs["num_sample_steps"] = args.num_sample_steps
        f.attrs["seed"] = args.seed
        f.attrs["reseed_per_grade"] = bool(args.reseed_per_grade)
        f.attrs["edm_sigma_data_type"] = sigma_type
        f.create_dataset("images", data=images, dtype="uint8", compression="gzip", compression_opts=6)
        f.create_dataset("labels", data=labels, dtype="float64")
    print(" Saved {}\n".format(out_h5))
    print(" Preview grids: {}/sample_grade_{{0..max}}.png".format(args.out_dir))


if __name__ == "__main__":
    main()
