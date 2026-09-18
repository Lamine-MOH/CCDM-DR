# adapted from https://github.com/NVlabs/edm/blob/main/training/networks.py
# The noise and condition mapping layers are modified

import os
import numpy as np
import torch
from torch import nn
from torch.nn.functional import silu
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat, pack, unpack

#----------------------------------------------------------------------------
# helper function

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

#----------------------------------------------------------------------------
# Self-attention backend.
#
# The original EDM `AttentionOp` (below) materializes the full softmax(Q^T K)
# matrix in FP32 (~2 GiB spike per layer) and saves it for backprop (~1
# GB/image/layer at 128px), so batch>1 OOMs at 256px. torch's
# scaled_dot_product_attention with the flash/memory-efficient backend neither
# materializes the matrix nor stores it (recomputed on backward). The backend
# used for a run is chosen at import time from the environment:
#
#    CCDM_ATTN_BACKEND=sdpa      (default) SDPA flash/mem-eff ONLY when truly
#                                eligible (CUDA autocast fp16/bf16 + compute
#                                >= 8.0 + torch >= 2.0); otherwise the chunked
#                                op below is used. The full-matrix SDPA math
#                                backend is never accepted.
#    CCDM_ATTN_BACKEND=chunked   fp32 softmax computed in query-chunks with
#                                recompute-on-backward; same math as the
#                                original op, peak O(chunk*L*heads) instead of
#                                O(L^2). Works on any dtype/torch/device.
#    CCDM_ATTN_BACKEND=original  bit-exact legacy `AttentionOp` (may OOM at
#                                256px; kept for bit-for-bit reproducibility).
#
# Legacy toggles: CCDM_ATTN_MATH=1 -> original, CCDM_ATTN_CHUNKED=1 -> chunked.

def _torch_min_version(major, minor):
    try:
        from packaging import version as _version
        return _version.parse(torch.__version__) >= _version.parse(f'{major}.{minor}')
    except Exception:
        try:
            parts = torch.__version__.split('+')[0].split('.')
            return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0) >= (major, minor)
        except Exception:
            return False

_ATTN_BACKEND = (os.environ.get('CCDM_ATTN_BACKEND', 'sdpa') or '').strip().lower()
if _ATTN_BACKEND not in ('sdpa', 'chunked', 'original'):
    raise ValueError('CCDM_ATTN_BACKEND must be one of sdpa|chunked|original, got %s' % _ATTN_BACKEND)

def _env_flag_on(name):
    return ((os.environ.get(name, '0') or '').strip().lower() in ('1', 'true', 'yes', 'on'))

## legacy toggles
if _env_flag_on('CCDM_ATTN_MATH'):
    _ATTN_BACKEND = 'original'
elif _env_flag_on('CCDM_ATTN_CHUNKED'):
    _ATTN_BACKEND = 'chunked'

# SDPA is only attempted when it is guaranteed memory-safe; any other mode,
# dtype, or device takes the (always bounded) chunked path below.
_USE_SDPA_ATTENTION = (_ATTN_BACKEND == 'sdpa') and _torch_min_version(2, 0)

def _sdpa_context():
    """sdpa_kernel restricted to flash/mem-eff (never math -> full softmax matrix)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if not torch.cuda.is_available():
        raise NotImplementedError('SDPA requires CUDA')
    device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
    if (device_properties.major, device_properties.minor) >= (8, 0):
        return sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION])
    return sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION])

#----------------------------------------------------------------------------
# classifier free guidance functions

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)



#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------
# Query-chunked softmax attention with the same FP32 math as AttentionOp, but
# with bounded peak memory: QK^T is never formed in full and the softmax
# weights are recomputed per chunk in backward instead of being saved. q/k/v
# use the (B, H, L, C) SDPA layout.

class AttentionOpChunked(torch.autograd.Function):
    q_chunk_size = 512

    @staticmethod
    def forward(ctx, q, k, v, q_chunk_size=512):
        q_f, k_f, v_f = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
        inv = 1.0 / np.sqrt(q.shape[-1])
        out = torch.empty_like(q)
        L = q.shape[-2]
        for start in range(0, L, q_chunk_size):
            end = min(start + q_chunk_size, L)
            w = (q_f[:, :, start:end] @ k_f.transpose(-2, -1)) * inv
            w = w.softmax(dim=-1)
            out[:, :, start:end] = (w.to(q.dtype) @ v_f).to(q.dtype)
        ctx.save_for_backward(q, k, v)
        ctx.q_chunk_size = q_chunk_size
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v = ctx.saved_tensors
        q_f, k_f, v_f = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
        dout_f = dout.to(torch.float32)
        inv = 1.0 / np.sqrt(q.shape[-1])
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        L = q.shape[-2]
        for start in range(0, L, ctx.q_chunk_size):
            end = min(start + ctx.q_chunk_size, L)
            qc = q_f[:, :, start:end]
            w = (qc @ k_f.transpose(-2, -1)) * inv
            w = w.softmax(dim=-1)
            dos = dout_f[:, :, start:end]
            dw = dos @ v_f.transpose(-2, -1)
            dlogits = w * (dw - (dw * w).sum(dim=-1, keepdim=True))
            dq[:, :, start:end] = (dlogits @ k_f * inv).to(q.dtype)
            dk += (dlogits.transpose(-2, -1) @ qc * inv).to(k.dtype)
            dv += (w.transpose(-2, -1) @ dos).to(v.dtype)
        return dq, dk, dv, None

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, cond_emb_channels=None, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        if cond_emb_channels is not None:
            self.affine_cond = Linear(in_features=cond_emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        else:
            self.affine_cond = None
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb, cond_emb=None):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.affine_cond is not None and cond_emb is not None:
            params = params + self.affine_cond(cond_emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            if _ATTN_BACKEND == 'original':
                q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
                w = AttentionOp.apply(q, k)
                a = torch.einsum('nqk,nck->ncq', w, v).reshape(*x.shape)
            else:
                a = self._attention(x)
            x = self.proj(a).add_(x)
            x = x * self.skip_scale
        return x

    def _attention(self, x):
        """Memory-bounded self-attention: SDPA flash/mem-eff when eligible,
        else the query-chunked fp32 op. Never uses the SDPA math backend --
        its full softmax matrix is what OOMs batch>1 at 256px."""
        head_dim = x.shape[1] // self.num_heads
        q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0], self.num_heads, head_dim, 3, -1).permute(3, 0, 1, 4, 2).unbind(0)
        if _USE_SDPA_ATTENTION and torch.is_autocast_enabled('cuda'):
            try:
                dtype = torch.get_autocast_gpu_dtype()
                qi = (q.to(dtype) * (head_dim ** -0.5)).contiguous()
                ki = k.to(dtype).contiguous()
                vi = v.to(dtype).contiguous()
                with _sdpa_context():
                    a = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0)
                return a.permute(0, 1, 3, 2).reshape(*x.shape)
            except (NotImplementedError, RuntimeError):
                pass
        return AttentionOpChunked.apply(q, k, v, AttentionOpChunked.q_chunk_size).permute(0, 1, 3, 2).reshape(*x.shape)

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    

#----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion

class UNet_EDM(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 128,          # Dimension of labels, 0 = unconditional.
        
        model_channels      = 64,           # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # dropout rate for resblock not for condition drop
        cond_drop_prob      = 0.10,         # Dropout probability of labels for classifier-free guidance.
        
        self_condition = False,             #not used. for compatibility only
    ):
        super().__init__()
        self.cond_drop_prob = cond_drop_prob
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels//2, cond_emb_channels=emb_channels//2, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)
        
        self.self_condition = self_condition #not used. for compatibility only
        

        # Noise/Time Mapping.        
        self.time_map = nn.Sequential(
            PositionalEmbedding(num_channels=model_channels), #embed noise level or time step
            Linear(in_features=model_channels, out_features=emb_channels//2, **init),
            nn.SiLU(),
            Linear(in_features=emb_channels//2, out_features=emb_channels//2, **init),
            nn.SiLU(),
        )
        
        # condition mapping
        
        ## null_cond_emb in (https://github.com/lucidrains/denoising-diffusion-pytorch/blob/1d9d8dffb72e02172da8a77bee039b1c72b7c6d5/denoising_diffusion_pytorch/classifier_free_guidance.py#L330) is defined as torch.randn with requires_grad=True
        self.null_cond_emb = nn.Parameter(-1*torch.abs(torch.randn(label_dim)), requires_grad=True)
        
        self.cond_map = nn.Sequential(
            nn.Linear(label_dim, emb_channels//2),
            nn.LayerNorm(emb_channels//2),
            nn.SiLU(),
        )
        
        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level # equal to img_resolution // (2**level)
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)
    

    # modified
    def forward(
        self, 
        x, 
        time,  #noise levels
        labels_emb, #condition labels after embedding
        cond_drop_prob = None,  #label drop probability
        keep_mask = None, # a given mask than indicates which labels are NOT dropped
        return_bottleneck=False,
        x_self_cond = None, #not used, for compatibility only.
        ):
        
        batch, device = x.shape[0], x.device
        
        # time/noise embeddings
        
        t_emb = self.time_map(time)
        
        # condition embeddings
        
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        
        if cond_drop_prob > 0: #randomly drop some labels by replacing them with 0
            
            if keep_mask is None: # if keep_mask is given, then use this given mask
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)                  
                
            null_cond_emb = repeat(self.null_cond_emb, 'd -> b d', b = batch)

            labels_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                labels_emb,
                null_cond_emb
            )
            
        c_emb = self.cond_map(labels_emb)
        
        emb = t_emb
        cond_emb = c_emb
        
        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb, cond_emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        if return_bottleneck:
            return x
        
        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb, cond_emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x
    
    
    # classifier-free guidance
    # borrowed from UNet_CCDM
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        rescaled_phi = 0.,
        remove_parallel_component = True,
        keep_parallel_frac = 0.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        update = logits - null_logits

        if remove_parallel_component:
            parallel, orthog = project(update, logits)
            update = orthog + parallel * keep_parallel_frac

        scaled_logits = logits + update * (cond_scale - 1.)

        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
        interpolated_rescaled_logits = rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

        return interpolated_rescaled_logits