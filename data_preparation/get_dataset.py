"""
get_dataset.py

Downloads and prepares diabetic retinopathy datasets into a common structure
that build_dr_h5.py can consume directly.

Supported datasets:
    APTOS     — APTOS 2019 Blindness Detection (kagglehub)
    IDRiD     — Indian Diabetic Retinopathy Image Dataset (Google Drive zip)
    DDR       — Diabetic Retinopathy Detection (kagglehub)
    Messidor-2 — Messidor-2 preprocessed (kagglehub)

Output structure (written to --save_path):
    {save_path}/{dataset_name}/
        Images/            — all images (train + test + val merged)
        labels.csv         — columns: id_code, diagnosis (ICDR 0-4)

CLI usage:
    python data_preparation/get_dataset.py --dataset Aptos
    python data_preparation/get_dataset.py --dataset IDRiD --save_path /data

After running, feed the output into build_dr_h5.py:
    python data_preparation/build_dr_h5.py \\
        --image_dir {dataset_path}/Images \\
        --csv_path  {dataset_path}/labels.csv \\
        --out_dir   {save_path}/DRGrading \\
        --img_size  128
"""

import argparse
import os
import shutil
import zipfile

import gdown
import kagglehub
import pandas as pd


def dataset_download(dataset_name, data_path="./data/"):
    """Download a raw DR dataset to data_path.

    Args:
        dataset_name: One of "Aptos", "IDRiD", "DDR", "Messidor-2".
        data_path: Root directory where the raw dataset is downloaded.

    Returns:
        Path to the downloaded dataset root directory.

    Sources:
        APTOS, DDR, Messidor-2 — downloaded via kagglehub.
        IDRiD — downloaded as a zip from Google Drive, extracted to
        {data_path}/IDRiD_RAW/, returns path to B.Disease Grading/ subfolder.
    """
    valid_datasets = ["Aptos", "IDRiD", "DDR", "Messidor-2"]
    if dataset_name not in valid_datasets:
        raise ValueError(f"Invalid dataset name '{dataset_name}'. Must be one of {valid_datasets}")
    
    print(f"Downloading {dataset_name} Dataset")
    
    # Aptos Dataset
    if dataset_name == "Aptos":        
        path = kagglehub.dataset_download("mariaherrerot/aptos2019", path=data_path)
        
    # IDRiD Dataset
    elif dataset_name == "IDRiD":        
        file_id = "https://drive.google.com/file/d/1QY2jrzOLf787qH1PrRnohTycAor-UibD/view?usp=sharing"
        zip_path = os.path.join(data_path, "IDRiD_Grading.zip")
        
        # Download
        gdown.download(file_id, output=zip_path,quiet=False)
        
        # Unzip
        print(f"Extracting file {zip_path} ...")
        
        path = os.path.join(data_path, "IDRiD_RAW/")
        with zipfile.ZipFile(zip_path, 'r') as ref:
            ref.extractall(path)
        
        path = os.path.join(path, "B.Disease Grading/")
                
    # DDR Dataset
    elif dataset_name == "DDR":
        path = kagglehub.dataset_download("mariaherrerot/ddrdataset", path=data_path)
        
    # Messidor-2 Dataset
    elif dataset_name == "Messidor-2":
        path = kagglehub.dataset_download("mariaherrerot/messidor2preprocess", path=data_path)
        
    return path


def dataset_prepare(dataset_name, download_path, save_path="./data/"):
    """Merge train/test/val splits into one image folder + one CSV.

    Produces a normalized output directory that build_dr_h5.py expects:
        {save_path}/{dataset_name}/Images/  — all images
        {save_path}/{dataset_name}/labels.csv — columns: id_code, diagnosis

    Image extensions in labels.csv:
        APTOS:    .png appended (original CSV has no extension)
        IDRiD:    .jpg appended (original CSV has no extension)
        DDR:      already included in CSV
        Messidor-2: already included in CSV

    Args:
        dataset_name: One of "Aptos", "IDRiD", "DDR", "Messidor-2".
        download_path: Root of the raw downloaded dataset (from dataset_download).
        save_path: Parent directory for the prepared output.

    Returns:
        Path to the prepared dataset directory.
    """
    dataset_path = os.path.join(save_path, dataset_name)
    images_path = os.path.join(dataset_path, "Images/") 
    
    # Ensure the target image directory exists before copying
    os.makedirs(images_path, exist_ok=True)
    
    print("Preparing Dataset Files...")
    
    # Aptos dataset
    if dataset_name == "Aptos":
        # Merge Images
        shutil.copytree(os.path.join(download_path, "train_images/train_images/"), images_path, dirs_exist_ok=True)
        shutil.copytree(os.path.join(download_path, "test_images/test_images/"), images_path, dirs_exist_ok=True)
        shutil.copytree(os.path.join(download_path, "val_images/val_images/"), images_path, dirs_exist_ok=True)

        # Merge Labels
        df_train = pd.read_csv(os.path.join(download_path, 'train_1.csv'))
        df_test = pd.read_csv(os.path.join(download_path, 'test.csv'))
        df_val = pd.read_csv(os.path.join(download_path, 'valid.csv'))

        # Concatenate all labels
        df = pd.concat([df_train, df_test, df_val], ignore_index=True) 

        # add img extension
        df["id_code"] = df["id_code"].astype(str) + ".png" 

        # Save labels
        df.to_csv(os.path.join(dataset_path, "labels.csv"), index=False)
        
    # IDRiD dataset
    elif dataset_name == "IDRiD":
        # Merge Images
        shutil.copytree(os.path.join(download_path, "1. Original Images/a. Training Set"), images_path, dirs_exist_ok=True)
        shutil.copytree(os.path.join(download_path, "1. Original Images/b. Testing Set"), images_path, dirs_exist_ok=True)
        
        # Merge Labels
        df_train = pd.read_csv(os.path.join(download_path, "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"))
        df_test = pd.read_csv(os.path.join(download_path, "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"))
        
        # Concat training and testing
        df = pd.concat([df_train, df_test], ignore_index=True)
        df = df.rename(columns={'Image name': 'id_code'})
        df = df.rename(columns={'Retinopathy grade': 'diagnosis'})
        
        # add img extension
        df["id_code"] = df["id_code"].astype(str) + ".jpg" 
        
        # Save to csv
        df.to_csv(os.path.join(dataset_path, "labels.csv"), index=False)

    # DDR dataset
    elif dataset_name == "DDR":
        # Copy images
        shutil.copytree(os.path.join(download_path, "DR_grading/DR_grading/"), images_path, dirs_exist_ok=True)

        # Merge Labels
        df = pd.read_csv(os.path.join(download_path, 'DR_grading.csv'))

        # save to csv
        df.to_csv(os.path.join(dataset_path, "labels.csv"), index=False)
    
    # Messidor-2 dataset
    elif dataset_name == "Messidor-2":
        # Merge Images
        shutil.copytree(os.path.join(download_path, "messidor-2/messidor-2/preprocess/"), images_path, dirs_exist_ok=True)

        # Merge Labels
        df = pd.read_csv(os.path.join(download_path, 'messidor_data.csv'))

        # save to csv
        df.to_csv(os.path.join(dataset_path, "labels.csv"), index=False)
        
    return dataset_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download and prepare a DR dataset for build_dr_h5.py")
    p.add_argument("--dataset", type=str, required=True, choices=["Aptos", "IDRiD", "DDR", "Messidor-2"],
                    help="Dataset to download")
    p.add_argument("--data_path", type=str, default="./data/", help="Where to download the raw dataset")
    p.add_argument("--save_path", type=str, default="./data/", help="Where to save the prepared output")
    args = p.parse_args()

    download_path = dataset_download(args.dataset, data_path=args.data_path)
    dataset_path = dataset_prepare(args.dataset, download_path, save_path=args.save_path)

    print(f"\nPrepared dataset at: {dataset_path}")
    print(f"Next, run build_dr_h5.py:")
    print(f"  python data_preparation/build_dr_h5.py \\")
    print(f"      --image_dir {os.path.join(dataset_path, 'Images')} \\")
    print(f"      --csv_path  {os.path.join(dataset_path, 'labels.csv')} \\")
    print(f"      --out_dir   {os.path.join(args.save_path, 'DRGrading')} \\")
    print(f"      --img_size  128")