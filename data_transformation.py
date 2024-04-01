import argparse
import logging
import os
import pickle
import sys
import tempfile
from glob import glob
import json
import mlflow
import monai
import nibabel as nib
import numpy as np
import torch
from monai.data import (
    CacheDataset,
    DataLoader,
    decollate_batch,
    load_decathlon_datalist,
    set_track_meta,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter


def transform_function():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    monai.utils.set_determinism(seed=123)
    torch.backends.cudnn.benchmark = True
    # define path
    data_file_base_dir = os.path.join(os.getcwd(),"workspace/data/spleen/Task09_Spleen")
    data_list_file_path = os.path.join(os.getcwd(),"workspace/data/spleen/Task09_Spleen/dataset.json")
    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # data loader
    train_datalist = load_decathlon_datalist(
        data_list_file_path,
        is_segmentation=True,
        data_list_key="training",
        base_dir=data_file_base_dir,
    )

    train_ds = CacheDataset(
        data=train_datalist[: int(0.8 * len(train_datalist))],
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=4,
    )
    val_ds = CacheDataset(
        data=train_datalist[int(0.8 * len(train_datalist)) :],
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=4,
)
    cwd = os.getcwd()
    tr_data = os.path.join(cwd,'train_dataset.pkl')
    val_data = os.path.join(cwd,'valid_dataset.pkl')
    with open(tr_data, 'wb') as f:
        pickle.dump(train_ds, f,protocol=pickle.HIGHEST_PROTOCOL)
    with open(val_data, 'wb') as f:
        pickle.dump(val_ds, f,protocol=pickle.HIGHEST_PROTOCOL)
    return "xo"

transform_function()