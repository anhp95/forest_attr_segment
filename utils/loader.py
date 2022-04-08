from .dataset import NFDataset
from torch.utils.data import DataLoader
import torch
import numpy as np


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    img_shape,
    batch_size,
    cnn_mode,
    num_workers=4,
    pin_memory=True,
):
    train_ds = NFDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        img_shape=img_shape,
        cnn_mode=cnn_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = NFDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        img_shape=img_shape,
        cnn_mode=cnn_mode,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def load_npy(npy_file, batch_size):

    dataset = torch.from_numpy(np.load(npy_file))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return data_loader
