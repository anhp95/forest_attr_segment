#%%
import os
import numpy as np
import torch
import torch.nn.functional as F
import argparse

from forest import gen_predicted_map
from model.deep_forest import DeepForestSpecies
from tqdm import tqdm
from utils.checkpoint import load_checkpoint
from recls import recls

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

IMG_SHAPE = (13, 4, 32, 32)
# IMG_SHAPE = (40, 32, 32)  # 27, 14
NUM_CLASSES = 4
INFER_DIR = "data/data_infer"


def load_npy(npy_file, batch_size=BATCH_SIZE):

    dataset = torch.from_numpy(np.load(npy_file))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return data_loader


def predict(params):

    forest_attr = params["forest_attr"]
    acc = params["acc"]
    region = params["region"]
    backbone = params["backbone"]
    features = params["features"]

    checkpoint_file = f"checkpoint/{forest_attr}/{backbone}/0.{acc}.pth.tar"
    in_npy = f"{INFER_DIR}/input/{region}_13b.npy"
    out_dir = f"{INFER_DIR}/output/{forest_attr}/{backbone}"
    out_npy = os.path.join(out_dir, f"{region}_{acc}.npy")

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    model = DeepForestSpecies(
        in_channels=IMG_SHAPE[0],
        out_channels=NUM_CLASSES,
        backbone=backbone,
        features=features,
    ).to(DEVICE)
    load_checkpoint(checkpoint_file, model)

    data_loader = load_npy(in_npy)
    loop = tqdm(data_loader)

    spec_arr = []
    for data in loop:
        data = data.to(DEVICE)
        prob_y = F.softmax(model(data), dim=1)
        preds = prob_y.max(1, keepdims=True)[1]
        spec_arr.append(preds)
    spec = torch.cat(spec_arr)

    spec = spec.to("cpu")
    print("----writing predicted npy-----")
    np.save(out_npy, spec)

    return out_npy


def main():

    # parser = argparse.ArgumentParser(
    #     description="Generating High-Resolution Tree Species Map"
    # )
    forest_attr = "spec"
    backbone = "3d_adj_emd_acb"
    region = "toki"
    features = [64, 128]
    acc = 7780

    n_cls = 7  # 10-ena, 7-nakat,

    low_res_tif = (
        f"data/spec_map/{region}/low-res/{region}_{forest_attr}_{backbone}_{acc}.tif"
    )
    high_res_tif = f"data/spec_map/{region}/high-res/{region}_{forest_attr}_{backbone}_{acc}_km{n_cls}.tif"

    l2_img_dir = fr"D:\co2_data\DL\large_img\sentinel\s2_{region}_recls\l2"

    params = {
        "forest_attr": forest_attr,
        "acc": acc,
        "region": region,
        "backbone": backbone,
        "features": features,
    }
    # pred_npy = predict(params)

    # gen_predicted_map(region, pred_npy, low_res_tif, forest_attr)
    recls(low_res_tif, high_res_tif, l2_img_dir, n_cls=n_cls)


main()
# %%

# %%
