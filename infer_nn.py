#%%
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from model.deep_forest import DeepForestSpecies
from recls import ReClassification

from forest import gen_predicted_map
from mypath import get_path_infer
from utils.checkpoint import load_checkpoint
from utils.loader import load_npy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

IMG_SHAPE_3D = (13, 4, 32, 32)
FEATURES_3D = [64, 128]
ACC_3D = 82  # 7780

# If you use 2D CNN, define your params here
IMG_SHAPE_2D = (30, 32, 32)
FEATURES_2D = [64, 128, 256, 512]
ACC_2D = ""

NUM_CLASSES_SPECIES = 4
NUM_CLASSES_AGE = 3

INFER_DIR = "data/data_infer"


class Recls:
    def __init__(self, args):
        self.region = args.region
        self._set_n_clusters()

    def _set_n_clusters(self):
        if self.region == "ena":
            self.n_clusters = 10
        elif self.region == "nakat":
            self.n_clusters = 5
        elif self.region == "tono":
            self.n_clusters = 5

    def recls(self):
        pass


class Inference:
    def __init__(self, args):
        self.forest_attr = args.forest_attr
        self.backbone = args.backbone
        self.region = args.region
        self.batch_size = args.batch_size

        self._set_params()

    def _set_params(self):

        if self.forest_attr == "spec":
            self.num_classes = NUM_CLASSES_SPECIES
        else:
            self.num_classes = NUM_CLASSES_AGE

        if "3d" in self.backbone:
            self.features = FEATURES_3D
            self.img_shape = IMG_SHAPE_3D
            self.acc = ACC_3D
        else:
            self.features = FEATURES_2D
            self.img_shape = IMG_SHAPE_2D
            self.acc = ACC_2D

    def predict(self):

        checkpoint_file = (
            f"checkpoint/{self.forest_attr}/{self.backbone}/0.{self.acc}.pth.tar"
        )
        in_npy = f"{INFER_DIR}/input/{self.region}_13b.npy"
        out_dir = f"{INFER_DIR}/output/{self.forest_attr}/{self.backbone}"
        out_npy = os.path.join(out_dir, f"{self.region}_{self.acc}.npy")

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        model = DeepForestSpecies(
            in_channels=self.img_shape[0],
            out_channels=self.num_classes,
            backbone=self.backbone,
            features=self.features,
        ).to(DEVICE)
        load_checkpoint(checkpoint_file, model)

        data_loader = load_npy(in_npy, BATCH_SIZE)
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

    parser = argparse.ArgumentParser(description="Generating Forest Atrribute Map")
    parser.add_argument(
        "--forest_attr",
        type=str,
        default="spec",
        choices=["spec", "age"],
        help="which forest attribute is going to be segmented (default: spec)",
    )
    # model params
    parser.add_argument(
        "--backbone",
        type=str,
        default="3d_adj_emd_acb",
        choices=[
            "2d_p2",
            "2d_p1p2",
            "2d_p1p2p3",
            "3d_org",
            "3d_adj",
            "3d_adj_dec_acb",
            "3d_adj_emd_acb",
            "3d_org_emd_acb",
        ],
        help="backbone of the model (default: 3d_adj_emd_acb)",
    )
    # loader params
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size for load input npy data to infer",
    )
    # experiment params
    parser.add_argument(
        "--region",
        type=str,
        default="ena",
        choices=[
            "ena",
            "nakat",
            "mizunami",
            "toki",
            "tajimi",
            "tono",
        ],
        help="region of interest - ROI (default: ena)",
    )
    # reclassification params
    parser.add_argument(
        "--recls",
        type=bool,
        default=0,
        choices=[0, 1],
        help="0: no reclassification, 1: apply reclassification (default: 0)",
    )
    parser.add_argument(
        "--n_clusters", type=int, help="number of clusters for reclassification step"
    )

    # parsing arguments
    args = parser.parse_args()

    if args.region == "ena":
        args.n_clusters = 10
    elif args.region == "nakat":
        args.n_cluster = 7
    elif args.region == "tono":
        args.n_cluster = 5

    # main

    # infer low-res forest attribute
    infer_obj = Inference(args)

    low_res, input_s2_tifs, out_tif_recls = get_path_infer(infer_obj, args)

    pred_npy = infer_obj.predict()
    gen_predicted_map(args.region, pred_npy, low_res, args.forest_attr)

    # reclassify to improve map resolution
    if args.recls and args.forest_attr == "spec":

        recls = ReClassification(low_res, input_s2_tifs, out_tif_recls)
        recls.reclassify()


if __name__ == "__main__":
    main()
