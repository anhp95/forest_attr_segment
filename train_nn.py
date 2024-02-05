#%%
import os
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from mypath import get_path_train
from model.deep_forest import DeepForestSpecies
from utils.check_accuracy import check_accuracy
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.loader import get_loaders
from utils.loader import load_npy
from PIL import Image

# INIT_LR = 1e-5
# BATCH_SIZE = 16
# NUM_EPOCHS = 50
# NUM_WORKERS = 2
# PIN_MEMORY = True
# LOAD_MODEL = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_ACC = 0.40
FEATURES = [64, 128]


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = DEVICE
        self.features = FEATURES
        self.max_acc = MAX_ACC

        self._form_input()
        self._build_weights()
        self._build_model()
        self._build_loader()
        self._build_logs()

    def _form_input(self):
        if "3d" in self.args.backbone:
            self.img_shape = (13, 4, 32, 32)
        elif "2d" in self.args.backbone:
            if "p1p2p3" in self.args.backbone:
                self.img_shape = (40, 32, 32)
            elif "p1p2" in self.args.backbone:
                self.img_shape = (27, 32, 32)
            else:
                self.img_shape = (14, 32, 32)

    def _build_weights(self):
        if self.args.forest_attr == "spec":
            self.num_classes = 4
            self.weights = [1, 0.153, 0.252, 0.241]
        elif self.args.forest_attr == "age":
            self.num_classes = 3
            self.weights = [1, 0.1, 0.05]

    def _build_model(self):
        self.model = DeepForestSpecies(
            in_channels=self.img_shape[0],
            out_channels=self.num_classes,
            backbone=self.args.backbone,
            features=FEATURES,
        ).to(self.device)

    def _build_loader(self):
        train_img_dir, train_mask_dir, val_img_dir, val_mask_dir = get_path_train(
            self.img_shape, self.args.forest_attr, self.args.backbone
        )
        self.train_loader, self.val_loader = get_loaders(
            train_img_dir,
            train_mask_dir,
            val_img_dir,
            val_mask_dir,
            self.img_shape,
            self.args.batch_size,
            self.args.backbone,
            self.args.no_workers,
            self.args.pin_memory,
        )

    def _build_logs(self):
        _logs_dir = f"logs/{self.args.forest_attr}/{self.args.backbone}"
        if not os.path.isdir(_logs_dir):
            os.makedirs(_logs_dir)
        self.logs_file = os.path.join(
            _logs_dir, f"{self.args.backbone}_{self.args.lr}.csv"
        )

    def train(self):
        if self.args.load_model is not None:
            load_checkpoint(self.args.load_model, self.model)
            check_accuracy(self.val_loader, self.model, DEVICE)

        loss_values = []
        acc_values = []
        f1_values = []
        kappa_values = []
        max_acc = self.max_acc

        for epoch in range(self.args.num_epochs):
            print(f"epoch: {epoch} with lr: {self.args.lr}")
            optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
            scaler = torch.cuda.amp.GradScaler()
            # train
            loop = tqdm(self.train_loader)
            torch.autograd.set_detect_anomaly(True)
            class_weights = torch.FloatTensor(self.weights).to(DEVICE)
            CE_loss = nn.CrossEntropyLoss(weight=class_weights)

            step_loss = []
            for batch_idx, (data, targets) in enumerate(loop):
                data = data.to(DEVICE)
                targets = targets.long().to(DEVICE)

                # forward
                with torch.cuda.amp.autocast():

                    preds = self.model(data)
                    loss = CE_loss(preds, targets)

                # backward
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # update tqdm loop
                loop.set_postfix(loss=loss.item())
                step_loss.append(loss.item())
            step_loss = np.array(step_loss)
            avg_loss = np.sum(step_loss) / len(step_loss)

            loss_values.append(avg_loss)
            print(f"avg epoch loss: {avg_loss}")

            # check accuracy
            acc, f1, kappa = check_accuracy(self.val_loader, self.model, DEVICE)
            acc_values.append(acc)
            f1_values.append(f1)
            kappa_values.append(kappa)

            # save model
            if acc > max_acc:
                max_acc = acc
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(
                    checkpoint,
                    acc,
                    folder=os.path.join(self.args.forest_attr, self.args.backbone),
                )
            if (epoch + 1) % 10 == 0:
                self.args.lr = self.args.lr * 0.5

        logs_df = pd.DataFrame()
        logs_df["loss"] = loss_values
        logs_df["acc"] = acc_values
        logs_df["f1"] = f1_values
        logs_df["kappa"] = kappa_values
        logs_df.loss.plot(label="Loss", legend=True)
        logs_df.acc.plot(secondary_y=True, label="Accuracy", legend=True)

        logs_df.to_csv(self.logs_file)


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Tree Species/Age Segmentation Training"
    )
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
    # training hyper params
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size (default: 16)",
    )
    # optimizer params
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate (default: 1e-5)",
    )
    # checkpoint
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="path to the checkpoint file (default: None)",
    )
    # logs
    parser.add_argument(
        "--logs_file",
        type=str,
        default="logs/",
        help="put the path to the logs directory (default: logs/)",
    )
    # loader params
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="whether use nesterov (default: False)",
    )
    parser.add_argument(
        "--no_workers",
        type=int,
        default=2,
        help="The number of wokers for dataloader (default: 2)",
    )
    args = parser.parse_args()

    # features depth
    if "2d" in args.backbone:
        args.features = [64, 128, 256, 512]  # original UNET 2d feature depth
    elif "3d_org" in args.backbone:
        args.features = [64, 128, 256]  # original UNET 3D feature depth
    else:
        args.features = [64, 128]  # adjusted features depth

    trainer = Trainer(args)
    trainer.train()


# if __name__ == "__main__":
#     main()

# %%
class Args:
    def __init__(self) -> None:
        self.forest_attr = "spec"
        self.backbone = "3d_adj_emd_acb"
        self.num_epochs = 20
        self.lr = 1e-5
        self.batch_size = 16
        self.load_model = None
        self.logs_file = "logs/"
        self.pin_memory = "store_true"
        self.no_workers = 0
        self.features = FEATURES

args = Args()
trainer = Trainer(args)
trainer.train()

# %%
checkpoint_file = "checkpoint/spec/2dp1p2/0.57.pth.tar"

model = DeepForestSpecies(
        in_channels=27,
        out_channels=4,
        backbone=args.backbone,
        features=FEATURES,
    ).to(DEVICE)
load_checkpoint(checkpoint_file, model)

for i in range(100):
    in_npy = f"data/data_train/data_spec_27d_32x32/val/image/{i}.npy"
    mask_npy = f"data/data_train/data_spec_27d_32x32/val/mask/{i}.npy"

    data = torch.from_numpy(np.load(in_npy).reshape(1, -1, 32, 32))
    mask = np.load(mask_npy)

    data = data.to(DEVICE)
    prob_y = F.softmax(model(data), dim=1)
    preds = prob_y.max(1, keepdims=True)[1].cpu().detach().numpy().reshape(32, 32)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(preds)
    axs[1].imshow(mask)
    axs[0].set_title("Prediction")
    axs[1].set_title("Ground Truth")
# %%
