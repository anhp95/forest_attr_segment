#%%
import os
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

from model.deep_forest import DeepForestSpecies
from utils.check_accuracy import check_accuracy
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.loader import get_loaders
from mypath import get_path

# INIT_LR = 1e-5
# BATCH_SIZE = 16
# NUM_EPOCHS = 50
# NUM_WORKERS = 2
# PIN_MEMORY = True
# LOAD_MODEL = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_ACC = 0.77
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
            self.img_shape = (40, 32, 32)

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
        train_img_dir, train_mask_dir, val_img_dir, val_mask_dir = get_path(
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
            load_checkpoint(torch.load(self.args.load_model), self.model)
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
        # logs_df.loss.plot(label="Loss", legend=True)
        # logs_df.acc.plot(secondary_y=True, label="Accuracy", legend=True)

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
        default="3d_enc_mid_dec_acb",
        choices=["2d", "3d_org", "3d_adj", "3d_dec_acb", "3d_enc_mid_dec_acb"],
        help="backbone of the model (default: 3d_enc_mid_dec_acb)",
    )
    parser.add_argument(
        "--features",
        type=list,
        default=[64, 128],
        choices=[[64, 128], [64, 128, 256, 512]],
        help="features depth of the model (default: [64, 128])",
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
        help="put the path to the checkpoint file (default: None)",
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

    # set features depth
    if args.backbone == "2d":
        args.features = [64, 128, 256, 512]  # original unet 2d
    elif "3d" in args.backbone:
        args.features = [64, 128]

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()

# %%
