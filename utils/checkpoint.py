import os
import torch


def save_checkpoint(state, acc, folder):
    print("=> Saving checkpoint")
    check_point_dir = os.path.join("./checkpoint/", folder)
    if not os.path.isdir(check_point_dir):
        os.makedirs(check_point_dir)
    filename = os.path.join(check_point_dir, f"{acc:.2f}.pth.tar")
    torch.save(state, filename)


def load_checkpoint(checkpoint_file, model):
    checkpoint = torch.load(checkpoint_file)
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
