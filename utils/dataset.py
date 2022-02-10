#%%
import os
from torch.utils.data import Dataset
import numpy as np


class NFDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_shape, cnn_mode):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # self.transform = transform
        self.images = os.listdir(image_dir)
        self.img_shape = img_shape
        self.cnn_mode = cnn_mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = np.load(img_path)
        image = image.reshape(self.img_shape)

        if self.cnn_mode == "2d":  # 2d cnn
            mask = np.load(mask_path)
            # mask = _relabel_spec(mask)
        elif "3d" in self.cnn_mode:  # # 3d cnn
            mask = np.load(mask_path)
            mask = np.stack([mask for _ in range(0, self.img_shape[1])])

        return image, mask

    def _relabel_spec(arr):
        """
        beech -> broadleaf, birch -> broadleaf, larch -> conifer, fir -> conifer
        1-> 3, 8 -> 3, 4 -> 5, 6 -> 5
        2->0, sugi
        3->1, broadleaf
        5->2, conifer
        7->3, cypress
        """
        arr[arr == 1] = 3
        arr[arr == 8] = 3

        arr[arr == 4] = 5
        arr[arr == 6] = 5

        arr[arr == 2] = 0
        arr[arr == 3] = 1
        arr[arr == 5] = 2
        arr[arr == 7] = 3
        return arr
