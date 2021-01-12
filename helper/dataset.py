import sys
sys.path.append('../core')

import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import InputPadder

DEVICE = "cuda"

class FlowForwardBackwardDataset(Dataset):

    def __init__(self, in_dir_left, in_dir_right):
        super(FlowForwardBackwardDataset, self).__init__()

        self.in_dir_left = in_dir_left
        self.in_dir_right = in_dir_right

        self.images_left = glob.glob(os.path.join(self.in_dir_left, "*.jpg")) + \
            glob.glob(os.path.join(in_dir_left, "*.png"))
    
        self.images_right = glob.glob(os.path.join(in_dir_right, "*.jpg")) + \
            glob.glob(os.path.join(in_dir_right, "*.png"))

        self.images_left = sorted(self.images_left)
        self.images_right = sorted(self.images_right)

        assert len(self.images_left) == len(self.images_right), \
            f"number of images in {self.in_dir_left} and {self.in_dir_right} not the same"

    def __len__(self):
        return len(self.images_left)

    def __getitem__(self, item):
        path_left = self.images_left[item]
        path_right = self.images_right[item]

        file_str = path_left.split("/")[-1].split(".")[0]

        img_left = np.array(Image.open(path_left)).astype(np.uint8)
        img_left = torch.from_numpy(img_left).permute(2, 0, 1).float()
        img_right = np.array(Image.open(path_right)).astype(np.uint8)
        img_right = torch.from_numpy(img_right).permute(2, 0, 1).float()

        img_left = img_left[None].to(DEVICE)
        img_right = img_right[None].to(DEVICE)

        padder = InputPadder(img_left.shape)

        img_left, img_right = padder.pad(img_left, img_right)

        return file_str, img_left, img_right


