import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append('core')  # NOQA

from raft import RAFT  # NOQA
from datasets import FlowForwardBackwardDataset  # NOQA
from utils.utils import InputPadder  # NOQA

DEVICE = "cuda"


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def create_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print(f"Failed to create dir: {path}")
    else:
        print(f"Created dir: {path}")


def scale_and_save_flow(flow, out_file):

    assert flow.ndim == 3, "input flow must have three dims"
    assert flow.shape[2], "input flow must have shape [H,W,2]"

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    flow = np.dstack((u, v))

    np.save(out_file, flow)


def run(args):

    in_path_left = os.path.join(args.path, "image_left")
    in_path_right = os.path.join(args.path, "image_right")
    out_path_foward = os.path.join(args.path, "flow_forward")
    out_path_backward = os.path.join(args.path, "flow_backward")

    print("Computing forward and backwrd optical flow between:")
    print(f"    {in_path_left}")
    print(f"    {in_path_right}\n")

    create_dir(out_path_foward)
    create_dir(out_path_backward)

    num_gpus = torch.cuda.device_count()
    batch_size = num_gpus

    data_loader = DataLoader(
        dataset=FlowForwardBackwardDataset(in_path_left, in_path_right),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    # model = model.module ## -> No idea why this was needed
    model.to(DEVICE)
    model.eval()

    print("\nStart ...")
    print(f"Running on {num_gpus} GPUs\n")

    with torch.no_grad():

        for img_str, img_left, img_right in tqdm(data_loader):

            padder = InputPadder(img_left.shape)
            img_left, img_right = padder.pad(img_left, img_right)

            # get foward flow - left -> right
            _, forward_flow_up = model(
                img_left, img_right, iters=20, test_mode=True)

            forward_flow = forward_flow_up.permute(0, 2, 3, 1).cpu().numpy()

            # get backward flow - right -> left
            _, backward_flow_up = model(
                img_right, img_left, iters=20, test_mode=True)
            backward_flow = backward_flow_up.permute(0, 2, 3, 1).cpu().numpy()

            # Loop over the batches
            for i in range(batch_size):
                scale_and_save_flow(forward_flow[i], os.path.join(
                    out_path_foward, img_str[i] + "_flo"))
                scale_and_save_flow(backward_flow[i], os.path.join(
                    out_path_backward, img_str[i] + "_flo"))

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument(
        "--path", help="folder containing the folders image_left and image_right")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision',
                        action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true',
                        help='use efficent correlation implementation')
    args = parser.parse_args()

    run(args)
