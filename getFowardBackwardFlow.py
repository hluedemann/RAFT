import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from raft import RAFT
from utils.utils import InputPadder
from tqdm import tqdm

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
    out_path_foward = os.path.join(args.path, "flow_foward")
    out_path_backward = os.path.join(args.path, "flow_backward")

    print("Computing forward and backwrd optical flow between:")
    print(f"    {in_path_left}")
    print(f"    {in_path_right}\n")
    
    create_dir(out_path_foward)
    create_dir(out_path_backward)

    images_left = glob.glob(os.path.join(in_path_left, "*.jpg")) + \
        glob.glob(os.path.join(in_path_left, "*.png"))
    
    images_right = glob.glob(os.path.join(in_path_right, "*.jpg")) + \
        glob.glob(os.path.join(in_path_right, "*.png"))

    images_left = sorted(images_left)
    images_right = sorted(images_right)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    print("Start ... \n")

    with torch.no_grad():

        for imfile1, imfile2 in tqdm(zip(images_left, images_right), total=len(images_left)):

            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # get foward flow - left -> right
            forward_flow_low, forward_flow_up = model(image1, image2, iters=20, test_mode=True)
            forward_flow = forward_flow_up[0].permute(1,2,0).cpu().numpy()

            # get backward flow - right -> left
            backward_flow_low, backward_flow_up = model(image2, image1, iters=20, test_mode=True)
            backward_flow = backward_flow_up[0].permute(1,2,0).cpu().numpy()

            
            file_str = imfile1.split("/")[-1].split(".")[0]

            scale_and_save_flow(forward_flow, os.path.join(out_path_foward, file_str + "flo"))
            scale_and_save_flow(backward_flow, os.path.join(out_path_backward, file_str + "flo"))

    print("Done!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--path", help="folder containing the folders image_left and image_right")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    run(args)
