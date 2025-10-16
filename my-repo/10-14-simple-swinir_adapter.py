from models.network_swinir import *
import os
from collections import OrderedDict
from datetime import datetime
import json
import re
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def prepare_oom_mr_image(path):
    """
    Read an MR image, convert to RGB, resize to 500×500, 
    and generate a noisy low-quality version.
    Returns (imgname, img_lq, img_gt)
    """
    # --- get name ---
    imgname, _ = os.path.splitext(os.path.basename(path))

    # --- read image ---
    img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2RGB)

    # --- resize to 500×500 ---
    img_gt = cv2.resize(img_gt, (256, 256), interpolation=cv2.INTER_AREA)

    # --- normalize to [0, 1] ---
    img_gt = img_gt.astype(np.float32) / 255.0

    # --- generate noisy version (LQ) ---
    np.random.seed(0)
    noise = np.random.normal(0, 50 / 255., img_gt.shape)
    img_lq = np.clip(img_gt + noise, 0, 1)

    return imgname, img_lq, img_gt

def get_image_pair(path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    np.random.seed(seed=0)
    img_lq = img_gt + np.random.normal(0, 50 / 255., img_gt.shape)

    return imgname, img_lq, img_gt

ckpt_path = "/home/data1/musong/workspace/python/SwinIR/model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
folder_gt = "/home/data1/musong/workspace/python/SwinIR/testsets/McMaster"
mr_img_path = '/home/data1/musong/data/fastmri/diffusion_images/file_brain_AXT2_208_2080298_3.jpg'
border = 0
window_size = 8
device = 'cuda:1'

model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2, upsampler='', resi_connection='1conv')

pretrained_model = torch.load(ckpt_path)
param_key_g = 'params'
model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
model = model.to(device)

data = get_image_pair(f'{folder_gt}/1.tif')
img_lq = data[1]
img_gt = data[2]

_, img_lq, img_gt = prepare_oom_mr_image(mr_img_path)

img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1)) 
img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device) 
img_gt = np.transpose(img_gt if img_gt.shape[2] == 1 else img_gt[:, :, [2, 1, 0]], (2, 0, 1)) 
img_gt = torch.from_numpy(img_gt).float().unsqueeze(0)

with torch.no_grad():
    output = model(img_lq)

print(output.shape)