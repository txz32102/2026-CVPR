import cv2
import numpy as np

def prepare_oom_mr_image(path='/home/data1/musong/data/fastmri/diffusion_images/file_brain_AXT2_208_2080298_3.jpg'):
    # --- read image ---
    img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2RGB)
    img_gt = cv2.resize(img_gt, (256, 256), interpolation=cv2.INTER_AREA)

    # --- normalize to [0, 1] ---
    img_gt = img_gt.astype(np.float32) / 255.0

    # --- generate noisy version (LQ) ---
    np.random.seed(0)
    noise = np.random.normal(0, 50 / 255., img_gt.shape)
    img_lq = np.clip(img_gt + noise, 0, 1)

    return img_lq, img_gt

def get_image_pair(path='/home/data1/musong/workspace/python/SwinIR/testsets/McMaster/1.tif'):
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    np.random.seed(seed=0)
    img_lq = img_gt + np.random.normal(0, 50 / 255., img_gt.shape)
    img_lq = cv2.resize(img_lq, (256, 256), interpolation=cv2.INTER_AREA)
    img_gt = cv2.resize(img_gt, (256, 256), interpolation=cv2.INTER_AREA)
    return img_lq, img_gt