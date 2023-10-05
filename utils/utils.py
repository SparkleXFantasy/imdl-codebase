import numpy as np
import os
import random
import torch
from matplotlib import pyplot as plt
from torch import nn

def set_random_seed(seed_value):
    seed_value = seed_value
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True


def grayscale_to_color(gray_img, colormap='jet'):
    # Normalize grayscale image from numpy to [0, 1]
    eps = 1e-8
    norm_img = (gray_img - np.min(gray_img)) / (np.max(gray_img) - np.min(gray_img) + eps)

    # Use colormap to create color image
    cmap = plt.get_cmap(colormap)
    color_img = cmap(norm_img)[:, :, :3]  # Keep only RGB channels, ignore alpha channel if present

    return (color_img * 255).astype(np.uint8)