import torch
import numpy as np
from PIL import Image
from einops import rearrange
import os


''' 
    Denormalizes a tensor of images.
    img_t: [C, H, W]
    Output: [C, H, W]
'''
def denormalize_t(img_t, mean, std):
    try:
        assert(len(mean) == len(std))
        assert(len(mean) == img_t.shape[0])
    except:
        print(f'Unmatched channels between image tensors and normalization mean and std. Got {img_t.shape[0]}, {len(mean)}, {len(std)}.')
    img_denorm = torch.empty_like(img_t)
    for t in range(img_t.shape[0]):
        img_denorm[t, :, :] = (img_t[t, :, :] * std[t]) + mean[t]
    return img_denorm


''' 
    Denormalizes a batch of tensors of images.
    img_t: [B, C, H, W]
    Output: [B, C, H, W]
'''
def denormalize_batch_t(img_t, mean, std):
    try:
        assert(len(mean) == len(std))
        assert(len(mean) == img_t.shape[1])
    except:
        print(f'Unmatched channels between image tensors and normalization mean and std. Got {img_t.shape[1]}, {len(mean)}, {len(std)}.')
    img_denorm = torch.empty_like(img_t)
    for t in img_t.shape[1]:
        img_denorm[:, t, :, :] = (img_t[:, t, :, :].clone() * std[t]) + mean[t]
    return img_denorm


''' 
    Change a ndarray to a pil image.
    img_nd: [H, W, C]    C in {1, 3}
    img_nd values in [0, 255]
    output: Image in PIL
'''
def aligned_nd_2_pil(img_nd):
    img = np.uint8(img_nd)
    return Image.fromarray(img)


''' 
    Change a ndarray to a pil image.
    img_nd: [H, W, C]    C in {1, 3}
    img_nd values in [0., 1.]
    output: Image in PIL
'''
def normalized_nd_2_pil(img_nd):
    img = np.uint8(img_nd * 255.)
    return Image.fromarray(img)


'''
    Check whether the path exists. If not, then create the path.
'''
def check_or_create_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)


'''
    save a single img.
    img: either in tensor or ndarray type.
    path: the output folder path.
    name: the output file name.
    aligned: set True when img_nd in [0, 255], False when in [0., 1.]
    reshape_format: the formatted string used in rearrange from einops
'''
def save_single_img(img, path, name, aligned=False, reshape_format=None):
    if torch.is_tensor(img):
        img = img.cpu().detach().numpy()
    if reshape_format:
        img = rearrange(img, reshape_format)
    if img.shape[-1] == 1:
        img = np.squeeze(img, axis=-1)    # squeeze the last axis for grayscale
    img_pil = aligned_nd_2_pil(img) if aligned else normalized_nd_2_pil(img)
    check_or_create_path(path)
    img_pil.save(os.path.join(path, name))


'''
    save a single img. The image requires denormalizing at first.
    img: in tensor type.
    path: the output folder path.
    name: the output file name.
    aligned: set True when img_nd in [0, 255], False when in [0., 1.]
    reshape_format: the formatted string used in rearrange from einops
'''
def save_single_normalized_img(img, path, name, mean, std, aligned=False, reshape_format=None):
    img_denorm = denormalize_t(img, mean, std)
    save_single_img(img_denorm, path, name, aligned=aligned, reshape_format=reshape_format)