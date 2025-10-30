"""
    Data augmentation functions.
"""
import numpy as np
import cv2
import torch
from utils.utils import get_transform


def augmentBatch(batch, config):
    """Rotates the cartesian radar image by a random amount, adjusts the ground truth transform accordingly."""
    rot_max = config['augmentation']['rot_max']
    batch_size = config['batch_size']
    window_size = config['window_size']
    
    data = batch['data'].numpy()
    mask = batch['mask'].numpy()
    T_21 = batch['T_21'].numpy()
    
    B, C, H, W = data.shape
    if B != batch_size * window_size:
        print(f'Batch num is {B}, but batch_size is {batch_size} and window_size is {window_size}.')
        assert IOError

    # 每帧对应一个独立的旋转角度和变换矩阵
    rotations = []
    transforms = []
    inv_transforms = []
    
    for k in range(B):
        rot = np.random.uniform(-rot_max, rot_max)
        rotations.append(rot)

        T = get_transform(0, 0, rot)
        T_inv = get_transform(0, 0, -rot)
        transforms.append(T)
        inv_transforms.append(T_inv)

        img = data[k].squeeze()
        mmg = mask[k].squeeze()

        M = cv2.getRotationMatrix2D((W / 2, H / 2), np.rad2deg(rot), 1.0)
        data[k] = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC).reshape(C, H, W)
        mask[k] = cv2.warpAffine(mmg, M, (W, H), flags=cv2.INTER_CUBIC).reshape(1, H, W)


    # T_21[k] is transform from data[k] to data[k+1]
    for k in range(B - 1):
        R_kp1_inv = inv_transforms[k]
        R_kp2 = transforms[k + 1]
        T_21[k] = R_kp2 @ T_21[k] @ R_kp1_inv

    # 最后一帧 T_21[B - 1] 只做右乘修正
    T_21[B - 1] = T_21[B - 1] @ inv_transforms[B - 1]
    
    
    batch['data'] = torch.from_numpy(data)
    batch['mask'] = torch.from_numpy(mask > 0.5).type(batch['data'].dtype)    # make into a binary mask
    batch['T_21'] = torch.from_numpy(T_21)
    
    return batch
