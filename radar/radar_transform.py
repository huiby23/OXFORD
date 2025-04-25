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
    _, C, H, W = data.shape

    for i in range(batch_size):
        rot = np.random.uniform(-rot_max, rot_max)
        T = get_transform(0, 0, rot)
        for j in range(1, window_size):
            k = j + i * window_size
            img = data[k].squeeze()
            mmg = mask[k].squeeze()
            
            M = cv2.getRotationMatrix2D((W / 2, H / 2), rot * 180 * j / np.pi, 1.0)
            data[k] = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC).reshape(C, H, W)
            mask[k] = cv2.warpAffine(mmg, M, (W, H), flags=cv2.INTER_CUBIC).reshape(1, H, W)
            
            T_21[k - 1] = np.matmul(T, T_21[k - 1])
    
    batch['data'] = torch.from_numpy(data)
    batch['mask'] = torch.from_numpy(mask > 0.5).type(batch['data'].dtype)    # make into a binary mask
    batch['T_21'] = torch.from_numpy(T_21)
    
    return batch
