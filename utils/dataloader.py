import os
import cv2
import torch
import shutil
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from typing import AnyStr, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path



class CustomDataset(Dataset):
    def __init__(self, data_root_dir,img_sz, mode='train'):
        self.mode=mode
        self.root_dir = data_root_dir
        self.data_folders = [os.path.join(data_root_dir, folder) for folder in os.listdir(data_root_dir) if os.path.isdir(os.path.join(data_root_dir, folder))]
        self.train_data=self.data_folders[0:int(len(self.data_folders)*0.8)]
        self.test_dara=self.data_folders[int(len(self.data_folders)*0.8):]
        self.transform = transforms.Compose(
                                                [
                                                    transforms.Resize((img_sz, img_sz)),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize(mean=[0.5], std=[0.5])
                                                ]
                                             )
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        else:
            return len(self.test_dara)

    def __getitem__(self, idx):
        if self.mode == 'train':
            folder_path = self.train_data[idx]
        else:
            folder_path = self.test_dara[idx]

        # 导入radar扫描图像
        image_1_path = os.path.join(folder_path, 'image_1.png')
        image_2_path = os.path.join(folder_path, 'image_2.png')
        image_1 = Image.open(image_1_path)
        image_2 = Image.open(image_2_path)
        
        # # 改变数据类型为ndarray
        # image_1 = np.array(image_1)
        # image_2 = np.array(image_2)
        
        # 导入位姿变换矩阵
        pose_tran_path = os.path.join(folder_path, 'pose_tran.npy')
        pose_tran = np.load(pose_tran_path)
        
        # 改变数据类型为tensor
        image_1 = self.transform(image_1)
        image_2 = self.transform(image_2)
        pose_tran = torch.from_numpy(pose_tran).float()
        
        return image_1, image_2, pose_tran



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Load train data from the dataset.')
#     parser.add_argument('--dataset_dir', type=Path, help='Directory containing dataset.')
#     parse\
#             # ......