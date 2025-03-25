import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class Stenosis_Dataset(Dataset):
    def __init__(self, mode="train"):
        data_root = f"stenosis_data/{mode}"
        file_names = os.listdir(f"stenosis_data/{mode}/images")
        self.imgs = [pil_to_tensor(Image.open(f"{data_root}/images/{file_name}"))[0, :, :].unsqueeze(0) / 255 for file_name in file_names]
        self.masks = [pil_to_tensor(Image.open(f"{data_root}/masks/{file_name}")).squeeze(0) / 255 for file_name in file_names]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx].float(), self.masks[idx].long()