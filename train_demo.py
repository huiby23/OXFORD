import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from robotcar_dataset_sdk.python.play_radar_new import play_radar_data

# 自定义数据集类
class RadarDataset(Dataset):
    def __init__(self, cart_data, transform=None):
        """
        初始化数据集
        :param cart_data: List[np.ndarray]，存储雷达图像的列表
        :param transform: 可选的数据预处理函数
        """
        self.cart_data = cart_data
        self.transform = transform

    def __len__(self):
        """返回数据集的大小"""
        return len(self.cart_data)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本
        :param idx: 样本索引
        :return: 图像张量（假设是单通道图像）
        """
        image = self.cart_data[idx]  # 获取雷达图像

        # 转换为 PyTorch 张量
        image = torch.from_numpy(image).float()

        # 如果图像是 2D 的（H x W），添加通道维度（C x H x W）
        if image.ndim == 2:
            image = image.unsqueeze(0)  # 添加通道维度

        # 应用数据预处理（如果有）
        if self.transform:
            image = self.transform(image)

        return image


# U-Net 网络结构
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=248):
        """
        U-Net 网络结构
        :param in_channels: 输入通道数（雷达图像为单通道）
        :param out_channels: 输出通道数（描述符的维度）
        """
        super(UNet, self).__init__()

        # 编码器部分
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # 解码器部分
        self.decoder1 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder3 = self.upconv_block(128, 64)

        # 最终输出层
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        # 关键点头部
        self.locations_head = nn.Conv2d(out_channels, 1, kernel_size=1)  # 关键点位置
        self.scores_head = nn.Conv2d(out_channels, 1, kernel_size=1)  # 关键点分数

    def conv_block(self, in_channels, out_channels):
        """卷积块：两个卷积层 + ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        """上采样块：上采样 + 卷积"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        # 解码器
        dec1 = self.decoder1(enc4)
        dec2 = self.decoder2(torch.cat([dec1, enc3], dim=1))
        dec3 = self.decoder3(torch.cat([dec2, enc2], dim=1))

        # 最终输出
        descriptors = self.final(dec3)  # 描述符输出

        # 关键点位置和分数
        locations = self.locations_head(descriptors)  # 关键点位置
        scores = torch.sigmoid(self.scores_head(descriptors))  # 关键点分数

        return descriptors, locations, scores


# 训练循环
def train(model, dataloader, criterion, optimizer, num_epochs=10):
    """训练函数"""
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            # 前向传播
            descriptors, locations, scores = model(batch)

            # 计算损失（假设是自编码器任务）
            loss = criterion(descriptors, batch)  # 假设输入和输出相同

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# 主程序
if __name__ == "__main__":
    # 获取数据集的文件路径
    current_dir = Path(__file__).resolve().parent
    dataset_dir = current_dir / 'dataset' / 'sample_tiny' / '2019-01-10-14-36-48-radar-oxford-10k-partial'
    radar_data_dir = dataset_dir / 'radar'

    if radar_data_dir.exists():
        print(f"{radar_data_dir}路径存在")
    else:
        print(f"{radar_data_dir}路径不存在")


    # 调用play_radar_data()函数，输出雷达数据，并分别存储极坐标系和笛卡尔坐标系的雷达数据
    # display_time是每帧雷达图像输出持续时间，单位为ms，按Esc可以退出图片显示
    # polar_data_list: 存储极坐标系数据的列表，每个元素是一个元组 (timestamps, azimuths, valid, fft_data, radar_resolution)
    # cart_data_list: 存储笛卡尔坐标系数据的列表，每个元素是一个笛卡尔坐标图像 (np.ndarray)

    polar_data, cart_data = play_radar_data(str(radar_data_dir), display_time = 1000)

    # 创建数据集和 DataLoader
    dataset = RadarDataset(cart_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    # 初始化 U-Net 模型
    model = UNet(in_channels=1, out_channels=248)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 假设是回归任务
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train(model, dataloader, criterion, optimizer, num_epochs=10)
