import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_GRT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        bilinear = config['networks']['unet']['bilinear']
        first_feature_dimension = config['networks']['unet']['first_feature_dimension']
        self.score_sigmoid = config['networks']['unet']['score_sigmoid']
        self.descriptor_relu = config['networks']['unet']['descriptor_relu']
        
        # down
        input_channels = 1
        self.inc = DoubleConv_old(input_channels, first_feature_dimension)
        self.down1 = Down(first_feature_dimension, first_feature_dimension * 2)
        self.down2 = Down(first_feature_dimension * 2, first_feature_dimension * 4)
        self.down3 = Down(first_feature_dimension * 4, first_feature_dimension * 8)
        self.down4 = Down(first_feature_dimension * 8, first_feature_dimension * 16)

        # self.up1_pts = Up_old(first_feature_dimension * (16 + 8), first_feature_dimension * 8, bilinear)
        # self.up2_pts = Up_old(first_feature_dimension * (8 + 4), first_feature_dimension * 4, bilinear)
        self.up3_pts = Up_old(first_feature_dimension * (4 + 2), first_feature_dimension * 2, bilinear)
        self.up4_pts = Up_old(first_feature_dimension * (2 + 1), first_feature_dimension * 1, bilinear)
        self.outc_pts = OutConv(first_feature_dimension, 1)
        
        self.up1_score = Up(first_feature_dimension * (16 + 8), first_feature_dimension * 8, bilinear)
        self.up2_score = Up(first_feature_dimension * (8 + 4), first_feature_dimension * 4, bilinear)
        self.up3_score = Up(first_feature_dimension * (4 + 2), first_feature_dimension * 2, bilinear)
        self.up4_score = Up(first_feature_dimension * (2 + 1), first_feature_dimension * 1, bilinear)
        self.outc_score = OutConv(first_feature_dimension, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
        self.feature_aspp4 = ASPP(first_feature_dimension * 4, first_feature_dimension * 4)
        self.feature_aspp5 = ASPP(first_feature_dimension * 8, first_feature_dimension * 8)
        self.feature_relu = PReLU()

        
    def forward(self, x):
        _, _, height, width = x.size()
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # x4_up_pts = self.up1_pts(x5, x4)
        # x3_up_pts = self.up2_pts(x4_up_pts, x3)
        x2_up_pts = self.up3_pts(x3, x2)
        x1_up_pts = self.up4_pts(x2_up_pts, x1)
        detector_scores = self.outc_pts(x1_up_pts)

        x4_up_score = self.up1_score(x5, x4)
        x3_up_score = self.up2_score(x4_up_score, x3)
        x2_up_score = self.up3_score(x3_up_score, x2)
        x1_up_score = self.up4_score(x2_up_score, x1)
        weight_scores = self.outc_score(x1_up_score)
        if self.score_sigmoid:
            weight_scores = self.sigmoid(weight_scores)
        
        x4_aspp = self.feature_aspp4(x4)
        x5_aspp = self.feature_aspp4(x5)
        
        f1 = F.interpolate(x1, size=(height, width), mode='bilinear')
        f2 = F.interpolate(x2, size=(height, width), mode='bilinear')
        f3 = F.interpolate(x3, size=(height, width), mode='bilinear')
        f4 = F.interpolate(x4_aspp, size=(height, width), mode='bilinear')
        f5 = F.interpolate(x5_aspp, size=(height, width), mode='bilinear')

        feature_list = [f1, f2, f3, f4, f5]
        descriptors = torch.cat(feature_list, dim=1)
        if self.descriptor_relu:
            descriptors = self.feature_relu(descriptors)
        
        return detector_scores, weight_scores, descriptors



class DoubleConv_old(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up_old(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            d = int(pow(2, np.floor(np.log(in_channels) / np.log(2))))
            self.up = nn.ConvTranspose2d(d, d, kernel_size=2, stride=2)
        self.conv = DoubleConv_old(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is NCHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2, with Gaussian Transformer Layer (GTL)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.gtl = GaussianTransformerLayer(out_channels)  # Add GTL for attention-based feature extraction

    def forward(self, x):
        x = self.double_conv(x)
        return self.gtl(x)  # Apply GTL after convolution


class Down(nn.Module):
    """Downscaling with maxpool then double conv, with attention sampling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.attention_downsample = AttentiveDownsampling(out_channels)  # Add attentive downsampling module

    def forward(self, x):
        x = self.maxpool_conv(x)
        return self.attention_downsample(x)  # Apply attention during downsampling


class Up(nn.Module):
    """Upscaling then double conv, with attention sampling"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            d = int(pow(2, np.floor(np.log(in_channels) / np.log(2))))
            self.up = nn.ConvTranspose2d(d, d, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.attention_upsample = AttentiveUpsampling(out_channels)  # Add attentive upsampling module

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is NCHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return self.attention_upsample(x)  # Apply attention during upsampling

    
    
class GaussianTransformerLayer(nn.Module):
    """A custom Gaussian Transformer Layer to replace softmax"""
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.MultiheadAttention(in_channels, num_heads=4)

    def forward(self, x):
        # Reshape to fit multihead attention
        x = x.flatten(2).transpose(0, 2)  # Flatten spatial dimensions
        x, _ = self.attention(x, x, x)
        return x.transpose(0, 2).reshape_as(x)  # Return back to the original shape


class AttentiveDownsampling(nn.Module):
    """Attention-based downsampling layer"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.attention = nn.Softmax(dim=1)  # Attention across channels

    def forward(self, x):
        x = self.conv(x)
        return self.attention(x)


class AttentiveUpsampling(nn.Module):
    """Attention-based upsampling layer"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.attention = nn.Softmax(dim=1)  # Attention across channels

    def forward(self, x):
        x = self.conv(x)
        return self.attention(x)


class OutConv(nn.Module):
    """Output 1x1 convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super(PReLU, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, x):
        return F.prelu(x, self.alpha)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, d=1):
        super().__init__()
        if p is None:
            p = (k // 2) * d
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ASPP(nn.Module):
    def __init__(self, in_ch = 512, out_ch = 256, rates=(1, 6, 12, 18)):
        super().__init__()
        self.atrous_block = nn.ModuleList([
            nn.Sequential(ConvBNReLU(in_ch, out_ch, k=1, s=1, p=0)),
            ConvBNReLU(in_ch, out_ch, k=3, d=rates[1]),
            ConvBNReLU(in_ch, out_ch, k=3, d=rates[2]),
            ConvBNReLU(in_ch, out_ch, k=3, d=rates[3]),
        ])
        self.pool_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.outc = ConvBNReLU(out_ch * 5, out_ch, k=1, p=0)
        
    def forward(self, x):
        size = x.shape[-2:]
        feats = [b(x) for b in self.atrous_block]
        
        img = self.pool_block(x)
        img = F.interpolate(img, size=size, mode='bilinear', align_corners=False)
        feats.append(img)
        
        x = torch.cat(feats, dim=1)
        
        return self.outc(x)