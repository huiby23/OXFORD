import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic residual block for HRNet"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """Bottleneck residual block for HRNet"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HRNet(torch.nn.Module):
    """HRNet backbone with UNet-style outputs"""
    def __init__(self, config):
        super().__init__()
        first_feature_dimension = config['networks']['unet']['first_feature_dimension']
        self.score_sigmoid = config['networks']['unet']['score_sigmoid']
        
        # Stem network
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1 - single branch
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        
        # Stage 2 - two branches
        self.transition1 = self._make_transition_layer([256], [first_feature_dimension, first_feature_dimension*2])
        self.stage2, pre_stage_channels = self._make_stage(
            num_branches=2, 
            num_blocks=4, 
            num_channels=[first_feature_dimension, first_feature_dimension*2]
        )
        
        # Stage 3 - three branches  
        self.transition2 = self._make_transition_layer(pre_stage_channels, [first_feature_dimension, first_feature_dimension*2, first_feature_dimension*4])
        self.stage3, pre_stage_channels = self._make_stage(
            num_branches=3, 
            num_blocks=4, 
            num_channels=[first_feature_dimension, first_feature_dimension*2, first_feature_dimension*4]
        )
        
        # Stage 4 - four branches
        self.transition3 = self._make_transition_layer(pre_stage_channels, [first_feature_dimension, first_feature_dimension*2, first_feature_dimension*4, first_feature_dimension*8])
        self.stage4, pre_stage_channels = self._make_stage(
            num_branches=4, 
            num_blocks=4, 
            num_channels=[first_feature_dimension, first_feature_dimension*2, first_feature_dimension*4, first_feature_dimension*8]
        )
        
        # Last layers for multi-scale feature fusion (HRNetV2 style)
        last_inp_channels = np.int(np.sum(pre_stage_channels))
        self.last_layer = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output heads
        self.detector_head = nn.Conv2d(last_inp_channels, 1, kernel_size=3, padding=1)
        self.weight_head = nn.Conv2d(last_inp_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        
        # For descriptors - we'll use features from multiple resolutions
        self.descriptor_channels = last_inp_channels

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i],
                                 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, num_branches, num_blocks, num_channels):
        blocks = []
        for i in range(num_branches):
            blocks.append(
                self._make_layer(BasicBlock, num_channels[i], num_channels[i], num_blocks)
            )
        return nn.ModuleList(blocks), num_channels

    def forward(self, x):
        """Forward pass of HRNet"""
        _, _, H, W = x.size()
        
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 1
        x = self.layer1(x)
        
        # Stage 2
        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = []
        for i in range(2):
            y_list.append(self.stage2[i](x_list[i]))
        
        # Stage 3
        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = []
        for i in range(3):
            y_list.append(self.stage3[i](x_list[i]))
        
        # Stage 4
        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = []
        for i in range(4):
            y_list.append(self.stage4[i](x_list[i]))
        
        # HRNetV2 style fusion - upsample all branches to highest resolution
        h, w = y_list[0].size(2), y_list[0].size(3)
        y0 = y_list[0]
        y1 = F.interpolate(y_list[1], size=(h, w), mode='bilinear', align_corners=True)
        y2 = F.interpolate(y_list[2], size=(h, w), mode='bilinear', align_corners=True) 
        y3 = F.interpolate(y_list[3], size=(h, w), mode='bilinear', align_corners=True)
        
        # Concatenate all branches
        x = torch.cat([y0, y1, y2, y3], 1)
        x = self.last_layer(x)
        
        # Upsample to input resolution
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        # Generate outputs
        detector_scores = self.detector_head(x)
        weight_scores = self.weight_head(x)
        
        if self.score_sigmoid:
            weight_scores = self.sigmoid(weight_scores)
        
        # Use the fused features as descriptors
        descriptors = x
        
        return detector_scores, weight_scores, descriptors

# 计算参数量的辅助函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 使用示例
if __name__ == "__main__":
    # 模拟配置
    config = {
        'networks': {
            'unet': {
                'first_feature_dimension': 32,  # 可以调整这个值来控制模型大小
                'bilinear': True,
                'score_sigmoid': True
            }
        }
    }
    
    # 创建模型
    model = HRNet(config)
    
    # 计算参数量
    total_params = count_parameters(model)
    print(f"总参数量: {total_params:,}")
    print(f"总参数量 (MB): {total_params * 4 / (1024 * 1024):.2f}")  # 假设float32，每个参数4字节
    
    # 测试输入
    batch_size, channels, height, width = 2, 1, 256, 256
    x = torch.randn(batch_size, channels, height, width)
    
    # 前向传播
    detector_scores, weight_scores, descriptors = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Detector scores shape: {detector_scores.shape}")  # 应该是 (2, 1, 256, 256)
    print(f"Weight scores shape: {weight_scores.shape}")      # 应该是 (2, 1, 256, 256) 
    print(f"Descriptors shape: {descriptors.shape}")          # 应该是 (2, C, 256, 256)
    
    # 计算描述符通道数
    first_feature_dim = config['networks']['unet']['first_feature_dimension']
    descriptor_channels = first_feature_dim * (1 + 2 + 4 + 8)  # 四个分支的通道数总和
    print(f"Descriptor channels: {descriptor_channels}")