import torch
import torch.nn.functional as F
from torch import nn


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Dual_UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=8)
        self.Conv2 = conv_block(ch_in=8, ch_out=16)
        self.Conv3 = conv_block(ch_in=16, ch_out=32)
        self.Conv4 = conv_block(ch_in=32, ch_out=64)
        self.Conv5 = conv_block(ch_in=64, ch_out=128)

        self.loc_decoder = Decoder(128,64,32,16,8)
        self.score_decoder = Decoder(128,64,32,16,8)

        self.Loc_Conv_1x1 = nn.Conv2d(8, output_ch, kernel_size=1)
        self.Score_Conv_1x1 = nn.Conv2d(8, output_ch, kernel_size=1)
    
    # def spatial_softmax_keypoints(self, locations, num_keypoints=400):
    #     """
    #     Compute sub-pixel keypoint locations using spatial softmax.
        
    #     Args:
    #         locations (torch.Tensor): Input tensor of shape (B, 1, H, W), where B is batch size,
    #                                 H and W are the height and width of the feature map.
    #         num_keypoints (int): Number of keypoints to predict.

    #     Returns:
    #         keypoints (torch.Tensor): Sub-pixel keypoint locations of shape (B, num_keypoints, 2).
    #     """
    #     B, _, H, W = locations.shape
    #     locations = locations.squeeze(1)  # Remove channel dimension
        
    #     cell_size = int((H * W) ** 0.5 / num_keypoints ** 0.5)  # Approximate cell size
    #     grid_h, grid_w = H // cell_size, W // cell_size  # Number of cells along each dimension
        
    #     keypoints = []
    #     for b in range(B):
    #         kps = []
    #         for i in range(grid_h):
    #             for j in range(grid_w):
    #                 cell = locations[b, i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
    #                 cell_flat = cell.view(-1)
    #                 softmaxed = F.softmax(cell_flat, dim=0)  # Apply spatial softmax
                    
    #                 # Compute weighted sum of coordinates
    #                 y_grid, x_grid = torch.meshgrid(
    #                     torch.linspace(i * cell_size, (i + 1) * cell_size - 1, cell_size),
    #                     torch.linspace(j * cell_size, (j + 1) * cell_size - 1, cell_size),
    #                     indexing='ij'
    #                 )
    #                 x_coord = (softmaxed * x_grid.flatten()).sum()
    #                 y_coord = (softmaxed * y_grid.flatten()).sum()
    #                 kps.append(torch.stack([x_coord, y_coord]))
    #         keypoints.append(torch.stack(kps))
        
    #     return torch.stack(keypoints) #(B,num_keypoints,2) = (B, 400, 2)

    def spatial_softmax_keypoints(self, locations, num_keypoints=400):
        """
        Compute sub-pixel keypoint locations using spatial softmax.

        Args:
            locations (torch.Tensor): Input tensor of shape (B, 1, H, W), where B is batch size,
                                    H and W are the height and width of the feature map.
            num_keypoints (int): Number of keypoints to predict.

        Returns:
            keypoints (torch.Tensor): Sub-pixel keypoint locations of shape (B, num_keypoints, 2).
        """
        B, _, H, W = locations.shape
        locations = locations.squeeze(1)  # (B, H, W)

        # 计算 cell 大小，使得所有 keypoints 均匀分布
        cell_size = int((H * W) ** 0.5 / num_keypoints ** 0.5)  
        grid_h, grid_w = H // cell_size, W // cell_size  # 划分成 grid_h × grid_w 个 cell
        num_keypoints = grid_h * grid_w  # 可能与输入 `num_keypoints` 不符，但保证划分均匀

        # 使用 unfold 切分窗口，形状变为 (B, grid_h, grid_w, cell_size, cell_size)
        patches = locations.unfold(1, cell_size, cell_size).unfold(2, cell_size, cell_size)
        patches = patches.contiguous().view(B, grid_h * grid_w, -1)  # (B, num_keypoints, cell_size²)

        # spatial softmax
        softmaxed = F.softmax(patches, dim=-1)  # (B, num_keypoints, cell_size²)

        # 预先计算网格坐标 (不需要 for 循环)
        y_indices, x_indices = torch.meshgrid(
            torch.linspace(0, cell_size - 1, cell_size),
            torch.linspace(0, cell_size - 1, cell_size),
            indexing="ij"
        )  # (cell_size, cell_size)
        coords = torch.stack([x_indices.flatten(), y_indices.flatten()], dim=-1)  # (cell_size², 2)

        # 计算 keypoints 坐标 (B, num_keypoints, 2)
        keypoints = torch.einsum('bnd,dc->bnc', softmaxed, coords)

        # 将 keypoints 变换到全局坐标
        grid_x = torch.arange(grid_w, device=locations.device) * cell_size
        grid_y = torch.arange(grid_h, device=locations.device) * cell_size
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid_offsets = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (num_keypoints, 2)

        keypoints = keypoints + grid_offsets.unsqueeze(0)  # (B, num_keypoints, 2)

        return keypoints
    
    def bilinear_sample(self, X, keypoints):
        """
        Perform bilinear sampling on feature map X at the coordinates keypoints.
        
        Args:
            X (torch.Tensor): Feature map of shape (B, C, H, W).
            keypoints (torch.Tensor): Keypoint coordinates of shape (B, num_keypoints, 2),
                                    where each keypoint has (x, y) coordinates.

        Returns:
            descriptors (torch.Tensor): Keypoint descriptors of shape (B, num_keypoints, C).
        """
        B, C, H, W = X.shape
        num_keypoints = keypoints.shape[1]

        # Normalize keypoints to the range [-1, 1] for grid_sample
        x = (keypoints[..., 0] / (W - 1)) * 2 - 1  # Normalize to [-1, 1]
        y = (keypoints[..., 1] / (H - 1)) * 2 - 1
        grid = torch.stack((x, y), dim=-1).unsqueeze(1)  # (B, 1, num_keypoints, 2)

        # Bilinear sampling using grid_sample
        sampled = F.grid_sample(X, grid, mode='bilinear', padding_mode='border', align_corners=True)

        # Reshape output to (B, num_keypoints, C)
        descriptors = sampled.squeeze(2).permute(0, 2, 1)  # (B, num_keypoints, C)

        return descriptors


    def l2_normalize(self, descriptors, eps=1e-6):
        """
        Perform L2 normalization on descriptors.
        
        Args:
            descriptors (torch.Tensor): Descriptors of shape (B, num_keypoints, C).
            eps (float): Small constant to avoid division by zero.
        
        Returns:
            descriptors (torch.Tensor): L2 normalized descriptors.
        """
        return descriptors / (descriptors.norm(p=2, dim=-1, keepdim=True) + eps)


    def extract_keypoint_descriptors(self, X, keypoints):
        """
        Extract keypoint descriptors using bilinear sampling and L2 normalization.
        
        Args:
            X (torch.Tensor): Feature map of shape (B, C, H, W).
            keypoints (torch.Tensor): Keypoint coordinates of shape (B, num_keypoints, 2).
        
        Returns:
            descriptors (torch.Tensor): L2 normalized keypoint descriptors of shape (B, num_keypoints, C).
        """
        descriptors = self.bilinear_sample(X, keypoints)
        descriptors = self.l2_normalize(descriptors)
        
        return descriptors

    # def bilinear_sample(self, X, y):
    #     """
    #     Perform bilinear sampling on feature map X at the coordinates y.
        
    #     Args:
    #         X (torch.Tensor): Feature map of shape (B, C, H, W), where B is batch size,
    #                         C is the number of channels, and H, W are the height and width.
    #         y (torch.Tensor): Keypoint coordinates of shape (B, num_keypoints, 2).
    #                         Each keypoint has (x, y) coordinates.
        
    #     Returns:
    #         descriptors (torch.Tensor): Keypoint descriptors of shape (B, num_keypoints, C).
    #     """
    #     B, C, H, W = X.shape
    #     num_keypoints = y.shape[1]

    #     # Normalize coordinates to be within [0, H-1] and [0, W-1]
    #     y = y.clamp(min=0, max=torch.tensor([H - 1, W - 1], dtype=y.dtype, device=y.device))

    #     # Perform bilinear interpolation
    #     x = y[..., 0]  # (B, num_keypoints)
    #     y = y[..., 1]  # (B, num_keypoints)

    #     # Get grid for bilinear sampling
    #     grid = torch.stack([x, y], dim=-1)  # (B, num_keypoints, 2)
    #     grid = grid.unsqueeze(1).repeat(1, C, 1, 1)  # (B, C, num_keypoints, 2)

    #     # Bilinear sampling
    #     sampled = F.grid_sample(X, grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)
    #     descriptors = sampled.squeeze(-1).transpose(1, 2)  # (B, num_keypoints, C)
        
    #     return descriptors


    # def l2_normalize(self, descriptors):
    #     """
    #     Perform L2 normalization on descriptors.
        
    #     Args:
    #         descriptors (torch.Tensor): Descriptors to normalize, shape (B, num_keypoints, C).
        
    #     Returns:
    #         descriptors (torch.Tensor): L2 normalized descriptors.
    #     """
    #     norm = descriptors.norm(p=2, dim=-1, keepdim=True)
    #     descriptors = descriptors / norm  # L2 normalization
    #     return descriptors


    # def extract_keypoint_descriptors(self, X, keypoints):
    #     """
    #     Extract keypoint descriptors using bilinear sampling and L2 normalization.
        
    #     Args:
    #         X (torch.Tensor): Feature map of shape (B, C, H, W).
    #         keypoints (torch.Tensor): Keypoint coordinates of shape (B, num_keypoints, 2).
        
    #     Returns:
    #         descriptors (torch.Tensor): L2 normalized keypoint descriptors of shape (B, num_keypoints, C).
    #     """
    #     # Step 1: Perform bilinear sampling
    #     descriptors = self.bilinear_sample(X, keypoints)
        
    #     # Step 2: L2 normalize descriptors
    #     descriptors = self.l2_normalize(descriptors)
        
    #     return descriptors

    def Descriptor(self,x1,x2,x3,x4,x5):
        x2= F.interpolate(x2, scale_factor=2, mode='bilinear')
        x3= F.interpolate(x3, scale_factor=4, mode='bilinear')
        x4= F.interpolate(x4, scale_factor=8, mode='bilinear')
        x5= F.interpolate(x5, scale_factor=16, mode='bilinear')
        descriptors_map=torch.cat((x1,x2,x3,x4,x5), dim=1)
        
        return descriptors_map

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))

        location_feature=self.loc_decoder(x1, x2, x3, x4, x5)
        score_feature=self.score_decoder(x1, x2, x3, x4, x5)

        locations_map = self.Spatial_softmax(self.Loc_Conv_1x1(location_feature)) #(B,1,H,W)
        scores_map = nn.Sigmoid(self.Score_Conv_1x1(score_feature)) #(B,1,H,W)
        descriptors_map = self.Descriptor(x1, x2, x3, x4, x5) #(B,248,H,W)
        descriptors = self.extract_keypoint_descriptors(descriptors_map, locations)

        return locations_map, scores_map, descriptors_map 

class Decoder(nn.Module):
    def __init__(self,ch1,ch2,ch3,ch4,ch5):
        super().__init__()
        self.Up5 = up_conv(ch_in=ch1, ch_out=ch2)
        self.Up_conv5 = conv_block(ch_in=ch2, ch_out=ch2)
        self.Up4 = up_conv(ch_in=ch2, ch_out=ch3)
        self.Up_conv4 = conv_block(ch_in=ch3, ch_out=ch3)
        self.Up3 = up_conv(ch_in=ch3, ch_out=ch4)
        self.Up_conv3 = conv_block(ch_in=ch4, ch_out=ch4)
        self.Up2 = up_conv(ch_in=ch4, ch_out=ch5)
        self.Up_conv2 = conv_block(ch_in=ch5, ch_out=ch5)

    def forward(self, x1, x2, x3, x4, x5):
        d5 = self.Up5(x5)
        d5 = self.Up_conv5(torch.cat((x4, d5), dim=1))
        d4 = self.Up4(d5)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))
        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))
        return d2