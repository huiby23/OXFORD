
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.cuda.set_per_process_memory_fraction(0.5)

class Point_Matching_Loss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=10):

        super().__init__()
        self.alpha = alpha

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
        coords = torch.stack([x_indices.flatten(), y_indices.flatten()], dim=-1).to(softmaxed.device) # (cell_size², 2)

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
        assert keypoints.shape[-1] == 2

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
    
    def sample_function(self, sample_matrix, feature):
        B, _, H, W = feature.shape#(B,1,H,W)
        sample_matrix[:, 0, :, :, :] = (sample_matrix[:, 0, :, :, :] / (W - 1)) * 2 - 1  # x 方向
        sample_matrix[:, 1, :, :, :] = (sample_matrix[:, 1, :, :, :] / (H - 1)) * 2 - 1  # y 方向
        sample_matrix = sample_matrix.permute(0, 2, 3, 4, 1)
        sampled_features = F.grid_sample(feature, sample_matrix, align_corners=True)  # (B, 1, num_keypoints, H, W)
        sampled_features = sampled_features.squeeze(1).mean(dim=(-1, -2))

    def soft_argmax(self, S):
        B, num_keypoints, H, W = S.shape

        # 生成网格坐标
        device = S.device
        grid_x = torch.arange(W, device=device).float().view(1, 1, 1, W)  # (1,1,1,W)
        grid_y = torch.arange(H, device=device).float().view(1, 1, H, 1)  # (1,1,H,1)

        # 计算期望坐标
        x_coords = (S * grid_x).sum(dim=(2, 3))  # (B, num_keypoints)
        y_coords = (S * grid_y).sum(dim=(2, 3))  # (B, num_keypoints)

        # 最终关键点坐标 (B, num_keypoints, 2)
        keypoints = torch.stack([x_coords, y_coords], dim=-1)

        return keypoints
    
    def pix2world(self, locations, pos_tran):
        B, N, _ = locations.shape
        ones = torch.ones(B, N, 1, device=locations.device)  # 创建 (B, num_keypoints, 1) 的 1
        locations_homogeneous = torch.cat([locations, ones], dim=-1)  # (B, num_keypoints, 3)
        transformed_homogeneous = torch.bmm(pos_tran, locations_homogeneous.transpose(1, 2))  # (B, 3, num_keypoints)
        world_coordinates = (transformed_homogeneous[:, :2, :] / transformed_homogeneous[:, 2:3, :]).transpose(1, 2)  # (B, num_keypoints, 2)

        return world_coordinates
    
    def point_match(self,ps, ds, dd, ss, sd):
        wi=0
        return wi
    
    def pose_estimation(self, ps, pd, w):
        t=0
        return t
    
    def forward(self,  locations_map1, scores_map1, descriptors_map1,locations_map2, scores_map2, descriptors_map2,pos_trans, alfa):

        B, C, H, W = descriptors_map1.shape#(B,1,H,W)
        T=0.1
        epsilon = 1e-6
        X = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1)#(B,2,H,W)

        ps = self.spatial_softmax_keypoints(locations_map1)#(B,num_keypoints,2)
        ds = self.extract_keypoint_descriptors(descriptors_map1,ps)#(B,num_keypoints,C)
        _,num_keypoints,_ = ds.shape

        # ds_extend = ds.unsqueeze(3).unsqueeze(4)  # (B, num_keypoints, feature_dim, 1)
        # descriptors_map2_extend = descriptors_map2.unsqueeze(1)

        # descriptors_map2_flat = descriptors_map2.view(B, 248, -1)  # (12,248,448*448)

        # 执行批量矩阵乘法：(12,400,248) × (12,248,200704) → (12,400,200704)
        ci = torch.matmul(ds, descriptors_map2.view(B,C,-1))

        # ci = torch.einsum('bik,bjkhw->bijhw', ds, descriptors_map2)
        # ci = torch.matmul(ds_extend, descriptors_map2_extend)
        # ci=((ds.unsqueeze(3).unsqueeze(4))*(descriptors_map2.unsqueeze(1))).sum(dim=2)#(B,num_keypoints,H,W)
        S = F.softmax(ci/T, dim=-1).view(B,num_keypoints,H,W)#(B,num_keypoints,H,W)
        pd = self.soft_argmax(S)#(B,num_keypoints,2)
        dd = self.extract_keypoint_descriptors(descriptors_map2,pd)#(B,num_keypoints,C)
        ss = self.bilinear_sample(scores_map1,pd)#(B, num_keypoints, 1)
        sd = self.bilinear_sample(scores_map2,ps)#(B, num_keypoints, 1)
        w = (((ds*dd).sum(dim=-1).unsqueeze(-1)+1)*(ss*sd))/2#(B, num_keypoints, 1)

        Qs,Qd= self.pix2world(ps,pos_trans),self.pix2world(pd,pos_trans)# (B, num_keypoints, 2)
        Qs_avg=torch.sum(w * Qs, dim=1) / (torch.sum(w, dim=1)+epsilon)#(B,2)
        Qd_avg=torch.sum(w * Qd, dim=1) / (torch.sum(w, dim=1)+epsilon)#(B,2)
        xi=Qs-Qs_avg[:, None, :]# (B, num_keypoints, 2)
        yi=Qd-Qd_avg[:, None, :]# (B, num_keypoints, 2)
        S = xi.transpose(1, 2) @ (w * yi) # (B, 2, 2)
        U, Sigma, V_T = torch.svd(S)  # U=(B,2,2),Sigma=Σ=(B,2),V=(B,2,2)

        det_sign = torch.det(V_T @ U.transpose(-2, -1)).unsqueeze(-1).unsqueeze(-1) 
        d = torch.eye(2, device=S.device).unsqueeze(0).repeat(S.shape[0], 1, 1)  # (B, 2, 2)
        d[:, -1, -1] = det_sign.squeeze()
        R = V_T @ d @ U.transpose(-2, -1)# (B, 2, 2)
        t = Qd_avg - (R @ Qs_avg.unsqueeze(-1)).squeeze(-1) # (B, 2)
        t_real = pos_trans[:, :2, 2] # (B,2)
        R_real = pos_trans[:, :2, :2] # (B,2,2)

        loss_t=torch.norm(t_real - t, p=2, dim=1).mean() #
        loss_R=torch.norm(R_real @ R.transpose(-2, -1) - torch.eye(2, device=R.device), p=2, dim=(1, 2)).mean()
        loss = loss_t + alfa*loss_R

        return loss.mean()


