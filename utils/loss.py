
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed  # 并行处理
# torch.cuda.set_per_process_memory_fraction(0.5)

class Point_Matching_Loss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=10):

        super().__init__()
        self.alpha = alpha

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
    #     locations = locations.squeeze(1)  # (B, H, W)

    #     # 计算 cell 大小，使得所有 keypoints 均匀分布
    #     cell_size = int((H * W) ** 0.5 / num_keypoints ** 0.5)  
    #     grid_h, grid_w = H // cell_size, W // cell_size  # 划分成 grid_h × grid_w 个 cell
    #     num_keypoints = grid_h * grid_w  # 可能与输入 `num_keypoints` 不符，但保证划分均匀

    #     # 使用 unfold 切分窗口，形状变为 (B, grid_h, grid_w, cell_size, cell_size)
    #     patches = locations.unfold(1, cell_size, cell_size).unfold(2, cell_size, cell_size)
    #     patches = patches.contiguous().view(B, grid_h * grid_w, -1)  # (B, num_keypoints, cell_size²)

    #     # spatial softmax
    #     softmaxed = F.softmax(patches, dim=-1)  # (B, num_keypoints, cell_size²)

    #     # 预先计算网格坐标 (不需要 for 循环)
    #     y_indices, x_indices = torch.meshgrid(
    #         torch.linspace(0, cell_size - 1, cell_size),
    #         torch.linspace(0, cell_size - 1, cell_size),
    #         indexing="ij"
    #     )  # (cell_size, cell_size)
    #     coords = torch.stack([x_indices.flatten(), y_indices.flatten()], dim=-1).to(softmaxed.device) # (cell_size², 2)

    #     # 计算 keypoints 坐标 (B, num_keypoints, 2)
    #     keypoints = torch.einsum('bnd,dc->bnc', softmaxed, coords)

    #     # 将 keypoints 变换到全局坐标
    #     grid_x = torch.arange(grid_w, device=locations.device) * cell_size
    #     grid_y = torch.arange(grid_h, device=locations.device) * cell_size
    #     grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
    #     grid_offsets = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (num_keypoints, 2)

    #     keypoints = keypoints + grid_offsets.unsqueeze(0)  # (B, num_keypoints, 2)

    #     return keypoints



    # def spatial_softmax_keypoints(self,locations, num_keypoints=400):
    #     """
    #     计算空间 Softmax 后的关键点坐标，根据响应值选择前 num_keypoints 个点。
        
    #     Args:
    #         locations (torch.Tensor): 形状为 (B, 1, H, W) 的输入张量。
    #         num_keypoints (int): 选取的关键点数量。

    #     Returns:
    #         keypoints (torch.Tensor): 关键点坐标，形状为 (B, num_keypoints, 2)。
    #     """
    #     B, _, H, W = locations.shape
    #     locations = locations.squeeze(1)  # 变为 (B, H, W)

    #     # 计算空间 Softmax
    #     softmaxed = F.softmax(locations.view(B, -1), dim=-1)  # (B, H*W)

    #     # 获取前 num_keypoints 个最大响应值的索引
    #     topk_vals, topk_indices = torch.topk(softmaxed, num_keypoints, dim=-1)  # (B, num_keypoints)

    #     # 计算 keypoints 的全局坐标
    #     y_coords = topk_indices // W  # 行索引
    #     x_coords = topk_indices % W   # 列索引
    #     keypoints = torch.stack([x_coords, y_coords], dim=-1).float()  # (B, num_keypoints, 2)

    #     return keypoints

    def spatial_softmax_keypoints(self, locations, num_keypoints=400):
        """
        Compute keypoints using spatial softmax on a per-cell basis.
        
        Args:
            locations (torch.Tensor): Shape (B, 1, H, W), feature map.
            num_keypoints (int): Number of keypoints to predict.

        Returns:
            keypoints (torch.Tensor): Shape (B, num_keypoints, 2), keypoint coordinates.
        """
        B, _, H, W = locations.shape
        locations = locations.squeeze(1)  # (B, H, W)

        # 计算 cell 大小，使得总共生成 num_keypoints 个关键点
        cell_size = int((H * W) ** 0.5 / num_keypoints ** 0.5)  
        grid_h, grid_w = H // cell_size, W // cell_size  # 划分为 grid_h x grid_w 个 cells
        num_keypoints = grid_h * grid_w  # 可能略微调整，以保证网格均匀

        # 划分网格，并展平 cell
        patches = locations.unfold(1, cell_size, cell_size).unfold(2, cell_size, cell_size)
        patches = patches.contiguous().view(B, num_keypoints, -1)  # (B, num_keypoints, cell_size²)

        # 在每个 cell 内执行 softmax
        softmaxed = F.softmax(patches, dim=-1)  # (B, num_keypoints, cell_size²)

        # 计算 cell 内的相对坐标
        y_idx, x_idx = torch.meshgrid(
            torch.linspace(0, cell_size - 1, cell_size),
            torch.linspace(0, cell_size - 1, cell_size),
            indexing="ij"
        )  # 形状 (cell_size, cell_size)
        coords = torch.stack([x_idx.flatten(), y_idx.flatten()], dim=-1).to(softmaxed.device)  # (cell_size², 2)

        # 计算关键点的相对坐标
        keypoints = torch.einsum('bnd,dc->bnc', softmaxed, coords)  # (B, num_keypoints, 2)

        # 计算 cell 的起始坐标 (全局偏移量)
        grid_x = torch.arange(grid_w, device=locations.device) * cell_size
        grid_y = torch.arange(grid_h, device=locations.device) * cell_size
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid_offsets = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (num_keypoints, 2)

        # 加上 cell 偏移量，得到全局坐标
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
    
    def pix2world(self, locations, resolution, H, W):
        
        # 坐标原点转移到图像中心
        offset_x = (W - 1) / 2.0
        offset_y = (H - 1) / 2.0

        # Clone to avoid modifying input tensor
        world_coordinates = locations.clone().to(torch.float32)

        # Shift scale
        # x向右为正，y向下为正
        world_coordinates[..., 0] = (locations[..., 0] - offset_x) * resolution
        world_coordinates[..., 1] = (offset_y - locations[..., 1]) * resolution

        return world_coordinates
    
    def point_match(self,  locations_map1, scores_map1, descriptors_map1, scores_map2, descriptors_map2,):

        B, C, H, W = descriptors_map1.shape     # (B, C, H, W)
        
        # Parameters
        T = 1.0
        X = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1)    # (H, W, 2)
        # X = X.permute(2, 0, 1).unsqueeze(0)     # (1, 2, H, W)
        # X = X.expand(B, -1, -1, -1)             # (B, 2, H, W)

        ps = self.spatial_softmax_keypoints(locations_map1)     # (B, num_keypoints, 2)
        _, num_keypoints, _ = ps.shape

        ds = self.extract_keypoint_descriptors(descriptors_map1, ps)    # (B, num_keypoints, C)
        

        descriptors_map2_normalized = self.l2_normalize(descriptors_map2, 1e-6) # (B, C, H, W)

        ci = torch.matmul(ds, descriptors_map2_normalized.view(B, C, -1)) # (B, num_keypoints, H*W)
        # S = F.softmax(ci/T, dim=-1).view(B,num_keypoints,H,W)#(B,num_keypoints,H,W)
        # pd = self.soft_argmax(S)#(B,num_keypoints,2)

        S = F.softmax(ci * T, dim=-1)       # (B, num_keypoints, H*W)
        
        pd = torch.matmul(S, X.view(-1, 2)) # (B, num_keypoints, 2)

        # 计算对应的 (y, x) 坐标
        # y_coords = max_idx // W  # 行索引 (y)
        # max_idx = torch.argmax(S, dim=-1)   # (B, num_keypoints)
        # y_coords = torch.div(max_idx, W, rounding_mode='trunc')
        # x_coords = max_idx % W   # 列索引 (x)

        # pd = torch.stack([x_coords, y_coords], dim=-1).float()

        dd = self.extract_keypoint_descriptors(descriptors_map2, pd)    # (B, num_keypoints, C)
        ss = self.bilinear_sample(scores_map1, ps)   # (B, num_keypoints, 1)
        sd = self.bilinear_sample(scores_map2, pd)   # (B, num_keypoints, 1)
        w = (((ds * dd).sum(dim=-1).unsqueeze(-1) + 1) * (ss * sd))/2   # (B, num_keypoints, 1)

        return ps, pd, ds, dd, ss, sd, w
    
    def pose_estimation(self, ps, pd, w, H, W):
        
        epsilon = 1e-6
        resolution = 0.25

        # ps, pd: (B, num_keypoints, 2)
        # ps范围在[0, W-1]，pd范围在[0, H-1]
        # ps, pd坐标原点在左上角，x轴向右，y轴向下
        Qs = self.pix2world(ps, resolution, H, W)   # (B, num_keypoints, 2)
        Qd = self.pix2world(pd, resolution, H, W)   # (B, num_keypoints, 2)

        Qs_avg = torch.sum(w * Qs, dim=1) / (torch.sum(w, dim=1) + epsilon) # (B, 2)
        Qd_avg = torch.sum(w * Qd, dim=1) / (torch.sum(w, dim=1) + epsilon) # (B, 2)
        
        xi = Qs - Qs_avg[:, None, :]    # (B, num_keypoints, 2)
        yi = Qd - Qd_avg[:, None, :]    # (B, num_keypoints, 2)
        
        S = xi.transpose(1, 2) @ (w * yi)   # (B, 2, 2)
        U, Sigma, V_T = torch.svd(S)    # U=(B, 2, 2), Sigma=Σ=(B, 2), V=(B, 2, 2)

        det_sign = torch.det(V_T @ U.transpose(-2, -1)).unsqueeze(-1).unsqueeze(-1)     # (B, 1, 1)
        d = torch.eye(2, device=S.device).unsqueeze(0).repeat(S.shape[0], 1, 1)  # (B, 2, 2)
        d[:, -1, -1] = det_sign.squeeze()
        R = V_T @ d @ U.transpose(-2, -1)   # (B, 2, 2)
        
        t = Qd_avg - (R @ Qs_avg.unsqueeze(-1)).squeeze(-1) # (B, 2)

        return t, R
    
    def hungarian_match(self, cost_matrix, ps_i, pd_i):
        row_ind, col_ind = linear_sum_assignment(cost_matrix)  # 匈牙利算法
        return ps_i[row_ind], pd_i[col_ind]  # 匹配后的坐标

    def match(self,locations_map1, scores_map1, descriptors_map1, scores_map2, descriptors_map2,threshold=0.1):
        
        """
        ds, dd: 描述子 (B, numkeypoints, 248)
        ss, sd: 置信度得分 (B, numkeypoints, 1)
        ps, pd: 关键点坐标 (B, numkeypoints, 2)
        threshold: 置信度阈值
        返回：
        每个 batch 独立返回 (matched_ps, matched_pd)，如果无匹配则返回空张量
        """

        ps, pd, ds, dd, ss, sd, w = self.point_match(locations_map1, scores_map1, descriptors_map1, scores_map2, descriptors_map2)
        t, R = self.pose_estimation(ps, pd, w)

        B, numkeypoints, _ = ds.shape
        results = []

        combined_confidence = ss * sd  # 或者使用加法 ss + sd
        # threshold = combined_confidence.mean()
    
        # 筛选出综合置信度大于阈值的关键点
        # mask = combined_confidence.squeeze(-1) > threshold

        for i in range(B):
            mask = combined_confidence[i].squeeze(-1)
            # threshold = combined_confidence.mean()
            threshold = combined_confidence.quantile(0.8)
            mask_i = mask > threshold

            ps_i = ps[i]  # 当前批次的前一帧坐标
            pd_i = pd[i]  # 当前批次的后一帧坐标
            # mask_i = mask[i]

            results.append([ps_i[mask_i],pd_i[mask_i]])
        
        return results, t, R

        
        # # 根据mask筛选出满足条件的点
        # ps_filtered = ps[mask]
        # pd_filtered = pd[mask]
        # ps_filtered = ps
        # pd_filtered = pd
        # ss_filtered = ss[mask]
        # sd_filtered = sd[mask]

        # for i in range(B):  # 逐 batch 处理
            

        #     # 计算置信度加权
        #     ss_i = ss[i].squeeze(-1)  # (numkeypoints,)
        #     sd_i = sd[i].squeeze(-1)  # (numkeypoints,)
        #     confidence_weight = ss_i * sd_i  # (numkeypoints)
        #     # weighted_score = similarity * confidence_weight  # (numkeypoints, numkeypoints)

        #     mask = confidence_weight >= threshold
            

        #     results.append((matched_ps, matched_pd))

        # return ps_filtered,pd_filtered

        # confidence = ss.unsqueeze(2) * sd.unsqueeze(1)
        # match_scores = cosine_similarity * confidence.mean(dim=-1)
        # best_match_scores, best_matches = torch.topk(match_scores, k=top_k, dim=-1)
        # valid_mask = best_match_scores > threshold
        # best_matches = torch.where(valid_mask, best_matches, torch.full_like(best_matches, -1))
        # matched_ps = ps.unsqueeze(2).expand(-1, -1, top_k, -1)
        # matched_pd = torch.gather(pd.unsqueeze(1).expand(-1, numkeypoints, -1, -1), 2, best_matches.unsqueeze(-1).expand(-1, -1, -1, 2))


        # return best_matches, best_match_scores

    def forward(self,  locations_map1, scores_map1, descriptors_map1, scores_map2, descriptors_map2,pos_trans):

        _, _, H, W = descriptors_map1.shape     # (B, C, H, W)

        ps, pd, _,_,_,_, w = self.point_match(locations_map1, scores_map1, descriptors_map1, scores_map2, descriptors_map2)
        
        t, R = self.pose_estimation(ps, pd, w, H, W)

        t_real = pos_trans[:, :2, 2]    # (B, 2)
        R_real = pos_trans[:, :2, :2]   # (B, 2, 2)

        loss_t = torch.norm(t_real - t, p=2, dim=1).mean() #
        loss_R = torch.norm(R_real @ R.transpose(-2, -1) - torch.eye(2, device=R.device), p=2, dim=(1, 2)).mean()
        loss = loss_t + self.alpha * loss_R

        return loss.mean()


# class L2_Loss(nn.Module):
#     # BCEwithLogitLoss() with reduced missing label effects.
#     def __init__(self, alpha=10):

#         super().__init__()
#         self.alpha = alpha

#     def forward(self, R,t,pos_trans):

#         t_real = pos_trans[:, :2, 2] # (B,2)
#         R_real = pos_trans[:, :2, :2] # (B,2,2)

#         loss_t=torch.norm(t_real - t, p=2, dim=1).mean() #
#         loss_R=torch.norm(R_real @ R.transpose(-2, -1) - torch.eye(2, device=R.device), p=2, dim=(1, 2)).mean()
#         loss = loss_t + self.alpha*loss_R

#         return loss.mean()