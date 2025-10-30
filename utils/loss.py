import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_indices, normalize_coords, convert_to_radar_frame, convert_to_weight_matrix

# loss computation
def supervised_loss(R_tgt_src_pred, t_tgt_src_pred, batch, config, alpha=10.0):
    """This function computes the L1 loss between the predicted and groundtruth translation in addition to
        the rotation loss (R_pred.T * R) - I.
    Args:
        R_tgt_src_pred (torch.tensor): (b,3,3) predicted rotation
        t_tgt_src_pred (torch.tensor): (b,3,1) predicted translation
        batch (dict): input data for the batch
        config (json): parsed config file
    Returns:
        svd_loss (float): supervised loss
        dict_loss (dict): a dictionary containing the separate loss components
    """
    T_21 = batch['T_21'].to(config['gpuid'])
    batch_size = R_tgt_src_pred.size(0)
    num_wins = R_tgt_src_pred.size(0)
    if num_wins == 1:
        batch_size = 1      # val/test: batch_size = 1 
    
    # Get ground truth transforms
    kp_inds, _ = get_indices(batch_size, config['window_size'])
    T_tgt_src = T_21[kp_inds]
    R_tgt_src = T_tgt_src[:, :3, :3]
    t_tgt_src = T_tgt_src[:, :3, 3].unsqueeze(-1)
    
    identity = torch.eye(3).unsqueeze(0).repeat(num_wins, 1, 1).to(config['gpuid'])
    loss_fn = torch.nn.L1Loss()
    
    R_loss = loss_fn(torch.matmul(R_tgt_src_pred.transpose(2, 1), R_tgt_src), identity)
    t_loss = loss_fn(t_tgt_src_pred, t_tgt_src)
    
    if config['loss_alpha'] == alpha:
        svd_loss = t_loss + alpha * R_loss
    else:
        svd_loss = t_loss + config['loss_alpha'] * R_loss
    dict_loss = {'R_loss': R_loss, 't_loss': t_loss}
    
    return svd_loss, dict_loss

def supervised_loss_new(R_tgt_src_pred, t_tgt_src_pred, src_coords, tgt_coords, batch, config, alpha=10.0):
    """This function computes the L1 loss between the predicted and groundtruth translation in addition to
        the rotation loss (R_pred.T * R) - I.
    Args:
        R_tgt_src_pred (torch.tensor): (b,3,3) predicted rotation
        t_tgt_src_pred (torch.tensor): (b,3,1) predicted translation
        src_coords (torch.tensor): (b,N,2) source keypoint locations
        tgt_coords (torch.tensor): (b,N,2) target keypoint locations
        batch (dict): input data for the batch
        config (json): parsed config file
    Returns:
        svd_loss (float): supervised loss
        dict_loss (dict): a dictionary containing the separate loss components
    """
    T_21 = batch['T_21'].to(config['gpuid'])
    batch_size = config['batch_size']
    num_wins = R_tgt_src_pred.size(0)
    if num_wins == 1:
        batch_size = 1      # val/test: batch_size = 1 
        
    # Get ground truth transforms
    kp_inds, _ = get_indices(batch_size, config['window_size'])
    T_tgt_src = T_21[kp_inds]
    R_tgt_src = T_tgt_src[:, :3, :3]
    t_tgt_src = T_tgt_src[:, :3, 3].unsqueeze(-1)

    identity = torch.eye(3).unsqueeze(0).repeat(num_wins, 1, 1).to(config['gpuid'])
    loss_fn = torch.nn.L1Loss()
    loss_l2 = torch.nn.MSELoss(reduce=False)
    
    R_loss = loss_fn(torch.matmul(R_tgt_src_pred.transpose(2, 1), R_tgt_src), identity)
    t_loss = loss_fn(t_tgt_src_pred, t_tgt_src)

    # distance loss
    dist_temp = config['dist_loss_temp']
    dist_thred = config['dist_loss_thred']
    dist_map = torch.sqrt(torch.sum(loss_l2(src_coords, tgt_coords), dim=2, keepdim=True))      # B X N X 1
    dist_loss = soft_threshold_ratio(dist_map, dist_thred, dist_temp)
    
    if config['loss_alpha'] == alpha:
        svd_loss = config['loss_t'] * t_loss + alpha * R_loss + config['loss_beta'] * dist_loss
    else:
        svd_loss = config['loss_t'] * t_loss + config['loss_alpha'] * R_loss + config['loss_beta'] * dist_loss
    dict_loss = {'R_loss': R_loss, 't_loss': t_loss, 'dist_loss': dist_loss}
    
    return svd_loss, dict_loss

def soft_threshold_ratio(dist_map, threshold, temperature=1.0):
    """
    Args:
        dist_map: (b, N, 1) 欧氏距离
        threshold: 标量阈值
        temperature: 控制 Sigmoid 的陡峭程度（越小越接近硬阈值）
    Returns:
        ratio_loss: 可微分的超阈值比例估计
    """
    # 输入数据检查
    assert not torch.isnan(dist_map).any(), "dist_map contains NaN!"
    assert not torch.isinf(dist_map).any(), "dist_map contains Inf!"

    # 计算超阈值概率（Sigmoid 软二值化）
    prob = torch.sigmoid((dist_map - threshold) * temperature)  # B X N X 1
    ratio_loss = prob.mean()  # 求平均
    return ratio_loss


def supervised_loss_multi(R_tgt_src_pred, t_tgt_src_pred, src_coords, tgt_coords, batch, config, alpha=10.0):
    """This function computes the L1 loss between the predicted and groundtruth translation in addition to
        the rotation loss (R_pred.T * R) - I.
    Args:
        R_tgt_src_pred (torch.tensor): (b,3,3) predicted rotation
        t_tgt_src_pred (torch.tensor): (b,3,1) predicted translation
        src_coords (torch.tensor): (b,N,2) source keypoint locations
        tgt_coords (torch.tensor): (b,N,2) target keypoint locations
        batch (dict): input data for the batch
        config (json): parsed config file
    Returns:
        svd_loss (float): supervised loss
        dict_loss (dict): a dictionary containing the separate loss components
    """
    T_21 = batch['T_21'].to(config['gpuid'])
    batch_size = config['batch_size']
    num_wins = R_tgt_src_pred.size(0)
    if num_wins == 1:
        batch_size = 1      # val/test: batch_size = 1 
        
    # Get ground truth transforms
    kp_inds, _ = get_indices(batch_size, config['window_size'])
    T_tgt_src = T_21[kp_inds]
    R_tgt_src = T_tgt_src[:, :3, :3]
    t_tgt_src = T_tgt_src[:, :3, 3].unsqueeze(-1)
    
    # loss function
    loss_fn = torch.nn.L1Loss()
    loss_l2 = torch.nn.MSELoss(reduce=False)

    # multi wins loss
    loss_multi = 0
    if num_wins > 1:
        T_pred = torch.eye(4).unsqueeze(0).repeat(num_wins, 1, 1).to(config['gpuid'])
        T_pred[:, :3, :3] = R_tgt_src_pred
        T_pred[:, :3, 3] = t_tgt_src_pred.squeeze(-1)
        
        # 3 frames / 2 wins
        num_triple = num_wins - 1
        T_gt_triple_list = []
        T_pred_triple_list = []
        for i in range(num_triple):
            T_0 = T_tgt_src[i, :, :]
            T_1 = T_tgt_src[i+1, :, :]
            T_gt_triple = torch.matmul(T_1, T_0)
            T_gt_triple_list.append(T_gt_triple)
            
            T_0 = T_pred[i, :, :]
            T_1 = T_pred[i+1, :, :]
            T_pred_triple = torch.matmul(T_1, T_0)
            T_pred_triple_list.append(T_pred_triple)
        
        T_gt_triple = torch.stack(T_gt_triple_list)
        T_pred_triple = torch.stack(T_pred_triple_list)
        identity_triple = torch.eye(3).unsqueeze(0).repeat(num_triple, 1, 1).to(config['gpuid'])
        
        R_gt_triple = T_gt_triple[:, :3, :3]
        t_gt_triple = T_gt_triple[:, :3, 3].unsqueeze(-1)
        R_pred_triple = T_pred_triple[:, :3, :3]
        t_pred_triple = T_pred_triple[:, :3, 3].unsqueeze(-1)
        
        R_loss_triple = loss_fn(torch.matmul(R_pred_triple.transpose(2, 1), R_gt_triple), identity_triple)
        t_loss_triple = loss_fn(t_pred_triple, t_gt_triple)
        
        # 4 frames / 3 wins
        num_quadruple = num_wins - 2
        T_gt_quadruple_list = []
        T_pred_quadruple_list = []
        for i in range(num_quadruple):
            T_0 = T_tgt_src[i, :, :]
            T_1 = T_tgt_src[i+1, :, :]
            T_2 = T_tgt_src[i+2, :, :]
            T_gt_quadruple = torch.matmul(T_2, torch.matmul(T_1, T_0))
            T_gt_quadruple_list.append(T_gt_quadruple)
            
            T_0 = T_pred[i, :, :]
            T_1 = T_pred[i+1, :, :]
            T_2 = T_pred[i+2, :, :]
            T_pred_quadruple = torch.matmul(T_2, torch.matmul(T_1, T_0))
            T_pred_quadruple_list.append(T_pred_quadruple)
        
        T_gt_quadruple = torch.stack(T_gt_quadruple_list)
        T_pred_quadruple = torch.stack(T_pred_quadruple_list)
        identity_quadruple = torch.eye(3).unsqueeze(0).repeat(num_quadruple, 1, 1).to(config['gpuid'])
        
        R_gt_quadruple = T_gt_quadruple[:, :3, :3]
        t_gt_quadruple = T_gt_quadruple[:, :3, 3].unsqueeze(-1)
        R_pred_quadruple = T_pred_quadruple[:, :3, :3]
        t_pred_quadruple = T_pred_quadruple[:, :3, 3].unsqueeze(-1)
        
        R_loss_quadruple = loss_fn(torch.matmul(R_pred_quadruple.transpose(2, 1), R_gt_quadruple), identity_quadruple)
        t_loss_quadruple = loss_fn(t_pred_quadruple, t_gt_quadruple)
        
        # sum
        loss_multi_t = config['loss_t'] * t_loss_triple / 2 + config['loss_t'] * t_loss_quadruple / 3
        loss_multi_R = config['loss_alpha'] * R_loss_triple / 2 + config['loss_alpha'] * R_loss_quadruple / 3
        loss_multi = config['loss_m'] * (loss_multi_t + loss_multi_R)

    # single wins loss
    identity = torch.eye(3).unsqueeze(0).repeat(num_wins, 1, 1).to(config['gpuid'])
    
    R_loss = loss_fn(torch.matmul(R_tgt_src_pred.transpose(2, 1), R_tgt_src), identity)
    t_loss = loss_fn(t_tgt_src_pred, t_tgt_src)

    # distance loss
    dist_temp = config['dist_loss_temp']
    dist_thred = config['dist_loss_thred']
    dist_map = torch.sqrt(torch.sum(loss_l2(src_coords, tgt_coords), dim=2, keepdim=True))      # B X N X 1
    dist_loss = soft_threshold_ratio(dist_map, dist_thred, dist_temp)
    
    if config['loss_alpha'] == alpha:
        svd_loss = config['loss_t'] * t_loss + alpha * R_loss + config['loss_beta'] * dist_loss + loss_multi
    else:
        svd_loss = config['loss_t'] * t_loss + config['loss_alpha'] * R_loss + config['loss_beta'] * dist_loss + loss_multi
    dict_loss = {'R_loss': R_loss, 't_loss': t_loss, 'dist_loss': dist_loss, 'multi_loss': loss_multi}
    
    return svd_loss, dict_loss


def supervised_loss_multi_score(R_tgt_src_pred, t_tgt_src_pred, src_coords, tgt_coords, weight_scores, batch, config):
    """This function computes the L1 loss between the predicted and groundtruth translation in addition to
        the rotation loss (R_pred.T * R) - I.
    Args:
        R_tgt_src_pred (torch.tensor): (b,3,3) predicted rotation
        t_tgt_src_pred (torch.tensor): (b,3,1) predicted translation
        src_coords (torch.tensor): (b,N,2) source keypoint locations
        tgt_coords (torch.tensor): (b,N,2) target keypoint locations
        weight_scores (torch.tensor): (b,1,H,W) keypoint scores
        batch (dict): input data for the batch
        config (json): parsed config file
    Returns:
        svd_loss (float): supervised loss
        dict_loss (dict): a dictionary containing the separate loss components
    """
    T_21 = batch['T_21'].to(config['gpuid'])
    batch_size = config['batch_size']
    num_wins = R_tgt_src_pred.size(0)
    if num_wins == 1:
        batch_size = 1      # val/test: batch_size = 1 
        
    # Get ground truth transforms
    kp_inds, _ = get_indices(batch_size, config['window_size'])
    T_tgt_src = T_21[kp_inds]
    R_tgt_src = T_tgt_src[:, :3, :3]
    t_tgt_src = T_tgt_src[:, :3, 3].unsqueeze(-1)
    
    # loss function
    loss_fn = torch.nn.L1Loss()
    loss_l2 = torch.nn.MSELoss(reduce=False)

    # multi wins loss
    multi_loss = 0
    if config['loss_multi'] and num_wins > 1:
        T_pred = torch.eye(4).unsqueeze(0).repeat(num_wins, 1, 1).to(config['gpuid'])
        T_pred[:, :3, :3] = R_tgt_src_pred
        T_pred[:, :3, 3] = t_tgt_src_pred.squeeze(-1)
        
        # 3 frames / 2 wins
        num_triple = num_wins - 1
        T_gt_triple_list = []
        T_pred_triple_list = []
        for i in range(num_triple):
            T_0 = T_tgt_src[i, :, :]
            T_1 = T_tgt_src[i+1, :, :]
            T_gt_triple = torch.matmul(T_1, T_0)
            T_gt_triple_list.append(T_gt_triple)
            
            T_0 = T_pred[i, :, :]
            T_1 = T_pred[i+1, :, :]
            T_pred_triple = torch.matmul(T_1, T_0)
            T_pred_triple_list.append(T_pred_triple)
        
        T_gt_triple = torch.stack(T_gt_triple_list)
        T_pred_triple = torch.stack(T_pred_triple_list)
        identity_triple = torch.eye(3).unsqueeze(0).repeat(num_triple, 1, 1).to(config['gpuid'])
        
        R_gt_triple = T_gt_triple[:, :3, :3]
        t_gt_triple = T_gt_triple[:, :3, 3].unsqueeze(-1)
        R_pred_triple = T_pred_triple[:, :3, :3]
        t_pred_triple = T_pred_triple[:, :3, 3].unsqueeze(-1)
        
        R_loss_triple = loss_fn(torch.matmul(R_pred_triple.transpose(2, 1), R_gt_triple), identity_triple)
        t_loss_triple = loss_fn(t_pred_triple, t_gt_triple)
        
        # 4 frames / 3 wins
        num_quadruple = num_wins - 2
        T_gt_quadruple_list = []
        T_pred_quadruple_list = []
        for i in range(num_quadruple):
            T_0 = T_tgt_src[i, :, :]
            T_1 = T_tgt_src[i+1, :, :]
            T_2 = T_tgt_src[i+2, :, :]
            T_gt_quadruple = torch.matmul(T_2, torch.matmul(T_1, T_0))
            T_gt_quadruple_list.append(T_gt_quadruple)
            
            T_0 = T_pred[i, :, :]
            T_1 = T_pred[i+1, :, :]
            T_2 = T_pred[i+2, :, :]
            T_pred_quadruple = torch.matmul(T_2, torch.matmul(T_1, T_0))
            T_pred_quadruple_list.append(T_pred_quadruple)
        
        T_gt_quadruple = torch.stack(T_gt_quadruple_list)
        T_pred_quadruple = torch.stack(T_pred_quadruple_list)
        identity_quadruple = torch.eye(3).unsqueeze(0).repeat(num_quadruple, 1, 1).to(config['gpuid'])
        
        R_gt_quadruple = T_gt_quadruple[:, :3, :3]
        t_gt_quadruple = T_gt_quadruple[:, :3, 3].unsqueeze(-1)
        R_pred_quadruple = T_pred_quadruple[:, :3, :3]
        t_pred_quadruple = T_pred_quadruple[:, :3, 3].unsqueeze(-1)
        
        R_loss_quadruple = loss_fn(torch.matmul(R_pred_quadruple.transpose(2, 1), R_gt_quadruple), identity_quadruple)
        t_loss_quadruple = loss_fn(t_pred_quadruple, t_gt_quadruple)
        
        # sum
        multi_loss_t = config['loss_t'] * t_loss_triple / 2 + config['loss_t'] * t_loss_quadruple / 3
        multi_loss_R = config['loss_alpha'] * R_loss_triple / 2 + config['loss_alpha'] * R_loss_quadruple / 3
        multi_loss = config['loss_m'] * (multi_loss_t + multi_loss_R)

    # score regularization loss
    score_reg_loss = 0
    if config['loss_score_reg']:
        eps=1e-6
        # lambda_ent=0.01
        # lambda_kl=0.1
        # lambda_div=0.01
        # target_mean=0.4
        
        # flatten to (N, H*W)
        scores = weight_scores.view(weight_scores.size(0), -1)

        # Entropy regularization (encourage non-collapsed distribution)
        # mean over spatial dimension, then batch mean
        entropy = - (scores * torch.log(scores + eps) + (1 - scores) * torch.log(1 - scores + eps))
        ent_loss = - entropy.mean()

        # # KL regularization on mean score (push avg score toward target_mean)
        # mean_score = scores.mean(dim=1)  # per sample mean
        # kl_loss = ((mean_score - target_mean) ** 2).mean()

        # # Diversity regularization (encourage variance across spatial scores)
        # var_per_sample = scores.var(dim=1)  # variance per sample
        # div_loss = -var_per_sample.mean()   # maximize variance => minimize negative variance

        # total loss
        # score_reg_loss = lambda_ent * ent_loss + lambda_kl * kl_loss + lambda_div * div_loss
        score_reg_loss = config['loss_s_ent'] * ent_loss

    
    # single wins loss
    identity = torch.eye(3).unsqueeze(0).repeat(num_wins, 1, 1).to(config['gpuid'])
    
    R_loss = loss_fn(torch.matmul(R_tgt_src_pred.transpose(2, 1), R_tgt_src), identity)
    t_loss = loss_fn(t_tgt_src_pred, t_tgt_src)

    # distance loss
    dist_temp = config['dist_loss_temp']
    dist_thred = config['dist_loss_thred']
    dist_map = torch.sqrt(torch.sum(loss_l2(src_coords, tgt_coords), dim=2, keepdim=True))      # B X N X 1
    dist_loss = soft_threshold_ratio(dist_map, dist_thred, dist_temp)
    
    svd_loss = config['loss_t'] * t_loss + config['loss_alpha'] * R_loss + config['loss_beta'] * dist_loss + multi_loss + score_reg_loss
    
    dict_loss = {'R_loss': R_loss, 't_loss': t_loss, 'dist_loss': dist_loss}
    
    if config['loss_multi'] and num_wins > 1:
        dict_loss['multi_loss'] = multi_loss
    
    if config['loss_score_reg']:
        dict_loss['score_reg_loss'] = score_reg_loss
    
    return svd_loss, dict_loss



# Keypoints initialization
class Keypoint(torch.nn.Module):
    """
        Given a dense map of detector scores and weight scores, this modules computes keypoint locations, and their
        associated scores and descriptors. A spatial softmax is used over a regular grid of "patches" to extract a
        single location, score, and descriptor per patch.
    """
    def __init__(self, config):
        super().__init__()
        self.patch_size = config['networks']['keypoint_block']['patch_size']
        self.gpuid = config['gpuid']
        self.width = config['cart_pixel_width']
        v_coords, u_coords = torch.meshgrid([torch.arange(0, self.width), torch.arange(0, self.width)])
        self.v_coords = v_coords.unsqueeze(0).float()   # (1,H,W)
        self.u_coords = u_coords.unsqueeze(0).float()   # (1,H,W)

    def forward(self, detector_scores, weight_scores, descriptors):
        """ A spatial softmax is performed for each grid cell over the detector_scores tensor to obtain 2D
            keypoint locations. Bilinear sampling is used to obtain the correspoding scores and descriptors.
            num_patches is the number of keypoints output by this module.
        Args:
            detector_scores (torch.tensor): (b*w,1,H,W)
            weight_scores (torch.tensor): (b*w,S,H,W) Note that S=1 for scalar weights, S=3 for 2x2 weight matrices
            descriptors (torch.tensor): (b*w,C,H,W) C = descriptor dim
        Returns:
            keypoint_coords (torch.tensor): (b*w,num_patches,2) Keypoint locations in pixel coordinates
            keypoint_scores (torch.tensor): (b*w,S,num_patches)
            keypoint_desc (torch.tensor): (b*w,C,num_patches)
        """
        BW, C, _, _ = descriptors.size()
        
        v_patches = F.unfold(self.v_coords.expand(BW, 1, self.width, self.width), kernel_size = self.patch_size,
                             stride = self.patch_size).to(self.gpuid)   # BW x patch_elems x num_patches
        u_patches = F.unfold(self.u_coords.expand(BW, 1, self.width, self.width), kernel_size = self.patch_size,
                             stride = self.patch_size).to(self.gpuid)   # BW x patch_elems x num_patches
        
        score_dim = weight_scores.size(1)
        detector_patches = F.unfold(detector_scores, kernel_size = self.patch_size, stride = self.patch_size)
        softmax_attention = F.softmax(detector_patches, dim = 1)    # BW x patch_elems x num_patches
        
        expected_v = torch.sum(v_patches * softmax_attention, dim = 1)  # BW x num_patches
        expected_u = torch.sum(u_patches * softmax_attention, dim = 1)  # BW x num_patches
        keypoint_coords = torch.stack([expected_u, expected_v], dim = 2)    # BW x num_patches x 2
        
        num_patches = keypoint_coords.size(1)

        norm_keypoints2D = normalize_coords(keypoint_coords, self.width, self.width).unsqueeze(1)   # BW x 1 x num_patches x 2

        keypoint_desc = F.grid_sample(descriptors, norm_keypoints2D, mode='bilinear', align_corners=True)   # BW x C x 1 x num_patches
        keypoint_desc = keypoint_desc.view(BW, C, num_patches)  # BW x C x num_patches

        keypoint_scores = F.grid_sample(weight_scores, norm_keypoints2D, mode='bilinear', align_corners=True)   # BW x S x 1 x num_patches
        keypoint_scores = keypoint_scores.view(BW, score_dim, num_patches)  # BW x S x num_patches

        return keypoint_coords, keypoint_scores, keypoint_desc


# Differentiable point matching
class SoftmaxMatcher(nn.Module):
    """
        Performs soft matching between keypoint descriptors and a dense map of descriptors.
        A temperature-weighted softmax is used which can approximate argmax at low temperatures.
    """
    def __init__(self, config):
        super().__init__()
        self.softmax_temp = config['networks']['matcher_block']['softmax_temp']
        self.window_size = config['window_size']
        self.gpuid = config['gpuid']

    def forward(self, keypoint_scores, keypoint_desc, scores_dense, desc_dense):
        """
        Args:
            keypoint_scores (torch.tensor): (b*w,1,N)
            keypoint_desc (torch.tensor): (b*w,C,N)
            scores_dense (torch.tensor): (b*w,1,H,W)
            desc_dense (torch.tensor): (b*w,C,H,W)
        Returns:
            pseudo_coords (torch.tensor): (b,N,2)
            match_weights (torch.tensor): (b,1,N)
            kp_inds (List[int]): length(b) indices along batch dimension for 'keypoint' data
        """
        BW, C, N = keypoint_desc.size()     # BW x C x N
        batch_size = int(BW / self.window_size)     # B = BW / W
        _, _, H, W = desc_dense.size()      # BW x C x H x W
        kp_inds, dense_inds = get_indices(batch_size, self.window_size) # BW -1

        src_desc = keypoint_desc[kp_inds]  # (BW -1) x C x N
        src_desc = F.normalize(src_desc, dim=1) # (BW -1) x C x N
        B = src_desc.size(0)

        tgt_desc_dense = desc_dense[dense_inds] # (BW -1) x C x H x W
        tgt_desc_unrolled = F.normalize(tgt_desc_dense.view(B, C, -1), dim=1)   # (BW -1) x C x HW

        match_vals = torch.matmul(src_desc.transpose(2, 1), tgt_desc_unrolled)  # B x N x HW
        soft_match_vals = F.softmax(match_vals / self.softmax_temp, dim=2)  # B x N x HW

        v_coord, u_coord = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        v_coord = v_coord.reshape(H * W).float()    # HW
        u_coord = u_coord.reshape(H * W).float()    # HW
        coords = torch.stack((u_coord, v_coord), dim=1)     # HW x 2
        
        tgt_coords_dense = coords.unsqueeze(0).expand(B, H * W, 2).to(self.gpuid)   # B x HW x 2

        pseudo_coords = torch.matmul(tgt_coords_dense.transpose(2, 1),
                                     soft_match_vals.transpose(2, 1)).transpose(2, 1)   # B x N x 2

        # GET SCORES for pseudo point locations
        pseudo_norm = normalize_coords(pseudo_coords, H, W).unsqueeze(1)    # B x 1 x N x 2
        tgt_scores_dense = scores_dense[dense_inds]     # B x 1 x H x W
        pseudo_scores = F.grid_sample(tgt_scores_dense, pseudo_norm, mode='bilinear', align_corners=True)   # B x 1 x 1 x N
        pseudo_scores = pseudo_scores.reshape(B, 1, N)  # B x 1 x N
        
        # GET DESCRIPTORS for pseudo point locations
        pseudo_desc = F.grid_sample(tgt_desc_dense, pseudo_norm, mode='bilinear')   # B x C x 1 x N
        pseudo_desc = pseudo_desc.reshape(B, C, N)      # B x C x N

        desc_match_score = torch.sum(src_desc * pseudo_desc, dim=1, keepdim=True) / float(C)    # B x 1 x N
        
        src_scores = keypoint_scores[kp_inds]

        match_weights = 0.5 * (desc_match_score + 1) * src_scores * pseudo_scores


        return pseudo_coords, match_weights, kp_inds, soft_match_vals
    


# Differentiable pose estimation
class SVD(torch.nn.Module):
    """
        Computes a 3x3 rotation matrix SO(3) and a 3x1 translation vector from pairs of 3D point clouds aligned
        according to known correspondences. The forward() method uses singular value decomposition to do this.
        This implementation is differentiable and follows the derivation from State Estimation for Robotics (Barfoot).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_size = config['window_size']
        self.gpuid = config['gpuid']
        self.linalg_svd = config['networks']['svd_block']['linalg_svd']
        self.weight_nms = config['networks']['svd_block']['weight_nms']
        
        self.dist_filter = config['networks']['svd_block']['dist_filter']
        self.pixel_distance_threshold = config['networks']['svd_block']['pixel_distance_threshold']
        self.dist_temperature = config['networks']['svd_block']['dist_temp']

    def forward(self, src_coords, tgt_coords, weights, convert_from_pixels=True):
        """ This modules used differentiable singular value decomposition to compute the rotations and translations that
            best align matched pointclouds (src and tgt).
        Args:
            src_coords (torch.tensor): (b,N,2) source keypoint locations
            tgt_coords (torch.tensor): (b,N,2) target keypoint locations
            weights (torch.tensor): (b,1,N) weight score associated with each src-tgt match
            convert_from_pixels (bool): if true, input is in pixel coordinates and must be converted to metric
        Returns:
            R_tgt_src (torch.tensor): (b,3,3) rotation from src to tgt
            t_src_tgt_intgt (torch.tensor): (b,3,1) translation from tgt to src as measured in tgt
        """
        if src_coords.size(0) > tgt_coords.size(0):
            BW = src_coords.size(0)
            B = int(BW / self.window_size)
            kp_inds, _ = get_indices(B, self.window_size)
            src_coords = src_coords[kp_inds]
        assert(src_coords.size() == tgt_coords.size())
        
        B = src_coords.size(0)  # B x N x 2
        
        # distance loss
        loss_l2 = torch.nn.MSELoss(reduction='none')
        dist_thred = self.pixel_distance_threshold
        dist_temp = self.dist_temperature
        
        dist_map = torch.sqrt(torch.sum(loss_l2(src_coords, tgt_coords), dim=2, keepdim=True))  # B X N X 1
        assert not torch.isnan(dist_map).any(), "dist_map contains NaN!"
        assert not torch.isinf(dist_map).any(), "dist_map contains Inf!"
        
        dist_overthred_prob = torch.sigmoid((dist_map - dist_thred) * dist_temp).transpose(1, 2)  # B X 1 X N
        if self.dist_filter:
            weights = weights * (1.0 - dist_overthred_prob)
                

        # pixel -> world
        if convert_from_pixels:
            src_coords = convert_to_radar_frame(src_coords, self.config)
            tgt_coords = convert_to_radar_frame(tgt_coords, self.config)
        
        # 2d -> 3d
        if src_coords.size(2) < 3:
            pad = 3 - src_coords.size(2)
            src_coords = F.pad(src_coords, [0, pad, 0, 0])
        if tgt_coords.size(2) < 3:
            pad = 3 - tgt_coords.size(2)
            tgt_coords = F.pad(tgt_coords, [0, pad, 0, 0])
        
        src_coords = src_coords.transpose(2, 1)     # B x 3 x N
        tgt_coords = tgt_coords.transpose(2, 1)     # B x 3 x N
        

        # Compute weighted centroids
        w = torch.sum(weights, dim=2, keepdim=True) + 1e-4
        src_centroid = torch.sum(src_coords * weights, dim=2, keepdim=True) / w     # B x 3 x 1
        tgt_centroid = torch.sum(tgt_coords * weights, dim=2, keepdim=True) / w     # B x 3 x 1

        # Center keypoint coordinates
        src_centered = src_coords - src_centroid    # B x 3 x N
        tgt_centered = tgt_coords - tgt_centroid    # B x 3 x N


        if not self.linalg_svd:
            S = torch.bmm(tgt_centered * weights, src_centered.transpose(2, 1)) / w  # B x 3 x 3
            
            # torch.svd sometimes has convergence issues
            try:
                U, _, V = torch.svd(S)
            except RuntimeError:
                print('Differentiable Pose Estimation SVD RuntimeError')
                print('S:\n', S)
                print('Adding turbulence to patch convergence issue')
                U, _, V = torch.svd(S + 1e-4 * S.mean() * torch.rand(1, 3).to(self.gpuid))

            det_UV = torch.det(U) * torch.det(V)
            # print(S[0])
            # print(det_UV)
            ones = torch.ones(B, 2).type_as(V)
            Sigma = torch.diag_embed(torch.cat((ones, det_UV.unsqueeze(1)), dim=1))  # B x 3 x 3

            # Compute rotation and translation (T_tgt_src)
            R_tgt_src = torch.bmm(U, torch.bmm(Sigma, V.transpose(2, 1)))  # B x 3 x 3
            
            t_tgt_src_insrc = src_centroid - torch.bmm(R_tgt_src.transpose(2, 1), tgt_centroid)  # B x 3 x 1
            t_src_tgt_intgt = -R_tgt_src.bmm(t_tgt_src_insrc)  # B x 3 x 1
        else:
            S = torch.bmm(src_centered * weights, tgt_centered.transpose(2, 1)) / w  # B x 3 x 3
            
            try:
                U, _, VT = torch.linalg.svd(S)
            except RuntimeError:
                print('Differentiable Pose Estimation SVD RuntimeError')
                U, _, V = torch.svd(S + 1e-4 * S.mean() * torch.rand(1, 3).to(self.gpuid))
            
            det_UV = torch.det(U) * torch.det(VT)
            ones = torch.ones(B, 2).type_as(VT)
            Sigma = torch.diag_embed(torch.cat((ones, det_UV.unsqueeze(1)), dim=1))  # B x 3 x 3

            # Compute rotation and translation (T_tgt_src)
            R_tgt_src = torch.bmm(VT.transpose(2, 1), torch.bmm(Sigma, U.transpose(2, 1)))  # B x 3 x 3
            
            t_src_tgt_intgt =  tgt_centroid - torch.bmm(R_tgt_src, src_centroid)    # B x 3 x 1

        
        return R_tgt_src, t_src_tgt_intgt