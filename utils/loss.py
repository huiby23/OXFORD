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
    
    # Get ground truth transforms
    kp_inds, _ = get_indices(batch_size, config['window_size'])
    T_tgt_src = T_21[kp_inds]
    R_tgt_src = T_tgt_src[:, :3, :3]
    t_tgt_src = T_tgt_src[:, :3, 3].unsqueeze(-1)
    
    identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(config['gpuid'])
    loss_fn = torch.nn.L1Loss()
    
    R_loss = loss_fn(torch.matmul(R_tgt_src_pred.transpose(2, 1), R_tgt_src), identity)
    t_loss = loss_fn(t_tgt_src_pred, t_tgt_src)
    
    svd_loss = t_loss + alpha * R_loss
    dict_loss = {'R_loss': R_loss, 't_loss': t_loss}
    
    return svd_loss, dict_loss

def unsupervised_loss(out, batch, config, solver):
    """This function uses the reprojection between matched pairs of points as a training signal.
        Transformations aligning pairs of frames are estimated using a non-differentiable estimator.
    Args:
        out (dict): The output of the DNN
        batch (dict): input data for the batch
        config (json): parsed config file
        solver: The steam_solver python wrapper class
    Returns:
        total_loss (float): unsupervised loss
        dict_loss (dict): a dictionary containing the separate loss components
    """
    src_coords = out['src']                 # (b*(w-1),N,2) src keypoint locations in metric
    tgt_coords = out['tgt']                 # (b*(w-1),N,2) tgt keypoint locations in metric
    
    match_weights = out['match_weights']    # (b*(w-1),S,N) match weights S=1=scalar, S=3=matrix
    keypoint_ints = out['keypoint_ints']    # (b*(w-1),1,N) 0==reject, 1==keep
    
    BW = keypoint_ints.size(0)
    window_size = config['window_size']
    batch_size = int(BW / (window_size - 1))
    
    gpuid = config['gpuid']
    mah_thres = config['steam']['mah_thres']
    expect_approx_opt = config['steam']['expect_approx_opt']
    topk_backup = config['steam']['topk_backup']
    
    T_aug = []
    if 'T_aug' in batch:
        T_aug = batch['T_aug']
    point_loss = 0
    logdet_loss = 0
    unweighted_point_loss = 0
    zeropad = torch.nn.ZeroPad2d((0, 1, 0, 0))

    # loop through each batch
    bcount = 0
    for b in range(batch_size):
        bcount += 1
        i = b * (window_size-1)    # first index of window
        # loop for each window frame
        for w in range(i, i + window_size - 1):
            # filter by zero intensity patches
            ids = torch.nonzero(keypoint_ints[w, 0] > 0, as_tuple=False).squeeze(1)
            if ids.size(0) == 0:
                print('WARNING: filtering by zero intensity patches resulted in zero keypoints!')
                continue

            # points must be list of N x 3
            points1 = zeropad(src_coords[w, ids]).unsqueeze(-1)    # N x 3 x 1
            points2 = zeropad(tgt_coords[w, ids]).unsqueeze(-1)    # N x 3 x 1
            weights_mat, weights_d = convert_to_weight_matrix(match_weights[w, :, ids].T, w, T_aug)
            ones = torch.ones(weights_mat.shape).to(gpuid)

            # get R_21 and t_12_in_2
            R_21 = torch.from_numpy(solver.poses[b, w-i+1][:3, :3]).to(gpuid).unsqueeze(0)
            t_12_in_2 = torch.from_numpy(solver.poses[b, w-i+1][:3, 3:4]).to(gpuid).unsqueeze(0)
            error = points2 - (R_21 @ points1 + t_12_in_2)
            mah2_error = error.transpose(1, 2) @ weights_mat @ error

            # error threshold
            errorT = mah_thres**2
            if errorT > 0:
                ids = torch.nonzero(mah2_error.squeeze() < errorT, as_tuple=False).squeeze()
            else:
                ids = torch.arange(mah2_error.size(0))

            if ids.squeeze().nelement() <= 1:
                print('Warning: MAH threshold output has 1 or 0 elements.')
                error2 = error.transpose(1, 2) @ error
                k = min(len(error2.squeeze()), topk_backup)
                _, ids = torch.topk(error2.squeeze(), k, largest=False)

            # squared mah error
            if expect_approx_opt == 0:
                # only mean
                point_loss += torch.mean(error[ids].transpose(1, 2) @ weights_mat[ids] @ error[ids])
                unweighted_point_loss += torch.mean(error[ids].transpose(1, 2) @ ones[ids] @ error[ids])
            elif expect_approx_opt == 1:
                # sigmapoints
                Rsp = torch.from_numpy(solver.poses_sp[b, w-i, :, :3, :3]).to(gpuid).unsqueeze(1)  # s x 1 x 3 x 3
                tsp = torch.from_numpy(solver.poses_sp[b, w-i, :, :3, 3:4]).to(gpuid).unsqueeze(1)  # s x 1 x 3 x 1

                points2 = points2[ids].unsqueeze(0)  # 1 x n x 3 x 1
                points1_in_2 = Rsp @ (points1[ids].unsqueeze(0)) + tsp  # s x n x 3 x 1
                error = points2 - points1_in_2  # s x n x 3 x 1
                temp = torch.sum(error.transpose(2, 3) @ weights_mat[ids].unsqueeze(0) @ error, dim=0)/Rsp.size(0)
                unweighted_point_loss += torch.mean(error.transpose(2, 3) @ ones[ids].unsqueeze(0) @ error)
                point_loss += torch.mean(temp)
            else:
                raise NotImplementedError('Steam loss method not implemented!')

            # log det (ignore 3rd dim since it's a constant)
            logdet_loss -= torch.mean(torch.sum(weights_d[ids, 0:2], dim=1))

    # average over batches
    if bcount > 0:
        point_loss /= (bcount * (solver.window_size - 1))
        logdet_loss /= (bcount * (solver.window_size - 1))
    total_loss = point_loss + logdet_loss
    dict_loss = {'point_loss': point_loss, 'logdet_loss': logdet_loss, 'unweighted_point_loss': unweighted_point_loss}
    
    return total_loss, dict_loss



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
        kp_inds, dense_inds = get_indices(batch_size, self.window_size) # B(W-1)

        src_desc = keypoint_desc[kp_inds]  # B x C x N
        src_desc = F.normalize(src_desc, dim=1) # B x C x N
        B = src_desc.size(0)

        tgt_desc_dense = desc_dense[dense_inds] # B x C x H x W
        tgt_desc_unrolled = F.normalize(tgt_desc_dense.view(B, C, -1), dim=1)   # B x C x HW

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
        pseudo_scores = F.grid_sample(tgt_scores_dense, pseudo_norm, mode='bilinear')   # B x 1 x 1 x N
        pseudo_scores = pseudo_scores.reshape(B, 1, N)  # B x 1 x N
        
        # GET DESCRIPTORS for pseudo point locations
        pseudo_desc = F.grid_sample(tgt_desc_dense, pseudo_norm, mode='bilinear')   # B x C x 1 x N
        pseudo_desc = pseudo_desc.reshape(B, C, N)      # B x C x N

        desc_match_score = torch.sum(src_desc * pseudo_desc, dim=1, keepdim=True) / float(C)    # B x 1 x N
        
        src_scores = keypoint_scores[kp_inds]

        match_weights = 0.5 * (desc_match_score + 1) * src_scores * pseudo_scores


        return pseudo_coords, match_weights, kp_inds, soft_match_vals



class SoftmaxRefMatcher(nn.Module):
    """
        Performs soft matching between keypoint descriptors and a dense map of descriptors.
        A temperature-weighted softmax is used which can approximate argmax at low temperatures.
    """
    def __init__(self, config):
        super().__init__()
        self.softmax_temp = config['networks']['matcher_block']['softmax_temp']
        self.sparse = config['networks']['matcher_block']['sparse']
        self.window_size = config['window_size']
        self.gpuid = config['gpuid']
        self.width = config['cart_pixel_width']
        v_coord, u_coord = torch.meshgrid([torch.arange(0, self.width), torch.arange(0, self.width)])
        v_coord = v_coord.reshape(self.width**2).float()  # HW
        u_coord = u_coord.reshape(self.width**2).float()
        coords = torch.stack((u_coord, v_coord), dim=1)  # HW x 2
        self.src_coords_dense = coords.unsqueeze(0).to(self.gpuid)  # 1 x HW x 2

    def forward(self, keypoint_scores, keypoint_desc, desc_dense, keypoint_coords):
        """
        Args:
            keypoint_scores: (b*w,S,N)
            keypoint_desc: (b*w,C,N)
            desc_dense: (b*w,C,H,W)
        Returns:
            pseudo_coords (torch.tensor): (b*(w-1),N,2)
            match_weights (torch.tensor): (b*(w-1),S,N)
            tgt_ids (torch.tensor): (b*(w-1),) indices along batch dimension for target data
            src_ids (torch.tensor): (b*(w-1),) indices along batch dimension for source data
        """
        BW, encoder_dim, n_points = keypoint_desc.size()
        B = int(BW / self.window_size)
        src_desc_dense = desc_dense[::self.window_size]
        src_desc_unrolled = F.normalize(src_desc_dense.view(B, encoder_dim, -1), dim=1)  # B x C x HW
        # build pseudo_coords
        pseudo_coords = torch.zeros((B * (self.window_size - 1), n_points, 2),
                                    device=self.gpuid)  # B*(window - 1) x N x 2
        tgt_ids = torch.zeros(B * (self.window_size - 1), dtype=torch.int64, device=self.gpuid)    # B*(window - 1)
        src_ids = torch.zeros(B * (self.window_size - 1), dtype=torch.int64, device=self.gpuid)    # B*(window - 1)
        # loop for each batch
        if not self.sparse:
            for i in range(B):
                win_ids = torch.arange(i * self.window_size + 1, i * self.window_size + self.window_size).to(self.gpuid)
                tgt_desc = keypoint_desc[win_ids]  # (window - 1) x C x N
                tgt_desc = F.normalize(tgt_desc, dim=1)
                match_vals = torch.matmul(tgt_desc.transpose(2, 1), src_desc_unrolled[i:i+1])  # (window - 1) x N x HW
                soft_match_vals = F.softmax(match_vals / self.softmax_temp, dim=2)  # (window - 1) x N x HW
                pseudo_ids = torch.arange(i * (self.window_size - 1), i * (self.window_size - 1) + self.window_size - 1)
                pseudo_coords[pseudo_ids] = torch.matmul(self.src_coords_dense.transpose(2, 1),
                                                         soft_match_vals.transpose(2, 1)).transpose(2, 1)  # (w-1)xNx2
                tgt_ids[pseudo_ids] = win_ids
                src_ids[pseudo_ids] = i * self.window_size
        else:
            for i in range(B):
                win_ids = torch.arange(i * self.window_size + 1, i * self.window_size + self.window_size).to(self.gpuid)
                tgt_desc = keypoint_desc[win_ids]
                src_desc = keypoint_desc[i*self.window_size:i*self.window_size+1]
                tgt_desc = F.normalize(tgt_desc, dim=1)
                src_desc = F.normalize(src_desc, dim=1)
                match_vals = torch.matmul(tgt_desc.transpose(2, 1), src_desc)
                soft_match_vals = F.softmax(match_vals / self.softmax_temp, dim=2)
                src_coords = keypoint_coords[i*self.window_size:i*self.window_size+1]
                pseudo_ids = torch.arange(i * (self.window_size - 1), i * (self.window_size - 1) + self.window_size - 1)
                pseudo_coords[pseudo_ids] = torch.matmul(src_coords.transpose(2, 1), soft_match_vals.transpose(2, 1)).transpose(2, 1)
                tgt_ids[pseudo_ids] = win_ids
                src_ids[pseudo_ids] = i * self.window_size

        return pseudo_coords, keypoint_scores[tgt_ids], tgt_ids, src_ids


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
            t_tgt_src_insrc (torch.tensor): (b,3,1) translation from src to tgt as measured in src
            t_src_tgt_intgt (torch.tensor): (b,3,1) translation from tgt to src as measured in tgt
        """
        if src_coords.size(0) > tgt_coords.size(0):
            BW = src_coords.size(0)
            B = int(BW / self.window_size)
            kp_inds, _ = get_indices(B, self.window_size)
            src_coords = src_coords[kp_inds]
        assert(src_coords.size() == tgt_coords.size())
        
        B = src_coords.size(0)  # B x N x 2
        
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

        S = torch.bmm(tgt_centered * weights, src_centered.transpose(2, 1)) / w  # B x 3 x 3
        # S = torch.bmm(src_centered * weights, tgt_centered.transpose(2, 1)) / w  # B x 3 x 3

        if not self.linalg_svd:
            # torch.svd sometimes has convergence issues
            try:
                U, _, V = torch.svd(S)
            except RuntimeError:
                print('Differentiable Pose Estimation SVD RuntimeError')
                print('S:\n', S)
                print('Adding turbulence to patch convergence issue')
                U, _, V = torch.svd(S + 1e-4 * S.mean() * torch.rand(1, 3).to(self.gpuid))

            det_UV = torch.det(U) * torch.det(V)
            ones = torch.ones(B, 2).type_as(V)
            Sigma = torch.diag_embed(torch.cat((ones, det_UV.unsqueeze(1)), dim=1))  # B x 3 x 3

            # Compute rotation and translation (T_tgt_src)
            R_tgt_src = torch.bmm(U, torch.bmm(Sigma, V.transpose(2, 1)))  # B x 3 x 3
            
            t_tgt_src_insrc = src_centroid - torch.bmm(R_tgt_src.transpose(2, 1), tgt_centroid)  # B x 3 x 1
            t_src_tgt_intgt = -R_tgt_src.bmm(t_tgt_src_insrc)  # B x 3 x 1
        else:
            U, _, VT = torch.linalg.svd(S)

            det_UV = torch.det(U) * torch.det(VT)
            ones = torch.ones(B, 2).type_as(VT)
            Sigma = torch.diag_embed(torch.cat((ones, det_UV.unsqueeze(1)), dim=1))  # B x 3 x 3

            # Compute rotation and translation (T_tgt_src)
            R_tgt_src = torch.bmm(VT.transpose(2, 1), torch.bmm(Sigma, U.transpose(2, 1)))  # B x 3 x 3
            
            t_tgt_src_insrc =  tgt_centroid - torch.bmm(R_tgt_src, src_centroid)    # B x 3 x 1
            t_src_tgt_intgt = -R_tgt_src.bmm(t_tgt_src_insrc)  # B x 3 x 1

        
        return R_tgt_src, t_src_tgt_intgt