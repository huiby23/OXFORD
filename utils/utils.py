import io
import torch
import pickle
import PIL.Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision.transforms import ToTensor


def get_inverse_tf(T):
    """Returns the inverse of a given 4x4 homogeneous transform.
    Args:
        T (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: inv(T)
    """
    T2 = np.identity(4, dtype=np.float32)
    R = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)
    T2[0:3, 0:3] = R.transpose()
    T2[0:3, 3:] = np.matmul(-1 * R.transpose(), t)
    return T2


def get_transform(x, y, theta):
    """Returns a 4x4 homogeneous 3D transform for a given 2D (x, y, theta).
    Args:
        x (float): x-translation
        y (float): y-translation
        theta (float): rotation
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    T = np.identity(4, dtype=np.float32)
    T[0:2, 0:2] = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    T[0, 3] = x
    T[1, 3] = y
    return T


def get_transform2(R, t):
    """Returns a 4x4 homogeneous 3D transform
    Args:
        R (np.ndarray): (3,3) rotation matrix
        t (np.ndarray): (3,1) translation vector
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    T = np.identity(4, dtype=np.float32)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.squeeze()
    return T


def enforce_orthog(T, dim=3):
    """Enforces orthogonality of a 3x3 rotation matrix within a 4x4 homogeneous transformation matrix.
    Args:
        T (np.ndarray): 4x4 transformation matrix
        dim (int): dimensionality of the transform 2==2D, 3==3D
    Returns:
        np.ndarray: 4x4 transformation matrix with orthogonality conditions on the rotation matrix enforced.
    """
    if dim == 2:
        if abs(np.linalg.det(T[0:2, 0:2]) - 1) < 1e-10:
            return T
        R = T[0:2, 0:2]
        epsilon = 0.001
        if abs(R[0, 0] - R[1, 1]) > epsilon or abs(R[1, 0] + R[0, 1]) > epsilon:
            print("WARNING: this is not a proper rigid transformation:", R)
            return T
        a = (R[0, 0] + R[1, 1]) / 2
        b = (-R[1, 0] + R[0, 1]) / 2
        s = np.sqrt(a**2 + b**2)
        a /= s
        b /= s
        R[0, 0] = a
        R[0, 1] = b
        R[1, 0] = -b
        R[1, 1] = a
        T[0:2, 0:2] = R
    if dim == 3:
        if abs(np.linalg.det(T[0:3, 0:3]) - 1) < 1e-10:
            return T
        c1 = T[0:3, 1]
        c2 = T[0:3, 2]
        c1 /= np.linalg.norm(c1)
        c2 /= np.linalg.norm(c2)
        newcol0 = np.cross(c1, c2)
        newcol1 = np.cross(c2, newcol0)
        T[0:3, 0] = newcol0
        T[0:3, 1] = newcol1
        T[0:3, 2] = c2
    return T


def rotationError(T):
    """Calculates a single rotation value corresponding to the upper-left 3x3 rotation matrix.
        Uses axis-angle representation to get a single number for rotation
    Args:
        T (np.ndarray): 4x4 transformation matrix T = [C, r; 0 0 0 1]
    Returns:
        float: rotation
    """
    d = 0.5 * (np.trace(T[0:3, 0:3]) - 1)
    return np.arccos(max(min(d, 1.0), -1.0))


def translationError(T, dim=2):
    """Calculates a euclidean distance corresponding to the translation vector within a 4x4 transform.
    Args:
        T (np.ndarray): 4x4 transformation matrix T = [C, r; 0 0 0 1]
        dim (int): If dim=2 we only use x,y, otherwise we use all dims.
    Returns:
        float: translation distance
    """
    if dim == 2:
        return np.sqrt(T[0, 3]**2 + T[1, 3]**2)
    return np.sqrt(T[0, 3]**2 + T[1, 3]**2 + T[2, 3]**2)


def computeMedianError(T_gt, T_pred):
    """Computes the median translation and rotation errors along with their standard deviations.
    Args:
        T_gt (List[np.ndarray]): each entry in list is 4x4 transformation matrix
        T_pred (List[np.ndarray]): each entry in list is 4x4 transformation matrix
    Returns:
        t_err_med (float): median translation error
        t_err_std (float): standard dev translation error
        r_err_med (float): median rotation error
        r_err_std (float): standard dev rotation error
        t_err_mean (float): mean translation error
        r_err_mean (float): mean rotation error
        t_error (List[float]): list of all translation errors
        r_error (List[float]): list of all rotation errors
    """
    t_error = []
    r_error = []
    for i, T in enumerate(T_gt):
        T_error = np.matmul(T, get_inverse_tf(T_pred[i]))
        t_error.append(translationError(T_error))
        r_error.append(180 * rotationError(T_error) / np.pi)
    t_error = np.array(t_error)
    r_error = np.array(r_error)
    return [np.median(t_error), np.std(t_error), np.median(r_error), np.std(r_error), np.mean(t_error),
            np.mean(r_error), t_error, r_error]


def trajectoryDistances(poses):
    """Calculates path length along the trajectory.
    Args:
        poses (List[np.ndarray]): list of 4x4 poses (T_2_1 from current to next)
    Returns:
        List[float]: distance along the trajectory, increasing as a function of time / list index
    """
    dist = [0]
    for i in range(1, len(poses)):
        P1 = get_inverse_tf(poses[i - 1])
        P2 = get_inverse_tf(poses[i])
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dist.append(dist[i-1] + np.sqrt(dx**2 + dy**2))
    return dist


def lastFrameFromSegmentLength(dist, first_frame, length):
    """Retrieves the index of the last frame for our current analysis.
        last_frame should be 'dist' meters away from first_frame in terms of distance traveled along the trajectory.
    Args:
        dist (List[float]): distance along the trajectory, increasing as a function of time / list index
        first_frame (int): index of the starting frame for this sequence
        length (float): length of the current segment being evaluated
    Returns:
        last_frame (int): index of the last frame in this segment
    """
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + length:
            return i
    return -1


def calcSequenceErrors(poses_gt, poses_pred):
    """Calculate the translation and rotation error for each subsequence across several different lengths.
    Args:
        T_gt (List[np.ndarray]): each entry in list is 4x4 transformation matrix, ground truth transforms
        T_pred (List[np.ndarray]): each entry in list is 4x4 transformation matrix, predicted transforms
    Returns:
        err (List[Tuple]) each entry in list is [first_frame, r_err, t_err, length, speed]
    """
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    err = []
    step_size = 4  # Every second
    # Pre-compute distances from ground truth as reference
    dist = trajectoryDistances(poses_gt)

    for first_frame in range(0, len(poses_gt), step_size):
        for length in lengths:
            last_frame = lastFrameFromSegmentLength(dist, first_frame, length)
            if last_frame == -1:
                continue
            # Compute rotational and translation errors
            pose_delta_gt = np.matmul(poses_gt[last_frame], get_inverse_tf(poses_gt[first_frame]))
            pose_delta_res = np.matmul(poses_pred[last_frame], get_inverse_tf(poses_pred[first_frame]))
            pose_error = np.matmul(pose_delta_gt, get_inverse_tf(pose_delta_res))
            r_err = rotationError(pose_error)
            t_err = translationError(pose_error)
            # Approx speed
            num_frames = float(last_frame - first_frame + 1)
            speed = float(length) / (0.25 * num_frames)
            err.append([first_frame, r_err/float(length), t_err/float(length), length, speed])
    return err


def getStats(err):
    """Computes the average translation and rotation within a sequence (across subsequences of diff lengths)."""
    t_err = 0
    r_err = 0
    for e in err:
        t_err += e[2]
        r_err += e[1]
    t_err /= float(len(err))
    r_err /= float(len(err))
    return t_err, r_err


def computeKittiMetrics(T_gt, T_pred, seq_lens):
    """Computes the translational (%) and rotational drift (deg/m) in the KITTI style.
        KITTI rotation and translation metrics are computed for each sequence individually and then
        averaged across the sequences.
    Args:
        T_gt (List[np.ndarray]): List of 4x4 homogeneous transforms (Frame t to Frame t+1)
        T_pred (List[np.ndarray]): List of 4x4 homogeneous transforms (Frame t to Frame t+1)
        seq_lens (List[int]): List of sequence lengths
    Returns:
        t_err: Average KITTI Translation ERROR (%)
        r_err: Average KITTI Rotation Error (deg / m)
    """
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)
    err_list = []
    for indices in seq_indices:
        T_gt_ = np.identity(4)
        T_pred_ = np.identity(4)
        poses_gt = []
        poses_pred = []
        for i in indices:
            T_gt_ = np.matmul(T_gt[i], T_gt_)
            T_pred_ = np.matmul(T_pred[i], T_pred_)
            enforce_orthog(T_gt_)
            enforce_orthog(T_pred_)
            poses_gt.append(T_gt_)
            poses_pred.append(T_pred_)
        err = calcSequenceErrors(poses_gt, poses_pred)
        t_err, r_err = getStats(err)
        err_list.append([t_err, r_err])
    err_list = np.asarray(err_list)
    avg = np.mean(err_list, axis=0)
    t_err = avg[0]
    r_err = avg[1]
    return t_err * 100, r_err * 180 / np.pi


def saveKittiErrors(err, fname):
    pickle.dump(err, open(fname, 'wb'))


def loadKittiErrors(fname):
    return pickle.load(open(fname, 'rb'))


def save_in_yeti_format(T_gt, T_pred, timestamps, seq_lens, seq_names, root='./'):
    """This function converts outputs to a file format that is backwards compatible with the yeti repository.
    Args:
        T_gt (List[np.ndarray]): each entry in list is 4x4 transformation matrix, ground truth transforms
        T_pred (List[np.ndarray]): each entry in list is 4x4 transformation matrix, predicted transforms
        seq_lens (List[int]): length of each sequence in number of frames
        seq_names (List[AnyStr]): name of each sequence
        root (AnyStr): name of the root data folder
    """
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)

    for s, indices in enumerate(seq_indices):
        fname = root + 'accuracy' + seq_names[s] + '.csv'
        with open(fname, 'w') as f:
            f.write('x,y,yaw,gtx,gty,gtyaw,time1,time2\n')
            for i in indices:
                R_pred = T_pred[i][:3, :3]
                t_pred = T_pred[i][:3, 3:]
                yaw = -1 * np.arcsin(R_pred[0, 1])
                gtyaw = -1 * np.arcsin(T_gt[i][0, 1])
                t = np.matmul(-1 * R_pred.transpose(), np.reshape(t_pred, (3, 1)))
                T = get_inverse_tf(T_gt[i])
                f.write('{},{},{},{},{},{},{},{}\n'.format(t[0, 0], t[1, 0], yaw, T[0, 3], T[1, 3], gtyaw,
                                                           timestamps[i][0], timestamps[i][1]))


def load_icra21_results(results_loc, seq_names, seq_lens):
    """Loads ICRA 2021 results for MC-RANSAC (Burnett et al.) on the Oxford Radar Dataset.
    Args:
        results_loc (AnyStr): path to the folder containing the results.
        seq_names (List[AnyStr]): names of the sequences that we want results for.
        seq_lens (List[int]): length of each sequence in number of frames
    Returns:
        T_icra (List[np.ndarray]): each entry in list is 4x4 transformation matrix, MC-RANSAC results.
    """
    T_icra = []
    for i, seq_name in enumerate(seq_names):
        fname = results_loc + 'accuracy' + seq_name + '.csv'
        with open(fname, 'r') as f:
            f.readline()  # Clear out the header
            lines = f.readlines()
            count = 0
            for line in lines:
                line = line.split(',')
                # Retrieve the transform estimated by MC-RANSAC + DOPPLER compensation
                T_icra.append(get_inverse_tf(get_transform(float(line[11]), float(line[12]), float(line[13]))))
                count += 1
            # Append identity transforms at the end in case the ICRA results ended early by a couple frames
            if count < seq_lens[i]:
                print('WARNING: ICRA results shorter than seq_len by {}. Append last TF.'.format((seq_lens[i] - count)))
            while count < seq_lens[i]:
                T_icra.append(T_icra[-1])
                count += 1
    return T_icra


def normalize_coords(coords_2D, width, height):
    """Normalizes coords_2D (BW x N x 2) in pixel coordinates to be within [-1, 1]
    Args:
        coords_2D (torch.tensor): (b*w,N,2)
        width (float): width of the image in pixels
        height (float): height of the image in pixels
    Returns:
        torch.tensor: (b*w,N,2) coordinates normalized to be within [-1,1]
    """
    batch_size = coords_2D.size(0)
    u_norm = (2 * coords_2D[:, :, 0].reshape(batch_size, -1) / (width - 1)) - 1
    v_norm = (2 * coords_2D[:, :, 1].reshape(batch_size, -1) / (height - 1)) - 1
    return torch.stack([u_norm, v_norm], dim=2)  # BW x num_patches x 2


def convert_to_radar_frame(pixel_coords, config):
    """Converts pixel_coords (B x N x 2) from pixel coordinates to metric coordinates in the radar frame.
    Args:
        pixel_coords (torch.tensor): (B,N,2) pixel coordinates
        config (json): parse configuration file
    Returns:
        torch.tensor: (B,N,2) metric coordinates
    """
    cart_pixel_width = config['cart_pixel_width']
    cart_resolution = config['cart_resolution']
    gpuid = config['gpuid']
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    B, N, _ = pixel_coords.size()   # (u, v)
    R = torch.tensor([[0, -cart_resolution], [cart_resolution, 0]]).expand(B, 2, 2).to(gpuid)
    t = torch.tensor([[cart_min_range], [-cart_min_range]]).expand(B, 2, N).to(gpuid)
    return (torch.bmm(R, pixel_coords.transpose(2, 1)) + t).transpose(2, 1)     # (x, y)


def get_indices(batch_size, window_size):
    """Retrieves batch indices for for source and target frames.
       This is intended to be used with the Oxford_Radar model.
    """
    src_ids = []
    tgt_ids = []
    for i in range(batch_size):
        for j in range(window_size - 1):
            idx = i * window_size + j
            src_ids.append(idx)
            tgt_ids.append(idx + 1)
    
    return src_ids, tgt_ids

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_T_ba(out, a, b):
    """Retrieves the transformation matrix from a to b given the output of the DNN.
    Args:
        out (dict): output of the DNN, contains the predicted transforms.
        a (int): index of the start frame
        b (int): index of the end frame
    Returns:
        np.ndarray: 4x4 transformation matrix T_ba from a to b
    """
    T_b0 = np.eye(4)
    T_b0[:3, :3] = out['R'][0, b].detach().cpu().numpy()
    T_b0[:3, 3:4] = out['t'][0, b].detach().cpu().numpy()
    T_a0 = np.eye(4)
    T_a0[:3, :3] = out['R'][0, a].detach().cpu().numpy()
    T_a0[:3, 3:4] = out['t'][0, a].detach().cpu().numpy()
    return np.matmul(T_b0, get_inverse_tf(T_a0))


def convert_to_weight_matrix(w, window_id, T_aug=[]):
    """This function converts the S-dimensional weights estimated for each keypoint into
        a 2x2 weight (inverse covariance) matrix for each keypoint.
        If S = 1, Wout = diag(exp(w), exp(w), 1e4)
        If S = 3, use LDL^T to obtain 2x2 covariance, place on top-LH corner. 1e4 bottom-RH corner.
    Args:
        w (torch.tensor): (n_points, S), S = score_dim, S=1=scalar, S=3=matrix
        window_id (int): index of the window currently being analyzed
        T_aug (torch.tensor): optional argument which is passed from data augmention with random rotations.
    Returns:
        A (torch.tensor): (n_points,3,3) 3x3 weight matrices
        d (torch.tensor): (n_points, 3) 3-dim weight vectors (diagonal of the weight matrices)
    """
    z_weight = 9.2103  # 9.2103 = log(1e4), 1e4 is inverse variance of 1cm std dev
    if w.size(1) == 1:
        # scalar weight
        A = torch.zeros(w.size(0), 9, device=w.device)
        A[:, (0, 4)] = torch.exp(w)
        A[:, 8] = torch.exp(torch.tensor(z_weight))
        A = A.reshape((-1, 3, 3))
        d = torch.zeros(w.size(0), 3, device=w.device)
        d[:, 0:2] += w
        d[:, 2] += z_weight
    elif w.size(1) == 3:
        # 2x2 matrix
        L = torch.zeros(w.size(0), 4, device=w.device)
        L[:, (0, 3)] = 1
        L[:, 2] = w[:, 0]
        L = L.reshape((-1, 2, 2))
        D = torch.zeros(w.size(0), 4, device=w.device)
        D[:, (0, 3)] = torch.exp(w[:, 1:])
        D = D.reshape((-1, 2, 2))
        A2x2 = L @ D @ L.transpose(1, 2)

        if T_aug:  # if list is not empty
            Rot = T_aug[window_id].to(w.device)[:2, :2].unsqueeze(0)
            A2x2 = Rot.transpose(1, 2) @ A2x2 @ Rot

        A = torch.zeros(w.size(0), 3, 3, device=w.device)
        A[:, 0:2, 0:2] = A2x2
        A[:, 2, 2] = torch.exp(torch.tensor(z_weight))
        d = torch.ones(w.size(0), 3, device=w.device)*z_weight
        d[:, 0:2] = w[:, 1:]
    else:
        assert False, "Weight scores should be dim 1 or 3"

    return A, d


def wrapto2pi(phi):
    """Ensures that the output angle phi is within the interval [0, 2*pi)"""
    if phi < 0:
        return phi + 2 * np.pi * np.ceil(phi / (-2 * np.pi))
    elif phi >= 2 * np.pi:
        return (phi / (2 * np.pi) % 1) * 2 * np.pi
    return phi


def getApproxTimeStamps(points, times, flip_y=False):
    """Retrieves the approximate timestamp of each target point.
    Args:
        points (List[np.ndarray]): each entry in list is (N,2)
        times (List[np.ndarray]): each entry in list is (400,) corresponding to timestamps from sensor for each azimuth
    Returns:
        List[np.ndarray]: each entry in list is (N,) corresponding to the interpolated time for each measurement.
    """
    azimuth_step = (2 * np.pi) / 400
    timestamps = []
    for i, p in enumerate(points):
        p = points[i]
        ptimes = times[i]
        delta_t = ptimes[-1] - ptimes[-2]
        ptimes = np.append(ptimes, int(ptimes[-1] + delta_t))
        point_times = []
        for k in range(p.shape[0]):
            x = p[k, 0]
            y = p[k, 1]
            if flip_y:
                y *= -1
            phi = np.arctan2(y, x)
            phi = wrapto2pi(phi)
            time_idx = phi / azimuth_step
            t1 = ptimes[int(np.floor(time_idx))]
            t2 = ptimes[int(np.ceil(time_idx))]
            # interpolate to get slightly more precise timestamp
            ratio = time_idx % 1
            t = int(t1 + ratio * (t2 - t1))
            point_times.append(t)
        timestamps.append(np.array(point_times))
    return timestamps


def undistort_pointcloud(points, point_times, t_refs, solver):
    """Removes motion distortion from pointclouds.
    Args:
        points (List[np.ndarray]): each entry in list is (N, 2-4)
        point_times (List[np.ndarray]): each entry in list is (N,) timestamp for each point
        t_refs (List[int]): Reference time for each pointcloud, transform each point into the sensor frame at this time.
    Returns:
        List[np.ndarray]: pointclouds with motion distortion removed.
    """
    for i, p in enumerate(points):
        p = points[i]
        ptimes = point_times[i]
        t_ref = t_refs[i]
        for j, ptime in enumerate(ptimes):
            T_0a = np.identity(4, dtype=np.float32)
            solver.getPoseBetweenTimes(T_0a, ptime, t_ref)
            pbar = T_0a @ p[j].reshape(4, 1)
            p[j, :] = pbar[:]
        points[i] = p
    return points



# visualization functions
def convert_plt_to_img():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return PIL.Image.open(buf)


def convert_plt_to_tensor():
    return ToTensor()(convert_plt_to_img())


def draw_batch(batch, out, config):
    """Creates an image of the radar scan, scores, and keypoint matches for a single batch."""
    # Draw radar image
    radar = batch['data'][0].squeeze().numpy()
    plt.subplots()
    plt.imshow(radar, cmap='gray')
    radar_img = convert_plt_to_tensor()

    # Draw keypoint matches
    src = out['src'][0].squeeze().detach().cpu().numpy()
    tgt = out['tgt'][0].squeeze().detach().cpu().numpy()
    match_weights = out['match_weights'][0].squeeze().detach().cpu().numpy()

    nms = config['vis_keypoint_nms']
    max_w = np.max(match_weights)
    plt.imshow(radar, cmap='gray')
    for i in range(src.shape[0]):
        if match_weights[i] < nms * max_w:
            continue
        plt.plot([src[i, 0], tgt[i, 0]], [src[i, 1], tgt[i, 1]], c='w', linewidth=2, zorder=2)
        plt.scatter(src[i, 0], src[i, 1], c='g', s=5, zorder=3)
        plt.scatter(tgt[i, 0], tgt[i, 1], c='r', s=5, zorder=4)
    match_img = convert_plt_to_tensor()

    # Draw scores
    scores = out['scores'][0].squeeze().detach().cpu().numpy()
    plt.subplots()
    plt.imshow(scores, cmap='inferno')
    score_img = convert_plt_to_tensor()

    return vutils.make_grid([radar_img, score_img, match_img])


def draw_matches(batch, out, config, solver):
    azimuth_step = (2 * np.pi) / 400
    cart_pixel_width = config['cart_pixel_width']
    cart_resolution = config['cart_resolution']
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    T_met_pix = np.array([[0, -cart_resolution, 0, cart_min_range],
                          [cart_resolution, 0, 0, -cart_min_range],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    T_pix_met = np.linalg.inv(T_met_pix)

    keypoint_ints = out['keypoint_ints']
    ids = torch.nonzero(keypoint_ints[0, 0] > 0, as_tuple=False).squeeze(1)
    src = out['src_rc'][0, ids].squeeze().detach().cpu().numpy()
    tgt = out['tgt_rc'][0, ids].squeeze().detach().cpu().numpy()
    radar = batch['data'][0].squeeze().numpy()
    _, axs = plt.subplots(1, 3, tight_layout=True)
    # Raw locations overlayed, no transforms
    axs[0].imshow(radar, cmap='gray', extent=(0, 640, 640, 0), interpolation='none')
    axs[0].set_axis_off()
    axs[0].set_title('raw')
    for i in range(src.shape[0]):
        axs[0].plot([src[i, 0], tgt[i, 0]], [src[i, 1], tgt[i, 1]], c='w', linewidth=1, zorder=2)
        axs[0].scatter(src[i, 0], src[i, 1], c='limegreen', s=2, zorder=3)
        axs[0].scatter(tgt[i, 0], tgt[i, 1], c='r', s=2, zorder=4)

    src = out['src'][0, ids].squeeze().detach().cpu().numpy()
    tgt = out['tgt'][0, ids].squeeze().detach().cpu().numpy()
    # Use Rigid Transform
    axs[1].imshow(radar, cmap='gray', extent=(0, 640, 640, 0), interpolation='none')
    axs[1].set_axis_off()
    axs[1].set_title('rigid')
    T_tgt_src = get_T_ba(out, a=0, b=1)
    error = np.zeros((src.shape[0], 2))
    for i in range(src.shape[0]):
        x1 = np.array([src[i, 0], src[i, 1], 0, 1]).reshape(4, 1)
        x2 = np.array([tgt[i, 0], tgt[i, 1], 0, 1]).reshape(4, 1)
        x1 = T_tgt_src @ x1
        e = x1 - x2
        error[i, 1] = np.sqrt(e.T @ e)
        error[i, 0] = int(wrapto2pi(np.arctan2(x2[1, 0], x2[0, 0])) // azimuth_step)
        x1 = T_pix_met @ x1
        x2 = T_pix_met @ x2
        axs[1].plot([x1[0, 0], x2[0, 0]], [x1[1, 0], x2[1, 0]], c='w', linewidth=1, zorder=2)
        axs[1].scatter(x1[0, 0], x1[1, 0], c='limegreen', s=2, zorder=3)
        axs[1].scatter(x2[0, 0], x2[1, 0], c='r', s=2, zorder=4)

    # Use Interpolated Poses
    t1 = batch['timestamps'][0].numpy().squeeze()
    t2 = batch['timestamps'][1].numpy().squeeze()
    times1 = getApproxTimeStamps([src], [t1])[0]
    times2 = getApproxTimeStamps([tgt], [t2])[0]
    t_refs = batch['t_ref'].numpy()

    T_1a = np.identity(4, dtype=np.float32)
    T_1b = np.identity(4, dtype=np.float32)
    axs[2].imshow(radar, cmap='gray', extent=(0, 640, 640, 0), interpolation='none')
    axs[2].set_axis_off()
    axs[2].set_title('interp')
    error2 = np.zeros((src.shape[0], 2))
    for i in range(src.shape[0]):
        solver.getPoseBetweenTimes(T_1a, times1[i], t_refs[1, 0, 0])
        solver.getPoseBetweenTimes(T_1b, times2[i], t_refs[1, 0, 0])
        x1 = np.array([src[i, 0], src[i, 1], 0, 1]).reshape(4, 1)
        x2 = np.array([tgt[i, 0], tgt[i, 1], 0, 1]).reshape(4, 1)
        x1 = T_1a @ x1
        x2 = T_1b @ x2
        e = x1 - x2
        error2[i, 1] = np.sqrt(e.T @ e)
        error2[i, 0] = int(wrapto2pi(np.arctan2(x2[1, 0], x2[0, 0])) // azimuth_step)
        x1 = T_pix_met @ x1
        x2 = T_pix_met @ x2
        axs[2].plot([x1[0, 0], x2[0, 0]], [x1[1, 0], x2[1, 0]], c='w', linewidth=1, zorder=2)
        axs[2].scatter(x1[0, 0], x1[1, 0], c='limegreen', s=2, zorder=3)
        axs[2].scatter(x2[0, 0], x2[1, 0], c='r', s=2, zorder=4)

    plt.savefig('matches.pdf', bbox_inches='tight', pad_inches=0.0)
    plt.figure()

    idx = np.argsort(error[:, 0])
    error = error[idx, :]
    idx = np.argsort(error2[:, 0])
    error2 = error2[idx, :]
    plt.plot(error[:, 0], error[:, 1], color='b', label='raw error', linewidth=1)
    plt.plot(error2[:, 0], error2[:, 1], color='r', label='interp error', linewidth=1)
    plt.title('raw error')
    plt.legend()
    plt.savefig('matches2.pdf', bbox_inches='tight', pad_inches=0.0)


def draw_batch_steam(batch, out, config):
    """Creates an image of the radar scan, scores, and keypoint matches for a single batch."""
    # Draw radar image
    radar = batch['data'][0].squeeze().numpy()
    radar_tgt = batch['data'][-1].squeeze().numpy()
    plt.imshow(np.concatenate((radar, radar_tgt), axis=1), cmap='gray')
    plt.title('radar src-tgt pair')
    radar_img = convert_plt_to_tensor()

    # Draw keypoint matches
    src = out['src_rc'][-1].squeeze().detach().cpu().numpy()
    tgt = out['tgt_rc'][-1].squeeze().detach().cpu().numpy()
    keypoint_ints = out['keypoint_ints']

    ids = torch.nonzero(keypoint_ints[-1, 0] > 0, as_tuple=False).squeeze(1)
    ids_cpu = ids.cpu()

    plt.imshow(np.concatenate((radar, radar_tgt), axis=1), cmap='gray')
    delta = radar.shape[1]
    for i in range(src.shape[0]):
        if i in ids_cpu:
            custom_colour = 'g'
            plt.plot([src[i, 0], tgt[i, 0] + delta], [src[i, 1], tgt[i, 1]], c='y', linewidth=0.5, zorder=2)
            plt.scatter(src[i, 0], src[i, 1], c=custom_colour, s=5, zorder=3)
            plt.scatter(tgt[i, 0] + delta, tgt[i, 1], c=custom_colour, s=5, zorder=4)
    plt.title('matches')
    match_img = convert_plt_to_tensor()

    plt.imshow(np.concatenate((radar, radar_tgt), axis=0), cmap='gray')
    delta = radar.shape[1]
    for i in range(src.shape[0]):
        if i in ids_cpu:
            custom_colour = 'g'
            plt.plot([src[i, 0], tgt[i, 0]], [src[i, 1], tgt[i, 1] + delta], c='y', linewidth=0.5, zorder=2)
            plt.scatter(src[i, 0], src[i, 1], c=custom_colour, s=5, zorder=3)
            plt.scatter(tgt[i, 0], tgt[i, 1] + delta, c=custom_colour, s=5, zorder=4)
    plt.title('matches')
    match_img2 = convert_plt_to_tensor()

    # Draw scores
    scores = out['scores'][-1]
    if scores.size(0) == 3:
        scores = scores[1] + scores[2]
    scores = scores.squeeze().detach().cpu().numpy()
    plt.imshow(scores, cmap='inferno')
    plt.colorbar()
    plt.title('log det weight (weight score vis)')
    score_img = convert_plt_to_tensor()

    # Draw detector scores
    detector_scores = out['detector_scores'][-1].squeeze().detach().cpu().numpy()
    plt.imshow(detector_scores, cmap='inferno')
    plt.colorbar()
    plt.title('detector score')
    dscore_img = convert_plt_to_tensor()

    # Draw point-to-point error
    src_p = out['src'][-1].squeeze().T
    tgt_p = out['tgt'][-1].squeeze().T
    R_tgt_src = out['R'][0, -1, :2, :2]
    t_st_in_t = out['t'][0, -1, :2, :]
    error = tgt_p - (R_tgt_src @ src_p + t_st_in_t)
    error2_sqrt = torch.sqrt(torch.sum(error * error, dim=0).squeeze())
    error2_sqrt = error2_sqrt[ids_cpu].detach().cpu().numpy()

    plt.imshow(radar, cmap='gray')
    plt.scatter(src[ids_cpu, 0], src[ids_cpu, 1], c=error2_sqrt, s=5, zorder=2, cmap='rainbow')
    plt.clim(0.0, 1)
    plt.colorbar()
    plt.title('P2P error')
    p2p_img = convert_plt_to_tensor()

    return vutils.make_grid([dscore_img, score_img, radar_img]), vutils.make_grid([match_img, match_img2]), \
        vutils.make_grid([p2p_img])


def plot_sequences(T_gt, T_pred, seq_lens, returnTensor=True, T_icra=None, savePDF=False, fnames=None, flip=True):
    """Creates a top-down plot of the predicted odometry results vs. ground truth."""
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)

    matplotlib.rcParams.update({'font.size': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
                                'axes.linewidth': 1.5, 'font.family': 'serif', 'pdf.fonttype': 42})
    T_flip = np.identity(4)
    T_flip[1, 1] = -1
    T_flip[2, 2] = -1
    imgs = []
    for seq_i, indices in enumerate(seq_indices):
        T_gt_ = np.identity(4)
        T_pred_ = np.identity(4)
        T_icra_ = np.identity(4)
        if flip:
            T_gt_ = np.matmul(T_flip, T_gt_)
            T_pred_ = np.matmul(T_flip, T_pred_)
        x_gt = []
        y_gt = []
        x_pred = []
        y_pred = []
        x_icra = []
        y_icra = []
        for i in indices:
            T_gt_ = np.matmul(T_gt[i], T_gt_)
            T_pred_ = np.matmul(T_pred[i], T_pred_)
            enforce_orthog(T_gt_)
            enforce_orthog(T_pred_)
            T_gt_temp = get_inverse_tf(T_gt_)
            T_pred_temp = get_inverse_tf(T_pred_)
            x_gt.append(T_gt_temp[0, 3])
            y_gt.append(T_gt_temp[1, 3])
            x_pred.append(T_pred_temp[0, 3])
            y_pred.append(T_pred_temp[1, 3])
            if T_icra is not None:
                T_icra_ = np.matmul(T_icra[i], T_icra_)
                enforce_orthog(T_icra_)
                T_icra_temp = get_inverse_tf(T_icra_)
                x_icra.append(T_icra_temp[0, 3])
                y_icra.append(T_icra_temp[1, 3])

        plt.figure(figsize=(10, 10), tight_layout=True)
        plt.grid(color='k', which='both', linestyle='--', alpha=0.75, dashes=(8.5, 8.5))
        plt.axes().set_aspect('equal')
        plt.plot(x_gt, y_gt, 'k', linewidth=2.5, label='GT')
        if x_icra and y_icra:
            plt.plot(x_icra, y_icra, 'r', linewidth=2.5, label='MC-RANSAC')
        plt.plot(x_pred, y_pred, 'b', linewidth=2.5, label='HERO')
        plt.xlabel('x (m)', fontsize=16)
        plt.ylabel('y (m)', fontsize=16)
        plt.legend(loc="upper left", edgecolor='k', fancybox=False)
        if savePDF and fnames is not None:
            plt.savefig(fnames[seq_i], bbox_inches='tight', pad_inches=0.0)
        if returnTensor:
            imgs.append(convert_plt_to_tensor())
        else:
            imgs.append(convert_plt_to_img())
    return imgs
