import os
import cv2
import torch
import shutil
import argparse
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from radar.radar_sampler import RandomWindowBatchSampler, SequentialWindowBatchSampler
from utils.utils import get_inverse_tf

class Radar_Data_Preprocess():
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
  

    def radar_img_loader(self, display_time = 10, cart_width = 448):
        """
        Function:
            - 加载radar扫描数据，并显示输出
            - 按Esc可以退出图片输出，但是提前退出会导致只有部分图片被读取

        Args:
            - data_dir: radar扫描数据的文件路径
            - display_time: 每一帧radar扫描图像的显示持续时间(ms)
        
        Returns:
            - polar_data_list: 存储极坐标系数据的列表，每个元素是一个元组 (timestamps, azimuths, valid, fft_data, radar_resolution)
            - cart_data_list: 存储笛卡尔坐标系数据的列表，每个元素是一个笛卡尔坐标图像 (np.ndarray)
            - radar_timestamps: radar扫描数据的source timestamp序列       
        """
        radar_data_dir = os.path.join(str(self.dataset_dir), 'radar')

        # 判断radar扫描数据文件路径是否存在
        if not os.path.exists(radar_data_dir):
            raise IOError(f'{radar_data_dir}路径不存在，请检查radar扫描数据路径!')
        
        # 初始化
        polar_data_list = []  # 存储极坐标系数据
        cart_data_list = []   # 存储笛卡尔坐标系数据

        # 读取radar timestamps
        timestamps_path = os.path.join(os.path.join(radar_data_dir, os.pardir, 'radar.timestamps'))
        if not os.path.isfile(timestamps_path):
            raise IOError(f'{timestamps_path}路径不存在，请检查radar timestamps数据路径!')

        # 输出设置
        cart_resolution = .25
        cart_pixel_width = cart_width  # pixels
        interpolate_crossover = True

        title = "Radar Scan Img"

        # 按照timestamp从小到大，遍历读取radar扫描数据
        radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
        for radar_timestamp in radar_timestamps:
            filename = os.path.join(radar_data_dir, str(radar_timestamp) + '.png')
            
            if not os.path.isfile(filename):
                raise FileNotFoundError("Could not find radar example: {}".format(filename))

            # 加载雷达数据
            timestamps, azimuths, valid, fft_data, radar_resolution = self.load_radar(filename)

            # 将极坐标数据转换为笛卡尔坐标
            cart_img = self.radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                                                    interpolate_crossover)

            # 存储极坐标系数据
            polar_data_list.append((timestamps, azimuths, valid, fft_data, radar_resolution))

            # 存储笛卡尔坐标系数据
            cart_data_list.append(cart_img)

            # 可视化处理
            downsample_rate = 4
            fft_data_vis = fft_data[:, ::downsample_rate]
            resize_factor = float(cart_img.shape[0]) / float(fft_data_vis.shape[0])
            fft_data_vis = cv2.resize(fft_data_vis, (0, 0), None, resize_factor, resize_factor)
            vis = cv2.hconcat((fft_data_vis, fft_data_vis[:, :10] * 0 + 1, cart_img))

            # 调整窗口尺寸
            w = int(vis.shape[1] * 0.8)
            h = int(vis.shape[0] * 0.8)
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, w, h)
            
            cv2.putText(vis, f'Source Timestamp: {radar_timestamp}', (40, 40), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 0), 2)
            cv2.imshow(title, vis * 2.)  # The data is doubled to improve visualisation

            key = cv2.waitKey(display_time)  # 控制显示时间
            if key == 27:  # 按下 ESC 键退出
                break

        cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
        return polar_data_list, cart_data_list, radar_timestamps


    def load_radar(self, radar_data_path):
        """
        Function:
            - Decode a single Oxford Radar RobotCar Dataset radar example
        
        Args:
            - example_path (AnyStr): Oxford Radar RobotCar Dataset Example png
        
        Returns:
            - timestamps (np.ndarray): Timestamp for each azimuth in int64 (UNIX time)
            - azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
            - valid (np.ndarray) Mask of whether azimuth data is an original sensor reading or interpolated from adjacent azimuths
            - fft_data (np.ndarray): Radar power readings along each azimuth
            - radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        """
        # Hard coded configuration to simplify parsing code
        radar_resolution = np.array([0.0432], np.float32)
        encoder_size = 5600

        raw_example_data = cv2.imread(str(radar_data_path), cv2.IMREAD_GRAYSCALE)
        timestamps = raw_example_data[:, :8].copy().view(np.int64)
        azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
        valid = raw_example_data[:, 10:11] == 255
        fft_data = raw_example_data[:, 11:].astype(np.float32)[:, :, np.newaxis] / 255.

        #  return Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
        return timestamps, azimuths, valid, fft_data, radar_resolution


    def radar_polar_to_cartesian(self, azimuths: np.ndarray, fft_data: np.ndarray, radar_resolution: float,
                                cart_resolution: float, cart_pixel_width: int, interpolate_crossover=True):
        """Convert a polar radar scan to cartesian.
        Args:
            azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
            fft_data (np.ndarray): Polar radar power readings
            radar_resolution (float): Resolution of the polar radar data (metres per pixel)
            cart_resolution (float): Cartesian resolution (metres per pixel)
            cart_pixel_size (int): Width and height of the returned square cartesian output (pixels). Please see the Notes
                below for a full explanation of how this is used.
            interpolate_crossover (bool, optional): If true interpolates between the end and start  azimuth of the scan. In
                practice a scan before / after should be used but this prevents nan regions in the return cartesian form.

        Returns:
            np.ndarray: Cartesian radar power readings
        
        Notes:
            After using the warping grid the output radar cartesian is defined as as follows where
            X and Y are the `real` world locations of the pixels in metres:
            If 'cart_pixel_width' is odd:
                            +------ Y = -1 * cart_resolution (m)
                            |+----- Y =  0 (m) at centre pixel
                            ||+---- Y =  1 * cart_resolution (m)
                            |||+--- Y =  2 * cart_resolution (m)
                            |||| +- Y =  cart_pixel_width // 2 * cart_resolution (m) (at last pixel)
                            |||| +-----------+
                            vvvv             v
            +---------------+---------------+
            |               |               |
            |               |               |
            |               |               |
            |               |               |
            |               |               |
            |               |               |
            |               |               |
            +---------------+---------------+ <-- X = 0 (m) at centre pixel
            |               |               |
            |               |               |
            |               |               |
            |               |               |
            |               |               |
            |               |               |
            |               |               |
            +---------------+---------------+
            <------------------------------->
                cart_pixel_width (pixels)
            If 'cart_pixel_width' is even:
                            +------ Y = -0.5 * cart_resolution (m)
                            |+----- Y =  0.5 * cart_resolution (m)
                            ||+---- Y =  1.5 * cart_resolution (m)
                            |||+--- Y =  2.5 * cart_resolution (m)
                            |||| +- Y =  (cart_pixel_width / 2 - 0.5) * cart_resolution (m) (at last pixel)
                            |||| +----------+
                            vvvv            v
            +------------------------------+
            |                              |
            |                              |
            |                              |
            |                              |
            |                              |
            |                              |
            |                              |
            |                              |
            |                              |
            |                              |
            |                              |
            |                              |
            |                              |
            |                              |
            |                              |
            +------------------------------+
            <------------------------------>
                cart_pixel_width (pixels)
        """
        if (cart_pixel_width % 2) == 0:
            cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
        else:
            cart_min_range = cart_pixel_width // 2 * cart_resolution
        coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
        Y, X = np.meshgrid(coords, -coords)
        sample_range = np.sqrt(Y * Y + X * X)
        sample_angle = np.arctan2(Y, X)
        sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

        # Interpolate Radar Data Coordinates
        azimuth_step = azimuths[1] - azimuths[0]
        sample_u = (sample_range - radar_resolution / 2) / radar_resolution
        sample_v = (sample_angle - azimuths[0]) / azimuth_step

        # We clip the sample points to the minimum sensor reading range so that we
        # do not have undefined results in the centre of the image. In practice
        # this region is simply undefined.
        sample_u[sample_u < 0] = 0

        if interpolate_crossover:
            fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
            sample_v = sample_v + 1

        polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
        cart_img = np.expand_dims(cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
        
        # return np.ndarray
        return cart_img


    def road_odometry_loader(self):
        """
        Function:
            - 加载gt数据，计算SE(2)位姿变换矩阵
        
        Args:
            - None

        Returns:
            - pose_tran: SE(2)位姿变换矩阵序列
            - total_radar_timestamps: radar扫描数据的source timestamp序列
        """
        gt_dir = os.path.join(str(self.dataset_dir), 'gt', 'radar_odometry.csv')
        required_columns = ['source_timestamp', 'destination_timestamp', 'x', 'y', 'yaw', 
                            'source_radar_timestamp', 'destination_radar_timestamp']
        
        # 读取数据
        data = pd.read_csv(gt_dir)

        # 验证数据完整性
        assert all(col in data.columns for col in required_columns), f"数据文件缺少必要列: {required_columns}"


        # 初始化轨迹参数
        pose_tran = []
        global_pose = np.eye(3)
        trajectory = [global_pose[:2, 2]]  # 初始位置

        # 执行位姿积分
        for _, row in data.iterrows():
            rel_pose = self.se2_transform(row['x'], row['y'], row['yaw'])
            pose_tran.append(rel_pose)

            global_pose = global_pose @ rel_pose
            trajectory.append(global_pose[:2, 2])

        trajectory = np.array(trajectory)

        # 生成所有radar扫描时间戳序列，以及中心时刻序列
        total_radar_timestamps = data['source_radar_timestamp'].to_numpy()

        return pose_tran, total_radar_timestamps


    def se2_transform(self, x, y, yaw):
        """
        Function:
            - 生成SE(2)变换矩阵
        
        Args:
            - x: x坐标相对变化
            - y: y坐标相对变化
            - yaw: 偏航角相对变化
        
        Returns:
            - 3X3的SE(2)变换矩阵
        """

        return np.array([
            [np.cos(yaw), -np.sin(yaw), x],
            [np.sin(yaw),  np.cos(yaw), y],
            [0,           0,            1]
        ])
    

    def se3_transform(self, x, y, yaw):
        """Returns a 4x4 homogeneous 3D transform for given 2D parameters (x, y, theta).
        Note: (x,y) are position of frame 2 wrt frame 1 as measured in frame 1.
        Args:
            x (float): x translation
            x (float): y translation
            yaw (float): rotation
        Returns:
            np.ndarray: 4x4 transformation matrix from next time to current (T_1_2)
        """

        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0, x],
            [np.sin(yaw),  np.cos(yaw), 0, y],
            [0,           0,            0, 0],
            [0,           0,            0, 1]
        ])


    def delete_files_in_folder(self, folder_path):
        """
        Function:
            - 删除目标文件夹下的所有文件和子文件夹，但是保留目标文件夹
        
        Args:
            - folder_path: 目标文件夹
        """
        # 遍历文件夹中的所有文件和子文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 如果是文件或符号链接，则删除
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                # 如果是子文件夹，则递归删除其内容
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子文件夹及其内容
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


    def get_sequences(path, prefix='2019'):
        """Retrieves a list of all the sequences in the dataset with the given prefix.
            Sequences are subfolders underneath 'path'
        Args:
            path (AnyStr): path to the root data folder
            prefix (AnyStr): each sequence / subfolder must begin with this common string.
        Returns:
            List[AnyStr]: List of sequences / subfolder names.
        """
        sequences = [f for f in os.listdir(path) if prefix in f]
        sequences.sort()
        return sequences


    def get_frames(path, extension='.png'):
        """Retrieves all the file names within a path that match the given extension.
        Args:
            path (AnyStr): path to the root/sequence/sensor/ folder
            extension (AnyStr): each data frame must end with this common string.
        Returns:
            List[AnyStr]: List of frames / file names.
        """
        frames = [f for f in os.listdir(path) if extension in f]
        frames.sort()
        return frames
    

    def polar_threshold_mask(polar_data, multiplier=3.0):
        """Thresholds on multiplier*np.mean(azimuth_data) to create a polar mask of likely target points.
        Args:
            polar_data (np.ndarray): num_azimuths x num_range_bins polar data
            multiplier (float): multiple of mean that we treshold on
        Returns:
            np.ndarray: binary polar mask corresponding to likely target points
        """
        # 计算每个方位角的均值（保持维度以便广播）
        mean_per_azimuth = np.mean(polar_data, axis=1, keepdims=True)
        
        # 在每个方向角上，忽略无效数据
        # 阈值为每个方位角的均值乘以multiplier
        mask = (polar_data > multiplier * mean_per_azimuth).astype(np.float32)
        
        return mask
    

    def main(self, train_data_dir, radar_display_time = 10, cart_width = 448):
        """
        Function:
            - 主函数，可以从dataset提取radar扫描图像和gt数据，生成训练数据
        
        Args:
            - train_data_dir: 保存训练数据的文件路径
            - radar_display_time: 每一帧radar扫描图像的输出持续时间(ms)
        
        Notes:
            - 输出radar扫描图像时，按Esc可以退出图片输出，执行剩余代码
            - 但是提前退出会导致只有部分图片被读取，建议减小radar_display_time以代替
        """
        # polar_data: 存储极坐标系数据的列表，每个元素是一个元组 (timestamps, azimuths, valid, fft_data, radar_resolution)
        # cart_data: 存储笛卡尔坐标系数据的列表，每个元素是一个笛卡尔坐标的radar图像 (np.ndarray)
        # radar_timestamps：雷达扫描的source timestamp序列
        _, cart_data, radar_timestamps = self.radar_img_loader(radar_display_time, cart_width)

        # radar_pose_tran: 相邻两次雷达扫描的位姿变换矩阵
        # radar_timestamps：雷达扫描的source时间戳序列
        radar_pose_tran, total_radar_timestamps = self.road_odometry_loader()

        # 打印相关信息
        print('\nValid Frames Num:', len(radar_timestamps))
        print('Total Frames Num:', len(total_radar_timestamps))
        print('\nStart Timestamp:',radar_timestamps[0])
        print('End Timestamp:', radar_timestamps[-1])

        # 对齐radar数据和位姿变换数据的时间戳，提取出有radar扫描数据的时刻
        lower_bd = radar_timestamps[0]
        upper_bd = radar_timestamps[-1]

        indices = np.argwhere((total_radar_timestamps >= lower_bd) & (total_radar_timestamps <= upper_bd))

        print("\nIndices Num:", len(indices))
        print("Valid Indices:", indices[0], '~', indices[-1])


        # 对于有radar扫描数据的时刻，保存相邻两张radar图像数据，以及位姿变换矩阵
        # 每一组数据包含img1.png, img2.png, pose_tran.npy
        # 每一组数据保存在'./train_data/XXXX_YYYY'文件夹下，XXXX为img1.png的时间戳，YYYY为img2.png的时间戳
        # 其中XXXX比YYYY小，也即pose_tran对应img1到img2的位姿变换

        # 判断radar扫描数据文件路径是否存在
        if not os.path.exists(train_data_dir):
            raise IOError(f'{train_data_dir}路径不存在，请检查train data保存路径!')
        
        # 清空train_data文件夹
        self.delete_files_in_folder(train_data_dir)
        print(f"\n{train_data_dir} 文件夹下的所有内容已经成功清空！")


        # 按照要求保存训练数据
        for k in range(indices.shape[0] - 1):
            tran_index = indices[k][0]

            # 相邻radar扫描图像，位姿变换矩阵
            img_1 = (cart_data[k] * 255).astype(np.uint8)
            img_2 = (cart_data[k + 1] * 255).astype(np.uint8)
            pose_tran = radar_pose_tran[tran_index]

            # 相邻时间戳
            tt_1 = radar_timestamps[k]
            tt_2 = radar_timestamps[k + 1]

            # 保存路径
            folder_name = f'{tt_1}_{tt_2}'
            data_dir = train_data_dir / folder_name

            # 判断保存路径是否存在
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            cv2.imwrite(str(os.path.join(data_dir, 'image_1.png')), img_1)
            cv2.imwrite(str(os.path.join(data_dir, 'image_2.png')), img_2)
            np.save(os.path.join(str(data_dir), 'pose_tran.npy'), pose_tran)

        data_num = indices.shape[0] - 1
        print(f'{data_num}组训练数据已经成功保存，保存路径是{train_data_dir}\n')



class OxfordDataset(Dataset):
    """Oxford Radar Robotcar Dataset."""
    def __init__(self, config, split='train'):
        self.config = config
        self.data_dir = config['data_dir']
        self.processor = Radar_Data_Preprocess(self.data_dir)
        self.dataset_prefix = config['dataset_prefix']
        self.polar_mask = config['polar_mask']
        
        sequences = self.processor.get_sequences(self.data_dir, self.dataset_prefix)
        self.sequences = self.get_sequences_split(sequences, split)
        self.seq_idx_range = {}
        self.frames = []
        self.seq_lens = []
        for seq in self.sequences:
            seq_frames = self.processor.get_frames(os.path.join(self.data_dir, seq, 'radar'))
            seq_frames = self.get_frames_with_gt(seq_frames, os.path.join(self.data_dir, seq, 'gt', 'radar_odometry.csv'))
            self.seq_idx_range[seq] = [len(self.frames), len(self.frames) + len(seq_frames)]
            self.seq_lens.append(len(seq_frames))
            self.frames.extend(seq_frames)


    def get_sequences_split(self, sequences, split):
        """Retrieves a list of sequence names depending on train/validation/test split.
        Args:
            sequences (List[AnyStr]): list of all the sequences, sorted lexicographically
            split (List[int]): indices of a specific split (train or val or test) aftering sorting sequences
        Returns:
            List[AnyStr]: list of sequences that belong to the specified split
        """
        self.split = self.config['train_split']
        if split == 'validation':
            self.split = self.config['validation_split']
        elif split == 'test':
            self.split = self.config['test_split']
        return [seq for i, seq in enumerate(sequences) if i in self.split]


    def get_frames_with_gt(self, frames, gt_path):
        """Retrieves the subset of frames that have groundtruth
        Note: For the Oxford Dataset we do a search from the end backwards because some
            of the sequences don't have GT as the end, but they all have GT at the beginning.
        Args:
            frames (List[AnyStr]): List of file names
            gt_path (AnyStr): path to the ground truth csv file
        Returns:
            List[AnyStr]: List of file names with ground truth
        """
        def check_if_frame_has_gt(frame, gt_lines):
            for i in range(len(gt_lines) - 1, -1, -1):
                line = gt_lines[i].split(',')
                if frame == int(line[9]):
                    return True
            return False
        frames_out = frames
        with open(gt_path, 'r') as f:
            f.readline()
            lines = f.readlines()
            for i in range(len(frames) - 1, -1, -1):
                frame = int(frames[i].split('.')[0])
                if check_if_frame_has_gt(frame, lines):
                    break
                frames_out.pop()
        return frames_out


    def get_groundtruth_odometry(self, radar_time, gt_path):
        """Retrieves the groundtruth 4x4 transform from current time to next
        Args:
            radar_time (int): UNIX INT64 timestamp that we want groundtruth for (also the filename for radar)
            gt_path (AnyStr): path to the ground truth csv file
        Returns:
            T_2_1 (np.ndarray): 4x4 transformation matrix from current time to next
            time1 (int): UNIX INT64 timestamp of the current frame
            time2 (int): UNIX INT64 timestamp of the next frame
        """
        with open(gt_path, 'r') as f:
            f.readline()
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.split(',')
                if int(line[9]) == radar_time:
                    T = self.processor.se3_transform(float(line[2]), float(line[3]), float(line[7]))  # from next time to current
                    return get_inverse_tf(T), int(line[1]), int(line[0])    # T_2_1 from current time step to the next
        assert(0), 'ground truth transform for {} not found in {}'.format(radar_time, gt_path)

    def __len__(self):
        return len(self.frames)

    def get_seq_from_idx(self, idx):
        """Returns the name of the sequence that this idx belongs to.
        Args:
            idx (int): frame index in dataset
        Returns:
            AnyStr: name of the sequence that this idx belongs to
        """
        for seq in self.sequences:
            if self.seq_idx_range[seq][0] <= idx and idx < self.seq_idx_range[seq][1]:
                return seq
        assert(0), 'sequence for idx {} not found in {}'.format(idx, self.seq_idx_range)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = self.get_seq_from_idx(idx)
        frame = os.path.join(self.data_dir, seq, 'radar', self.frames[idx])
        timestamps, azimuths, _, polar, radar_resolution = self.processor.load_radar(frame)
        
        # Convert to cartesian
        data = self.processor.radar_polar_to_cartesian(azimuths, polar, radar_resolution,
                                        self.config['cart_resolution'], self.config['cart_pixel_width'])
        
        if self.polar_mask:
            polar_masked = self.processor.polar_threshold_mask(polar)
        else:
            polar_masked = polar
        mask = self.processor.radar_polar_to_cartesian(azimuths, polar_masked, radar_resolution,
                                        self.config['cart_resolution'], self.config['cart_pixel_width'])
        
        # Get ground truth transform between this frame and the next
        radar_time = int(self.frames[idx].split('.')[0])

        T_21, time1, time2 = self.get_groundtruth_odometry(radar_time, os.path.join(self.data_dir, seq, 'gt', 'radar_odometry.csv'))
        
        t_ref = np.array([time1, time2]).reshape(1, 2)
        polar = np.expand_dims(polar, axis=0)
        azimuths = np.expand_dims(azimuths, axis=0)
        timestamps = np.expand_dims(timestamps, axis=0)
        
        return {'data': data, 'T_21': T_21, 't_ref': t_ref, 'mask': mask, 'polar': polar, 'azimuths': azimuths,
                'timestamps': timestamps}


def get_dataloaders(config):
    """Returns the dataloaders for training models in pytorch.
    Args:
        config (json): parsed configuration file
    Returns:
        train_loader (DataLoader)
        valid_loader (DataLoader)
        test_loader (DataLoader)
    """
    vconfig = dict(config)
    vconfig['batch_size'] = 1
    train_dataset = OxfordDataset(config, 'train')
    valid_dataset = OxfordDataset(vconfig, 'validation')
    test_dataset = OxfordDataset(vconfig, 'test')
    train_sampler = RandomWindowBatchSampler(config['batch_size'], config['window_size'], train_dataset.seq_lens)
    valid_sampler = SequentialWindowBatchSampler(1, config['window_size'], valid_dataset.seq_lens)
    test_sampler = SequentialWindowBatchSampler(1, config['window_size'], test_dataset.seq_lens)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=config['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=config['num_workers'])
    return train_loader, valid_loader, test_loader



class CustomDataset(Dataset):
    def __init__(self, data_root_dir,img_sz, mode='train'):
        self.mode=mode
        self.root_dir = data_root_dir
        self.data_folders = [os.path.join(data_root_dir, folder) for folder in os.listdir(data_root_dir) if os.path.isdir(os.path.join(data_root_dir, folder))]
        self.train_data=self.data_folders[0:int(len(self.data_folders)*0.8)]
        self.test_data=self.data_folders[int(len(self.data_folders)*0.8):]
        # self.test_data=self.data_folders[0:int(len(self.data_folders)*0.8)]
        self.transform = transforms.Compose(
                                                [
                                                    transforms.Resize((img_sz, img_sz)),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize(mean=[0.5], std=[0.5])
                                                ]
                                             )
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.mode == 'train':
            folder_path = self.train_data[idx]
        else:
            folder_path = self.test_data[idx]

        img_name = folder_path.split('/')[-1]

        # 导入radar扫描图像
        image_1_path = os.path.join(folder_path, 'image_1.png')
        image_2_path = os.path.join(folder_path, 'image_2.png')
        image_1 = Image.open(image_1_path)
        image_2 = Image.open(image_2_path)
               
        # 导入位姿变换矩阵
        pose_tran_path = os.path.join(folder_path, 'pose_tran.npy')
        pose_tran = np.load(pose_tran_path)
        
        # 改变数据类型为tensor
        image_1 = self.transform(image_1)
        image_2 = self.transform(image_2)
        pose_tran = torch.from_numpy(pose_tran).float()
        
        if self.mode == 'train':
            return image_1, image_2, pose_tran
        else:
            return image_1, image_2, pose_tran ,img_name



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Load train data from the dataset.')
#     parser.add_argument('--dataset_dir', type=Path, help='Directory containing dataset.')
#     parser.add_argument('--train_data_dir', type=Path, help='Directory for savining train data.')
#     parser.add_argument('--radar_display_time', type=int, help='Display time of each radar frame.')
#     parser.add_argument('--cart_width', type=int, help='Width of the cartesian radar image.')
#     args = parser.parse_args()

#     dataset_path = args.dataset_dir
#     train_data_path = args.train_data_dir
#     radar_display_time = args.radar_display_time
#     cart_width = args.cart_width

#     print(dataset_path, train_data_path, radar_display_time)

#     data_preprocess = Radar_Data_Preprocess(dataset_dir = dataset_path)
#     data_preprocess.main(train_data_path, radar_display_time, cart_width)
