################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Dan Barnes (dbarnes@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import argparse
import os
from .radar import load_radar, radar_polar_to_cartesian
import numpy as np
import cv2

import os
import numpy as np
import cv2
from typing import List, Tuple
from .radar import load_radar, radar_polar_to_cartesian

def play_radar_data(data_dir, display_time=1) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]],
                                                        List[np.ndarray]]:
    """
    读取并输出雷达数据的主函数。
    :param data_dir: 包含雷达数据的目录路径
    :param display_time: 每帧图像的显示时间（单位为毫秒）。如果为 0，则等待用户按键。
    :return: 返回两个列表：
             - polar_data_list: 存储极坐标系数据的列表，每个元素是一个元组 (timestamps, azimuths, valid, fft_data, radar_resolution)
             - cart_data_list: 存储笛卡尔坐标系数据的列表，每个元素是一个笛卡尔坐标图像 (np.ndarray)。
    """
    polar_data_list = []  # 存储极坐标系数据
    cart_data_list = []   # 存储笛卡尔坐标系数据

    timestamps_path = os.path.join(os.path.join(data_dir, os.pardir, 'radar.timestamps'))
    if not os.path.isfile(timestamps_path):
        raise IOError("Could not find timestamps file")

    # Cartesian Visualsation Setup
    cart_resolution = .25
    cart_pixel_width = 501  # pixels
    interpolate_crossover = True

    title = "Radar Visualisation Example"

    radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
    for radar_timestamp in radar_timestamps:
        filename = os.path.join(data_dir, str(radar_timestamp) + '.png')

        if not os.path.isfile(filename):
            raise FileNotFoundError("Could not find radar example: {}".format(filename))

        # 加载雷达数据
        timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(filename)

        # 将极坐标数据转换为笛卡尔坐标
        cart_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
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

        cv2.imshow(title, vis * 2.)  # The data is doubled to improve visualisation
        key = cv2.waitKey(display_time)  # 控制显示时间

        if key == 27:  # 按下 ESC 键退出
            break

    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

    return polar_data_list, cart_data_list

def main():
    """
    命令行入口函数。
    """
    parser = argparse.ArgumentParser(description='Play back radar data from a given directory')
    parser.add_argument('dir', type=str, help='Directory containing radar data.')
    args = parser.parse_args()
    play_radar_data(args.dir)

if __name__ == "__main__":
    main()