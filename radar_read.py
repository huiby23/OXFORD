import os
from pathlib import Path
from robotcar_dataset_sdk.python.play_radar_new import play_radar_data
from robotcar_dataset_sdk.python.build_pointcloud import build_pointcloud



# 获取数据集文件路径
current_dir = Path(__file__).resolve().parent
dataset_dir = current_dir / 'dataset' / 'sample_tiny' / '2019-01-10-14-36-48-radar-oxford-10k-partial'
radar_data_dir = dataset_dir / 'radar'

if radar_data_dir.exists():
    print(f"{radar_data_dir}路径存在")
else:
    print(f"{radar_data_dir}路径不存在")


# 调用play_radar_data()函数，输出雷达数据，并分别存储极坐标系和笛卡尔坐标系的雷达数据
# display_time是每帧雷达图像输出持续时间，单位为ms，按Esc可以退出图片显示
# polar_data_list: 存储极坐标系数据的列表，每个元素是一个元组 (timestamps, azimuths, valid, fft_data, radar_resolution)
# cart_data_list: 存储笛卡尔坐标系数据的列表，每个元素是一个笛卡尔坐标图像 (np.ndarray)

polar_data, cart_data = play_radar_data(str(radar_data_dir), display_time = 1000)
# pointcloud, reflectance = build_pointcloud()

