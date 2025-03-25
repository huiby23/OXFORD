import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# gt数据文件路径
current_dir = Path(__file__).resolve().parent
dataset_dir = current_dir / 'data' / 'sample_tiny' / '2019-01-10-14-36-48-radar-oxford-10k-partial'
gt_dir = dataset_dir / 'gt' / 'radar_odometry.csv'


# 数据预处理（含异常过滤）
required_columns = ['source_timestamp', 'destination_timestamp', 'x', 'y', 'yaw']
data = pd.read_csv(gt_dir)

# 验证数据完整性
assert all(col in data.columns for col in required_columns), f"数据文件缺少必要列: {required_columns}"

# 时间戳数据预处理
data['source_ts'] = data['source_timestamp'] / 1e6
data['dest_ts'] = data['destination_timestamp'] / 1e6
data['duration'] = np.abs(data['dest_ts'] - data['source_ts'])


# 计算中点时间
data['mid_ts'] = (data['source_ts'] + data['dest_ts']) / 2


# SE(2)轨迹积分
def se2_transform(x, y, theta):
    """构建SE(2)变换矩阵（含旋转平移）"""
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta),  np.cos(theta), y],
        [0,             0,              1]
    ])

# 初始化轨迹参数
global_pose = np.eye(3)
trajectory = [global_pose[:2, 2]]  # 初始位置

# 执行位姿积分
for _, row in data.iterrows():
    rel_pose = se2_transform(row['x'], row['y'], row['yaw'])
    global_pose = global_pose @ rel_pose
    trajectory.append(global_pose[:2, 2])

trajectory = np.array(trajectory)


# 瞬时线速度计算
data['displacement'] = np.sqrt(data['x']**2 + data['y']**2)
data['velocity'] = data['displacement'] / data['duration']

# 加速度计算（中央差分法）
def robust_central_diff(y, t):
    derivative = np.zeros_like(y)
    valid_idx = np.where(~np.isnan(y))[0]
    
    for i in valid_idx:
        if i == 0:
            derivative[i] = (y[i+1] - y[i]) / (t[i+1] - t[i])
        elif i == len(y)-1:
            derivative[i] = (y[i] - y[i-1]) / (t[i] - t[i-1])
        else:
            derivative[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])
    return derivative

data['acceleration'] = robust_central_diff(data['velocity'].values, data['mid_ts'].values)


# gt数据可视化
plt.figure(figsize=(12, 9))

# 轨迹可视化
ax1 = plt.subplot(211)
ax1.plot(trajectory[:,0], trajectory[:,1], 'b-', lw=1.5)
ax1.grid(True)
ax1.set_xlabel('X (m)', fontsize=10)
ax1.set_ylabel('Y (m)', fontsize=10)
ax1.set_title('Radar Odometry Trajectory with Orientation', fontsize=12)

# 运动参数可视化
ax2 = plt.subplot(223)
ax2.plot(data['mid_ts'], data['velocity'], 'g-', lw=1.2)
ax2.set_ylabel('Velocity (m/s)', fontsize=8)
ax2.set_ylim(0, 20)

ax3 = plt.subplot(224)
ax3.plot(data['mid_ts'], data['acceleration'], 'r-', lw=1.2)
ax3.set_ylabel('Acceleration (m/s²)', fontsize=8)
ax3.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('results/radar_odometry_analysis.png', dpi=300)
plt.show()