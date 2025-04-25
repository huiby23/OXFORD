# 1. T = 0.01， epoch = 2400

 *batch = 50, 100, 150, 200* 的训练输出如下, *Valid Points* 筛选阈值为0.95

![logs/2025-04-19_04-56-07/300000.pt batch 50](/conclusion3/epoch_2400_batch_50.png)

![logs/2025-04-19_04-56-07/300000.pt batch 100](/conclusion3/epoch_2400_batch_100.png)

![logs/2025-04-19_04-56-07/300000.pt batch 150](/conclusion3/epoch_2400_batch_150.png)

![logs/2025-04-19_04-56-07/300000.pt batch 200](/conclusion3/epoch_2400_batch_200.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  0.317 %  | 0.00128 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![epoch_2400_odometry_estimation](/conclusion3/epoch_2400_odometry_estimation.png)

<br>
<br>
<br>
<br>
<br>


# 2. T = 0.01， epoch = 1600

 *batch = 50, 100, 150, 200* 的训练输出如下, *Valid Points* 筛选阈值为0.8

![logs/2025-04-19_04-56-07/200000.pt batch 50](/conclusion3/epoch_1600_batch_50.png)

![logs/2025-04-19_04-56-07/200000.pt batch 100](/conclusion3/epoch_1600_batch_100.png)

![logs/2025-04-19_04-56-07/200000.pt batch 150](/conclusion3/epoch_1600_batch_150.png)

![logs/2025-04-19_04-56-07/200000.pt batch 200](/conclusion3/epoch_1600_batch_200.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  0.603 %  | 0.00389 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![epoch_1600_odometry_estimation](/conclusion3/epoch_1600_odometry_estimation.png)

<br>
<br>
<br>
<br>
<br>


# 3. T = 0.01， epoch = 800

 *batch = 50, 100, 150, 200* 的训练输出如下, *Valid Points* 筛选阈值为0.8

![logs/2025-04-19_04-56-07/100000.pt batch 50](/conclusion3/epoch_800_batch_50.png)

![logs/2025-04-19_04-56-07/100000.pt batch 100](/conclusion3/epoch_800_batch_100.png)

![logs/2025-04-19_04-56-07/100000.pt batch 150](/conclusion3/epoch_800_batch_150.png)

![logs/2025-04-19_04-56-07/100000.pt batch 200](/conclusion3/epoch_800_batch_200.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  2.278 %  | 0.01431 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![epoch_800_odometry_estimation](/conclusion3/epoch_800_odometry_estimation.png)