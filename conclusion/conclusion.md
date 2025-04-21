# 训练结果
## 1. 使用预训练的权重

使用提供的预训练权重 *under_the_radar_res2592.pt* ，运行 *val.py* 进行推理，下面是几个 *batch* 的输出

![pretrain_weights/under_the_radar_res2592.pt batch 5](/conclusion/hero_batch_5.png)

![pretrain_weights/under_the_radar_res2592.pt batch 33](/conclusion/hero_batch_33.png)

![pretrain_weights/under_the_radar_res2592.pt batch 37](/conclusion/hero_batch_37.png)

![pretrain_weights/under_the_radar_res2592.pt batch 97](/conclusion/hero_batch_97.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下，基本和论文中结果吻合

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  2.830 %  | 0.01324 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下，可以看到偏差比较大

![under_the_radar_odometry_estimation](/conclusion/hero_odometry_estimation.png#pic_center)

<br>
<br>
<br>

## 2. 重新训练新的权重
## T = 0.01, epoch = 3000

不使用预训练权重，运行 *train.py* 重新训练，下面是几个 *batch* 的输出

![logs/2025-04-19_04-56-07/300000.pt batch 5](/conclusion/epoch_3000_batch_5.png)

![logs/2025-04-19_04-56-07/300000.pt batch 33](/conclusion/epoch_3000_batch_33.png)

![logs/2025-04-19_04-56-07/300000.pt batch 37](/conclusion/epoch_3000_batch_37.png)

![logs/2025-04-19_04-56-07/300000.pt batch 97](/conclusion/epoch_3000_batch_97.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  0.284 %  | 0.00092 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![epoch_3000_odometry_estimation](/conclusion/epoch_3000_odometry_estimation.png)

<br>
<br>
<br>

## 3. 在预训练权重的基础上继续训练
## 3.1 T=0.01, epoch = 50

在预训练权重 *pretrain_weights/under_the_radar_res2592.pt* 的基础上，运行 *train.py* 继续训练50轮，下面是几个 *batch* 的输出

![logs/2025-04-19_10-22-33/latest.pt batch 5](/conclusion/hero+50_batch_5.png)

![logs/2025-04-19_10-22-33/latest.pt batch 33](/conclusion/hero+50_batch_33.png)

![logs/2025-04-19_10-22-33/latest.pt batch 37](/conclusion/hero+50_batch_37.png)

![logs/2025-04-19_10-22-33/latest.pt batch 97](/conclusion/hero+50_batch_97.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  1.383 %  | 0.00810 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![hero+50_odometry_estimation](/conclusion/hero+50_odometry_estimation.png)

<br>
<br>
<br>

## 3.2 T=0.01, epoch = 100

在预训练权重 *pretrain_weights/under_the_radar_res2592.pt* 的基础上，运行 *train.py* 继续训练100轮，下面是几个 *batch* 的输出

![logs/2025-04-19_13-34-24/latest.pt batch 5](/conclusion/hero+100_batch_5.png)

![logs/2025-04-19_13-34-24/latest.pt batch 33](/conclusion/hero+100_batch_33.png)

![logs/2025-04-19_13-34-24/latest.pt batch 37](/conclusion/hero+100_batch_37.png)

![logs/2025-04-19_13-34-24/latest.pt batch 97](/conclusion/hero+100_batch_97.png)



KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  1.383 %  | 0.00810 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![hero+50_odometry_estimation](/conclusion/hero+100_odometry_estimation.png)

<br>
<br>
<br>