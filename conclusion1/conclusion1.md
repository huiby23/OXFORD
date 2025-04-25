# 4.20-4.21 训练结果
## 1. T=0.01, epoch = 300

训练 300 轮得到的新权重 *logs/40000.pt* ，运行 *val.py* 进行推理，下面是几个 *batch* 的输出

![logs/40000.pt batch 10](/conclusion1/epoch_300_batch_10.png)

![logs/40000.pt batch 20](/conclusion1/epoch_300_batch_20.png)

![logs/40000.pt batch 30](/conclusion1/epoch_300_batch_30.png)

![logs/40000.pt batch 40](/conclusion1/epoch_300_batch_40.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  19.49 %  |  0.7439 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![epoch_300_odometry_estimation](/conclusion1/epoch_300_odometry_estimation.png)


<br>
<br>
<br>

## 2. T=0.01, epoch = 350

训练 350 轮得到的新权重 *logs/45000.pt* ，运行 *val.py* 进行推理，下面是几个 *batch* 的输出

![logs/45000.pt batch 10](/conclusion1/epoch_350_batch_10.png)

![logs/45000.pt batch 20](/conclusion1/epoch_350_batch_20.png)

![logs/45000.pt batch 30](/conclusion1/epoch_350_batch_30.png)

![logs/45000.pt batch 40](/conclusion1/epoch_350_batch_40.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  20.67 %  |  0.3692 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![epoch_350_odometry_estimation](/conclusion1/epoch_350_odometry_estimation.png)


<br>
<br>
<br>

## 3. T=0.01, epoch = 400

训练 400 轮得到的新权重 *logs/50000.pt* ，运行 *val.py* 进行推理，下面是几个 *batch* 的输出

![logs/50000.pt batch 10](/conclusion1/epoch_400_batch_10.png)

![logs/50000.pt batch 20](/conclusion1/epoch_400_batch_20.png)

![logs/50000.pt batch 30](/conclusion1/epoch_400_batch_30.png)

![logs/50000.pt batch 40](/conclusion1/epoch_400_batch_40.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  18.34 %  |  0.7075 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![epoch_400_odometry_estimation](/conclusion1/epoch_400_odometry_estimation.png)


<br>
<br>
<br>

## 4. T=0.01, epoch = 450

训练 450 轮得到的新权重 *logs/55000.pt* ，运行 *val.py* 进行推理，下面是几个 *batch* 的输出

![logs/55000.pt batch 10](/conclusion1/epoch_450_batch_10.png)

![logs/55000.pt batch 20](/conclusion1/epoch_450_batch_20.png)

![logs/55000.pt batch 30](/conclusion1/epoch_450_batch_30.png)

![logs/55000.pt batch 40](/conclusion1/epoch_450_batch_40.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  56.70 %  |  0.7101 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![epoch_450_odometry_estimation](/conclusion1/epoch_450_odometry_estimation.png)


<br>
<br>
<br>

## 5. T=0.01, epoch = 500

训练 500 轮得到的新权重 *logs/latest.pt* ，运行 *val.py* 进行推理，下面是几个 *batch* 的输出

![logs/latest.pt batch 10](/conclusion1/epoch_500_batch_10.png)

![logs/latest.pt batch 20](/conclusion1/epoch_500_batch_20.png)

![logs/latest.pt batch 30](/conclusion1/epoch_500_batch_30.png)

![logs/latest.pt batch 40](/conclusion1/epoch_500_batch_40.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  26.94 %  |  0.7543 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![epoch_500_odometry_estimation](/conclusion1/epoch_500_odometry_estimation.png)


<br>
<br>
<br>
