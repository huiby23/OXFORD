# 训练集的评估结果
## 1. T=0.01, epoch = 300

训练 300 轮得到的新权重 *logs/40000.pt* ，运行 *val.py* 进行推理，下面是几个 *batch* 的输出

![logs/40000.pt batch 40](/conclusion2/train_epoch_300_batch_40.png)

![logs/40000.pt batch 80](/conclusion2/train_epoch_300_batch_80.png)

![logs/40000.pt batch 120](/conclusion2/train_epoch_300_batch_120.png)

![logs/40000.pt batch 160](/conclusion2/train_epoch_300_batch_160.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  6.453 %  |  0.0534 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![train_epoch_300_odometry_estimation](/conclusion2/train_epoch_300_odometry_estimation.png)


<br>
<br>
<br>

## 2. T=0.01, epoch = 350

训练 350 轮得到的新权重 *logs/45000.pt* ，运行 *val.py* 进行推理，下面是几个 *batch* 的输出

![logs/45000.pt batch 40](/conclusion2/train_epoch_350_batch_40.png)

![logs/45000.pt batch 80](/conclusion2/train_epoch_350_batch_80.png)

![logs/45000.pt batch 120](/conclusion2/train_epoch_350_batch_120.png)

![logs/45000.pt batch 160](/conclusion2/train_epoch_350_batch_160.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  8.511 %  |  0.0520 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![train_epoch_350_odometry_estimation](/conclusion2/train_epoch_350_odometry_estimation.png)


<br>
<br>
<br>

## 3. T=0.01, epoch = 400

训练 400 轮得到的新权重 *logs/50000.pt* ，运行 *val.py* 进行推理，下面是几个 *batch* 的输出

![logs/50000.pt batch 40](/conclusion2/train_epoch_400_batch_40.png)

![logs/50000.pt batch 80](/conclusion2/train_epoch_400_batch_80.png)

![logs/50000.pt batch 120](/conclusion2/train_epoch_400_batch_120.png)

![logs/50000.pt batch 160](/conclusion2/train_epoch_400_batch_160.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  4.678 %  |  0.0402 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![train_epoch_400_odometry_estimation](/conclusion2/train_epoch_400_odometry_estimation.png)


<br>
<br>
<br>

## 4. T=0.01, epoch = 450

训练 450 轮得到的新权重 *logs/55000.pt* ，运行 *val.py* 进行推理，下面是几个 *batch* 的输出

![logs/55000.pt batch 40](/conclusion2/train_epoch_450_batch_40.png)

![logs/55000.pt batch 80](/conclusion2/train_epoch_450_batch_80.png)

![logs/55000.pt batch 120](/conclusion2/train_epoch_450_batch_120.png)

![logs/55000.pt batch 160](/conclusion2/train_epoch_450_batch_160.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  3.825 %  |  0.0400 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![train_epoch_450_odometry_estimation](/conclusion2/train_epoch_450_odometry_estimation.png)


<br>
<br>
<br>

## 5. T=0.01, epoch = 500

训练 500 轮得到的新权重 *logs/latest.pt* ，运行 *val.py* 进行推理，下面是几个 *batch* 的输出

![logs/latest.pt batch 40](/conclusion2/train_epoch_500_batch_40.png)

![logs/latest.pt batch 80](/conclusion2/train_epoch_500_batch_80.png)

![logs/latest.pt batch 120](/conclusion2/train_epoch_500_batch_120.png)

![logs/latest.pt batch 160](/conclusion2/train_epoch_500_batch_160.png)


KITTI标准下，平移漂移误差和旋转漂移误差如下

| KITTI t_err | KITTI r_err |
| ----------- | ----------- |
|  7.682 %  |  0.0738 deg/m |


预测的运动轨迹和 *groundtruth* 轨迹的对比图如下

![train_epoch_500_odometry_estimation](/conclusion2/train_epoch_500_odometry_estimation.png)


<br>
<br>
<br>
