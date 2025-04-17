`config/radar.json`：包含模型训练中的可调参数


`networks/Oxford_Radar.py`：模型代码

`networks/UNet.py`：`UNet`网络结构代码


`radar/radar_sampler.py`：随机采样雷达数据，用于训练`train`；顺序采样雷达数据，用于推理评估`val`

`radar/radar_transform.py`：随机旋转雷达数据，增强训练模型的鲁棒性


`utils/dataloader.py`：

`Radar_Data_Preprocess`：从数据集中读取数据，并进行预处理

`OxfordDataset`：模型的数据集`Dateset`

`get_dataloaders`：传递`dataloader`给模型


`utils/loss.py`：

`supervised_loss`：计算损失`loss`

`unsupervised_loss`：计算损失`loss`

`Keypoint`：算法第一步，关键点初始化`Keypoints initialization`

`SoftmaxMatcher`：算法第二步，关键点匹配`Differentiable point matching`

`SoftmaxRefMatcher`：目前代码未使用

`SVD`：算法第三步，位姿变换矩阵估计`Differentiable pose estimation`


`utils/monitor.py`：功能是监测`train_loss`等参数的变化，并将数据传递给`Tensorboard`

`utils/utils.py`：包含各种辅助函数，供其它模块调用