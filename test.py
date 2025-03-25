import torch

# 假设这些是输入张量
B = 12  # batch size
num_keypoints = 400  # 关键点数
feature_dim = 248  # 特征维度
H, W = 448, 448  # 图像尺寸

# ds shape: (B, num_keypoints, feature_dim)
ds = torch.randn(B, num_keypoints, feature_dim)

# descriptors_map2 shape: (B, feature_dim, H, W)
descriptors_map2 = torch.randn(B, feature_dim, H, W)

# 使用 einsum 进行点积计算
ci = torch.einsum('bik,bjkhw->bijhw', ds, descriptors_map2)

# ci 现在的形状是 (B, num_keypoints, H, W)
print(ci.shape)  # 输出形状 (B, num_keypoints, H, W)
