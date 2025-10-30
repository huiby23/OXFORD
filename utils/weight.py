import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models


"""
class D2Net(nn.Module):
    def __init__(self, model_file=None, use_cuda=True):
        super(D2Net, self).__init__()

        self.dense_feature_extraction = DenseFeatureExtractionModule(
            finetune_feature_extraction=True,
            use_cuda=use_cuda
        )

        self.detection = SoftDetectionModule()

        if model_file is not None:
            if use_cuda:
                self.load_state_dict(torch.load(model_file)['model'])
            else:
                self.load_state_dict(torch.load(model_file, map_location='cpu')['model'])

    def forward(self, batch):
        b = batch['image1'].size(0)

        dense_features = self.dense_feature_extraction(
            torch.cat([batch['image1'], batch['image2']], dim=0)
        )

        scores = self.detection(dense_features)

        dense_features1 = dense_features[: b, :, :, :]
        dense_features2 = dense_features[b :, :, :, :]

        scores1 = scores[: b, :, :]
        scores2 = scores[b :, :, :]

        return {
            'dense_features1': dense_features1,
            'scores1': scores1,
            'dense_features2': dense_features2,
            'scores2': scores2
        }
"""


"""
class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, finetune_feature_extraction=False, use_cuda=True):
        super(DenseFeatureExtractionModule, self).__init__()

        model = models.vgg16()
        vgg16_layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
            'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
            'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
            'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
            'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'pool5'
        ]
        conv4_3_idx = vgg16_layers.index('conv4_3')

        self.model = nn.Sequential(
            *list(model.features.children())[: conv4_3_idx + 1]
        )

        self.num_channels = 512

        # Fix forward parameters
        for param in self.model.parameters():
            param.requires_grad = False
        if finetune_feature_extraction:
            # Unlock conv4_3
            for param in list(self.model.parameters())[-2 :]:
                param.requires_grad = True

        if use_cuda:
            self.model = self.model.cuda()

    def forward(self, batch):
        output = self.model(batch)
        return output
"""


class SoftDetectionModule(nn.Module):
    def __init__(self, config):
        super(SoftDetectionModule, self).__init__()

        # default: soft_local_max_size = 3
        self.soft_local_max_size = config['networks']['detector']['soft_local_max_size']

        self.pad = self.soft_local_max_size // 2

    def forward(self, batch):
        b = batch.size(0)

        batch = F.relu(batch)

        max_per_sample = torch.max(batch.view(b, -1), dim=1)[0]
        exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1))
        sum_exp = (
            self.soft_local_max_size ** 2 *
            F.avg_pool2d(
                F.pad(exp, [self.pad] * 4, mode='constant', value=1.),
                self.soft_local_max_size, stride=1
            )
        )
        local_max_score = exp / sum_exp

        depth_wise_max = torch.max(batch, dim=1)[0]
        depth_wise_max_score = batch / depth_wise_max.unsqueeze(1)

        all_scores = local_max_score * depth_wise_max_score
        score = torch.max(all_scores, dim=1)[0]

        score = score / torch.sum(score.view(b, -1), dim=1).view(b, 1, 1)

        return score.unsqueeze(1)   # (N, 1, H, W)





"""
class D2Net(nn.Module):
    def __init__(self, model_file=None, use_relu=True, use_cuda=True):
        super(D2Net, self).__init__()

        self.dense_feature_extraction = DenseFeatureExtractionModule(
            use_relu=use_relu, use_cuda=use_cuda
        )

        self.detection = HardDetectionModule()

        self.localization = HandcraftedLocalizationModule()

        if model_file is not None:
            if use_cuda:
                self.load_state_dict(torch.load(model_file)['model'])
            else:
                self.load_state_dict(torch.load(model_file, map_location='cpu')['model'])

    def forward(self, batch):
        _, _, h, w = batch.size()
        dense_features = self.dense_feature_extraction(batch)

        detections = self.detection(dense_features)

        displacements = self.localization(dense_features)

        return {
            'dense_features': dense_features,
            'detections': detections,
            'displacements': displacements
        }
"""


class HardDetectionModule(nn.Module):
    def __init__(self, edge_threshold=5):
        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = (batch == depth_wise_max)   # depth_wise_max.unsqueeze(1)
        del depth_wise_max

        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)
        is_local_max = (batch == local_max)
        del local_max

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        detected = torch.min(
            is_depth_wise_max,
            torch.min(is_local_max, is_not_edge)
        )
        del is_depth_wise_max, is_local_max, is_not_edge

        return detected


class HandcraftedLocalizationModule(nn.Module):
    def __init__(self):
        super(HandcraftedLocalizationModule, self).__init__()

        self.di_filter = torch.tensor(
            [[0, -0.5, 0], [0, 0, 0], [0,  0.5, 0]]
        ).view(1, 1, 3, 3)
        self.dj_filter = torch.tensor(
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]
        ).view(1, 1, 3, 3)

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)
        det = dii * djj - dij * dij

        inv_hess_00 = djj / det
        inv_hess_01 = -dij / det
        inv_hess_11 = dii / det
        del dii, dij, djj, det

        di = F.conv2d(
            batch.view(-1, 1, h, w), self.di_filter.to(device), padding=1
        ).view(b, c, h, w)
        dj = F.conv2d(
            batch.view(-1, 1, h, w), self.dj_filter.to(device), padding=1
        ).view(b, c, h, w)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)
        del inv_hess_00, inv_hess_01, inv_hess_11, di, dj

        return torch.stack([step_i, step_j], dim=1)
