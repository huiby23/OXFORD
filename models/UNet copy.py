import torch
from torch import nn


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet_Binary_Branch(nn.Module):
    def __init__(self, img_ch=1, output_ch=2):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=8)
        self.Conv2 = conv_block(ch_in=8, ch_out=16)
        self.Conv3 = conv_block(ch_in=16, ch_out=32)
        self.Conv4 = conv_block(ch_in=32, ch_out=64)
        self.Conv5 = conv_block(ch_in=64, ch_out=128)

        self.Loc_Up5 = up_conv(ch_in=128, ch_out=64)
        self.Loc_Up_conv5 = conv_block(ch_in=64, ch_out=64)
        self.Loc_Up4 = up_conv(ch_in=64, ch_out=32)
        self.Loc_Up_conv4 = conv_block(ch_in=32, ch_out=32)
        self.Loc_Up3 = up_conv(ch_in=32, ch_out=16)
        self.Loc_Up_conv3 = conv_block(ch_in=16, ch_out=16)
        self.Loc_Up2 = up_conv(ch_in=16, ch_out=8)
        self.Loc_Up_conv2 = conv_block(ch_in=8, ch_out=8)

        self.Loc_Conv_1x1 = nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)

        self.Score_Up5 = up_conv(ch_in=128, ch_out=64)
        self.Score_Up_conv5 = conv_block(ch_in=64, ch_out=64)
        self.Score_Up4 = up_conv(ch_in=64, ch_out=32)
        self.Score_Up_conv4 = conv_block(ch_in=32, ch_out=32)
        self.Score_Up3 = up_conv(ch_in=32, ch_out=16)
        self.Score_Up_conv3 = conv_block(ch_in=16, ch_out=16)
        self.Score_Up2 = up_conv(ch_in=16, ch_out=8)
        self.Score_Up_conv2 = conv_block(ch_in=8, ch_out=8)

        self.Score_Conv_1x1 = nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        loc_d5 = self.Up5(x5)
        loc_d5 = torch.cat((x4, loc_d5), dim=1)
        loc_d5 = self.Up_conv5(loc_d5)
        loc_d4 = self.Up4(loc_d5)
        loc_d4 = torch.cat((x3, loc_d4), dim=1)
        loc_d4 = self.Up_conv4(loc_d4)
        loc_d3 = self.Up3(loc_d4)
        loc_d3 = torch.cat((x2, loc_d3), dim=1)
        loc_d3 = self.Up_conv3(loc_d3)
        loc_d2 = self.Up2(loc_d3)
        loc_d2 = torch.cat((x1, loc_d2), dim=1)
        loc_d2 = self.Up_conv2(loc_d2)

        loc_d1 = self.Conv_1x1(loc_d2)

        score_d5 = self.Up5(x5)
        score_d5 = torch.cat((x4, score_d5), dim=1)
        score_d5 = self.Up_conv5(score_d5)
        score_d4 = self.Up4(score_d5)
        score_d4 = torch.cat((x3, d4), dim=1)
        score_d4 = self.Up_conv4(score_d4)
        score_d3 = self.Up3(score_d4)
        score_d3 = torch.cat((x2, score_d3), dim=1)
        score_d3 = self.Up_conv3(score_d3)
        score_d2 = self.Up2(score_d3)
        score_d2 = torch.cat((x1, score_d2), dim=1)
        score_d2 = self.Up_conv2(score_d2)

        score_d1 = self.Conv_1x1(score_d2)
        return loc_d1,score_d1


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Up5 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv5 = conv_block(ch_in=64, ch_out=64)
        self.Up4 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv4 = conv_block(ch_in=32, ch_out=32)
        self.Up3 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv3 = conv_block(ch_in=16, ch_out=16)
        self.Up2 = up_conv(ch_in=16, ch_out=8)
        self.Up_conv2 = conv_block(ch_in=8, ch_out=8)

    def forward(self, x1, x2, x3, x4, x5):
        d5 = self.Up5(x5)
        d5 = self.Up_conv5(torch.cat((x4, d5), dim=1))
        d4 = self.Up4(d5)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))
        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))
        return d2