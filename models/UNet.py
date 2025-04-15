import torch
import torch.nn.functional as F
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
        self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),


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
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Dual_UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=8)
        self.Conv2 = conv_block(ch_in=8, ch_out=16)
        self.Conv3 = conv_block(ch_in=16, ch_out=32)
        self.Conv4 = conv_block(ch_in=32, ch_out=64)
        self.Conv5 = conv_block(ch_in=64, ch_out=128)

        self.loc_decoder = Decoder(128,64,32,16,8)
        self.score_decoder = Decoder(128,64,32,16,8)

        self.Loc_Conv_1x1 = nn.Conv2d(8, output_ch, kernel_size=1)
        # self.Loc_Conv_1x1 = nn.Sequential(
        #                                     nn.Conv2d(8, output_ch, kernel_size=1),
        #                                     nn.Sigmoid()
        # )
        self.Score_Conv_1x1 = nn.Sequential(
                                            nn.Conv2d(8, output_ch, kernel_size=1),
                                            nn.Sigmoid()
        )
    

    def Descriptor(self,x1,x2,x3,x4,x5):
        x2= F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x3= F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=True)
        x4= F.interpolate(x4, scale_factor=8, mode='bilinear', align_corners=True)
        x5= F.interpolate(x5, scale_factor=16, mode='bilinear', align_corners=True)
        descriptors_map=torch.cat((x1,x2,x3,x4,x5), dim=1)
        
        return descriptors_map

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))

        location_feature=self.loc_decoder(x1, x2, x3, x4, x5)
        score_feature=self.score_decoder(x1, x2, x3, x4, x5)
        location_map=self.Loc_Conv_1x1(location_feature)#(B,1,H,W)
        scores_map=self.Score_Conv_1x1(score_feature) #(B,1,H,W)
        descriptors_map = self.Descriptor(x1, x2, x3, x4, x5) #(B,248,H,W)
        # locations_keypoints = self.spatial_softmax_keypoints(self.Loc_Conv_1x1(location_feature)) #(B,400,2)
        # scores_map = nn.Sigmoid(self.Score_Conv_1x1(score_feature)) #(B,1,H,W)
        
        # descriptors = self.extract_keypoint_descriptors(descriptors_map, locations_keypoints)#(B,400,1)

        return location_map, scores_map, descriptors_map 

class Decoder(nn.Module):
    def __init__(self,ch1,ch2,ch3,ch4,ch5):
        super().__init__()
        self.Up5 = up_conv(ch_in=ch1, ch_out=ch2)#up_sample
        self.Up_conv5 = conv_block(ch_in=ch1, ch_out=ch2)
        self.Up4 = up_conv(ch_in=ch2, ch_out=ch3)#up_sample
        self.Up_conv4 = conv_block(ch_in=ch2, ch_out=ch3)
        self.Up3 = up_conv(ch_in=ch3, ch_out=ch4)#up_sample
        self.Up_conv3 = conv_block(ch_in=ch3, ch_out=ch4)
        self.Up2 = up_conv(ch_in=ch4, ch_out=ch5)#up_sample
        self.Up_conv2 = conv_block(ch_in=ch4, ch_out=ch5)

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