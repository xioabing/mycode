# import pandas as pd
# import matplotlib.pyplot as plt
# hw=pd.read_csv('D:/yolo/yolov5_all-main/mask_yolo_format/labels/new12.csv')#导入csv文件
# plt.scatter(hw['w'], hw['h'],s=2)#s指的是点的面积
# plt.xlabel("squar")
# plt.ylabel("height/weight")
# #画出散点图
# plt.show()
# #将散点图显示出来


# import torch
# import torch.nn as nn
# m = nn.AdaptiveAvgPool2d((5,1))
# m1 = nn.AdaptiveAvgPool2d((None,1))
# m2 = nn.AdaptiveAvgPool2d(1)
# input = torch.randn(2, 64, 8, 9)
# output = m(input)
# output1 = m1(input)
# output2 = m2(input)
# print('nn.AdaptiveAvgPool2d((5,1)):',output.shape)
# print('nn.AdaptiveAvgPool2d((None,5)):',output1.shape)
# print('nn.AdaptiveAvgPool2d(1):',output2.shape)


import torch
from torch import nn


class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out


if __name__ == '__main__':
    x = torch.randn(1, 16, 128, 64)  # b, c, h, w
    ca_model = CA_Block(channel=16, h=128, w=64)
    y = ca_model(x)
    print(y.shape)