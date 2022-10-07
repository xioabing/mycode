import torch
from torch import nn

from models.common import conv_bn,conv_bn_relu_maxpool
import torch.nn.functional as F




class ASFF(nn.Module):
    def __init__(self, level, activate, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 128]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = conv_bn(256, self.inter_dim, kernel=3, stride=2)
            self.stride_level_2 = conv_bn(128, self.inter_dim, kernel=3, stride=2)
            self.expand = conv_bn(self.inter_dim, 512, kernel=3, stride=1)
        elif level == 1:
            self.compress_level_0 = conv_bn(512, self.inter_dim, kernel=1)
            self.stride_level_2 = conv_bn(128, self.inter_dim, kernel=3, stride=2)
            self.expand = conv_bn(self.inter_dim, 256, kernel=3, stride=1)
        elif level == 2:
            self.compress_level_0 = conv_bn(512, self.inter_dim, kernel=1, stride=1)
            self.compress_level_1= conv_bn(256,self.inter_dim,kernel=1,stride=1)
            self.expand = conv_bn(self.inter_dim, 128, kernel=3, stride=1)
        compress_c = 8 if rfb else 16
        self.weight_level_0 = conv_bn(self.inter_dim, compress_c, 1, 1, 0)
        self.weight_level_1 = conv_bn(self.inter_dim, compress_c, 1, 1, 0)
        self.weight_level_2 = conv_bn(self.inter_dim, compress_c, 1, 1, 0)
        self.weight_levels = conv_bias(compress_c * 3, 3, kernel=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            sh = torch.tensor(level_0_compressed.shape[-2:])*2
            level_0_resized = F.interpolate(level_0_compressed, tuple(sh), 'nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            sh = torch.tensor(level_0_compressed.shape[-2:])*4
            level_0_resized = F.interpolate(level_0_compressed, tuple(sh), 'nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            sh = torch.tensor(level_1_compressed.shape[-2:])*2
            level_1_resized = F.interpolate(level_1_compressed, tuple(sh),'nearest')
            level_2_resized = x_level_2
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
'''代码2'''
if __name__ == '__main__':
    model=ASFF(1,activate='leaky')
    l1=torch.ones(1,512,10,10)
    l2=torch.ones(1,256,20,20)
    l3=torch.ones(1,128,40,40)
    out=model(l1,l2,l3)
    print(out.shape)