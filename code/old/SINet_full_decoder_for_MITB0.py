# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st


def _get_kernel(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    """
        normalization
    :param: in_
    :return:
    """
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8)


class SA(nn.Module):
    """
        holistic attention src
    """
    def __init__(self):
        super(SA, self).__init__()
        gaussian_kernel = np.float32(_get_kernel(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention, x):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)       # normalization
        x = torch.mul(x, soft_attention.max(attention))     # mul
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


@HEADS.register_module()
class AdjustedSINetHead(BaseDecodeHead):

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.upsample_x2 = nn.Upsample(scale_factor=2, mode=self.interpolate_mode, align_corners=self.align_corners)
        self.conv_c1 = BasicConv2d(self.channels, self.channels, 3, padding=1)
        self.conv_c2 = BasicConv2d(2*self.channels, 2*self.channels, 3, padding=1)
        self.conv_c3 = BasicConv2d(3*self.channels, 3*self.channels, 3, padding=1)
        self.conv_c4 = BasicConv2d(4*self.channels, 4*self.channels, 3, padding=1)
        self.conv_c4to1 = BasicConv2d(4*self.channels, self.channels, 3, padding=1)

        self.SA = SA()

        # self.conv_for_x2 = ConvModule(
        #     in_channels=self.in_channels[1],
        #     out_channels=self.channels * 16,
        #     kernel_size=1,
        #     stride=1,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)


    def pdc_sm_forward(self, x1, x2, x3, x4):
        x1_1 = x1
        x2_1 = self.conv_c1(self.upsample_x2(x1)) * x2
        x3_1 = self.conv_c1(self.upsample_x2(self.upsample_x2(x1))) * self.conv_c1(self.upsample_x2(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_c1(self.upsample_x2(x1_1))), 1)
        x2_2 = self.conv_c2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_c2(self.upsample_x2(x2_2)), x4), 1)
        x3_2 = self.conv_c4(x3_2)

        x = self.conv_c4to1(x3_2)
        x = self.cls_seg(x)

        return x

    def pdc_im_forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_c1(self.upsample(x1)) * x2
        x3_1 = self.conv_c1(self.upsample_x2(self.upsample_x2(x1))) * self.conv_c1(self.upsample_x2(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_c1(self.upsample_x2(x1_1))), 1)
        x2_2 = self.conv_c2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_c2(self.upsample_x2(x2_2))), 1)
        x3_2 = self.conv_c3(x3_2)

        x = self.conv_c3(x3_2)
        x = self.cls_seg(x)

        return x

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        x = inputs[0]
        conv = self.convs[0]
        outs.append(
            resize(
                input=conv(x),
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners
            )
        )
        for idx in range(1, len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=[x * 2 for x in inputs[idx].shape[2:]],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # PDC1
        out = self.pdc_sm_forward(outs[3], outs[2], outs[1], outs[0])

        # # SA
        # x2 = inputs[1]
        # x2 = resize(
        #         input=self.conv_for_x2(x2),
        #         size=x2.shape[2:],
        #         mode=self.interpolate_mode,
        #         align_corners=self.align_corners
        #     )
        # sa_out = self.SA(map_sm.sigmoid(), x2)
        #
        # # PDC2 (IM) (EXPERIMENTAL)
        # x3_1 = self.conv_for_x3_1
        # x4_1 = self.conv_for_x4_1

        # out = self.fusion_conv(torch.cat(outs, dim=1))
        #
        # out = self.cls_seg(out)

        return out
