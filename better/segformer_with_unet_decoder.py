# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


@HEADS.register_module()
class SegformerUNetDecoder(BaseDecodeHead):
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        chs = self.in_channels[::-1]
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i+1] * 2, chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32

        inputs = self._transform_inputs(inputs)
        inputs = inputs[::-1]
        x = inputs[0]
        for idx in range(len(inputs)-1):
            x = self.upconvs[idx](x)
            next = inputs[idx + 1]
            x = torch.cat([x, next], dim=1)
            x = self.dec_blocks[idx](x)

        out = self.cls_seg(x)
        return out
