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
        self.dec_blocks = nn.ModuleList[Block(chs[0], chs[0])]
        for i in range(1, len(chs)):
            self.dec_blocks.append(Block(chs[i] * 2, chs[i]))

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        # 4 inputs
        # first input into block -> 256 to 256
        # then upsample 256 -> 160
        # concat 160 + 160 -> 320
        # next block 320 -> 160
        # upsample 160 -> 64
        # concat 64 + 64 -> 128
        # block 128 -> 64
        # upsample 64 -> 32
        # concat 32 + 32 -> 64
        # block 64 -> 32
        # classification from 32 -> 19

        # blocks [0 -> 0, 2 * 1 -> 1, 2 * 2 -> 2, 2 * 3 -> 3]
        # ups [0 -> 1, 1 -> 2, 2 -> 3]

        inputs = self._transform_inputs(inputs)
        inputs = inputs[::-1]
        x = self.dec_blocks[0](inputs[0])
        for idx in range(len(inputs)-1):
            x = self.upconvs[idx](x)
            next = inputs[idx + 1]
            x = torch.cat([x, next], dim=1)
            x = self.dec_blocks[idx + 1](x)

        out = self.cls_seg(x)
        return out
