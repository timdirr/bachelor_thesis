# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmseg.models.backbones.mit import *


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


@HEADS.register_module()
class UNetAttentionHeadFull(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self,
                 embed_dims=32,
                 num_layers=[2, 2],
                 num_heads=[8, 5],
                 sr_ratios=[1, 2, 4, 8, 16, 32],
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 patch_sizes=[3, 3],
                 with_cp=False,
                 num_attention_modules=1,
                 interpolate_mode='bilinear',
                 total_stages = 6,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        self.num_attention_modules = num_attention_modules
        self.total_stages = total_stages
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.layers = nn.ModuleList()

        dpr = [
            x.item()
            for x in torch.linspace(drop_path_rate, 0, sum([num_layers[val] for val in range(num_attention_modules)]))
        ]

        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(self.in_channels[i], self.in_channels[i + 1], 2, 2) for i in
                                      range(len(self.in_channels) - 1)])
        for i in range(len(self.in_channels) - 1, total_stages-1):
            self.upconvs.append(nn.ConvTranspose2d(self.in_channels[num_inputs - 1], self.in_channels[num_inputs - 1], 2, 2))
        for i in range(1, num_inputs):
            self.in_channels[i] = self.in_channels[i] * 2
        cur = 0
        for i in range(min(num_attention_modules, num_inputs)):
            att_layer = nn.ModuleList()
            att = nn.ModuleList()
            num_layer = num_layers[i]
            embed_dims_i = embed_dims * num_heads[i]
            att_layer.append(PatchEmbed(
                in_channels=self.in_channels[i],
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=1,
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg))
            for idx in range(num_layer):
                att.append(TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]))
            att_layer.append(att)
            att_layer.append(build_norm_layer(norm_cfg, embed_dims_i)[1])
            self.layers.append(att_layer)
            cur += num_layer

        for i in range(num_attention_modules, num_inputs):
            if i == 0:
                self.layers.append(
                    UNetBlock(self.in_channels[i], self.in_channels[i])
                )
            else:
                self.layers.append(
                    UNetBlock(self.in_channels[i], (self.in_channels[i] // 2))
                )
        for i in range(num_inputs, total_stages):
            if i >= self.num_attention_modules:
                self.layers.append(
                    UNetBlock(self.in_channels[num_inputs - 1] // 2, (self.in_channels[num_inputs - 1] // 2))
                )
            else:
                att_layer = nn.ModuleList()
                att = nn.ModuleList()
                num_layer = num_layers[i]
                embed_dims_i = embed_dims * num_heads[i]
                att_layer.append(PatchEmbed(
                    in_channels=self.in_channels[num_inputs - 1] // 2,
                    embed_dims=embed_dims_i,
                    kernel_size=patch_sizes[i],
                    stride=1,
                    padding=patch_sizes[i] // 2,
                    norm_cfg=norm_cfg))
                for idx in range(num_layer):
                    att.append(TransformerEncoderLayer(
                        embed_dims=embed_dims_i,
                        num_heads=num_heads[i],
                        feedforward_channels=mlp_ratio * embed_dims_i,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[cur + idx],
                        qkv_bias=qkv_bias,
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg,
                        with_cp=with_cp,
                        sr_ratio=sr_ratios[i]))
                att_layer.append(att)
                att_layer.append(build_norm_layer(norm_cfg, embed_dims_i)[1])
                self.layers.append(att_layer)
                cur += num_layer


    def forward(self, inputs):
        # 4 inputs
        # first input into attention -> 256 to 256
        # then upsample 256 -> 160

        # concat 160 + 160 -> 320
        # next attention 320 -> 160
        # upsample 160 -> 64

        # concat 64 + 64 -> 128
        # first block 128 -> 64
        # upsample 64 -> 32

        # concat 32 + 32 -> 64
        # next block 64 -> 32
        # classification from 32 -> 19

        # layers [a 0 -> 0, a 2 * 1 -> 1, b 2 * 2 -> 2, b 2 * 3 -> 3]
        # ->
        # att[0 -> 0, 2 * 1 -> 1]
        # blocks [2 * 2 -> 2, 2 * 3 -> 3]

        # ups [0 -> 1, 1 -> 2, 2 -> 3]

        inputs = self._transform_inputs(inputs)
        inputs = inputs[::-1]
        if self.num_attention_modules != 0:
            x = self.apply_attention(inputs[0], self.layers[0])
        else:
            x = self.layers[0](inputs[0])

        for idx in range(len(inputs) - 1):
            x = self.upconvs[idx](x)
            next = inputs[idx + 1]
            x = torch.cat([x, next], dim=1)
            if(idx < self.num_attention_modules - 1):
                x = self.apply_attention(x, self.layers[idx + 1])
            else:
                x = self.layers[idx + 1](x)

        for idx in range(len(inputs) - 1, self.total_stages-1):
            x = self.upconvs[idx](x)
            if (idx < self.num_attention_modules - 1):
                x = self.apply_attention(x, self.layers[idx + 1])
            else:
                x = self.layers[idx + 1](x)

        out = self.cls_seg(x)
        return out

    @staticmethod
    def apply_attention(x, layer):
        x, hw_shape = layer[0](x)
        for l in layer[1]:
            x = l(x, hw_shape)
        x = layer[2](x)
        x = nlc_to_nchw(x, hw_shape)
        return x

