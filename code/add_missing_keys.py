# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_mit(org, ckpt):
    new_ckpt = OrderedDict()
    # Process the concat between q linear weights and kv linear weights
    for k, v in ckpt.items():

        if k.startswith('head.fc.weight'):
            print(v)
        if k.startswith('head.fc.bias'):
            print(v)
        if k.startswith('xxx'):
            continue
        new_k = k
        new_v = v
        new_ckpt[new_k] = new_v
    for k, v in org.items():
        if k.startswith('head.fc.weight'):
            new_k = k
            new_v = v
            new_ckpt[new_k] = new_v
        if k.startswith('head.fc.bias'):
            new_k = k
            new_v = v
            new_ckpt[new_k] = new_v

    return new_ckpt



def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained segformer to '
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    checkpoint2 = CheckpointLoader.load_checkpoint(args.dst, map_location='cpu')
    if 'state_dict' in checkpoint2:
        state_dict2 = checkpoint2['state_dict']
    elif 'model' in checkpoint2:
        state_dict2 = checkpoint2['model']
    else:
        state_dict2 = checkpoint2



    weight = convert_mit(state_dict, state_dict2)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
