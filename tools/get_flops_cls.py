# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info

from mmcls.models import build_classifier
from get_flops import sra_flops

def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_classifier(cfg.model)
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)

    _, H, W = input_shape
    backbone1 = sra_flops(H // 4, W // 4,
                          8,
                          32,
                          1) * 2
    backbone2 = sra_flops(H // 8, W // 8,
                          4,
                          64,
                          2) * 2
    backbone3 = sra_flops(H // 16, W // 16,
                          2,
                          160,
                          5) * 2
    backbone4 = sra_flops(H // 32, W // 32,
                          1,
                          32,
                          256) * 2
    flops = flops + backbone1 + backbone2 + backbone3 + backbone4
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
