_base_ = [
    'configs/segformer_baseline.py', 'configs/imagenet_bs32.py',
    'configs/schedule.py', 'configs/default_runtime.py'
]

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    })
optimizer = dict(paramwise_cfg=paramwise_cfg)