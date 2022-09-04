_base_ = [
    '/data/Code/timformer/pretrain_weight_testing/configs/model/reversed/segformer_2_attention.py',
    '/data/Code/timformer/pretrain_weight_testing/configs/base/cityscapes_192x192.py',
    '/data/Code/timformer/pretrain_weight_testing/configs/base/default_runtime.py',
    '/data/Code/timformer/pretrain_weight_testing/configs/base/schedule_att.py'
]
checkpoint = '/data/Code/timformer/pretrain/valid_pretrained.pth'  # noqa

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    test_cfg=dict(mode='slide', crop_size=(192, 192), stride=(192, 192)))
    # test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=8, workers_per_gpu=8)