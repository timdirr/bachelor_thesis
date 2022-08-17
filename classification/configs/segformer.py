# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MixVisionTransformerFullRes',
        in_channels=3,
        embed_dims=32,
        num_stages=6,
        num_layers=[0, 0, 2, 2, 2, 2],
        num_heads=[1, 1, 1, 2, 5, 8],
        patch_sizes=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, 2],
        sr_ratios=[0, 0, 8, 4, 2, 1],
        out_indices=(0, 1, 2, 3, 4, 5),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    neck=None,
    # neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ]))
