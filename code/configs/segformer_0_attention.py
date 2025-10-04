# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
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
    decode_head=dict(
        type='UNetAttentionHead',
        in_channels=[256, 160, 64, 32, 32, 32],
        num_attention_modules=0,
        num_layers=[],
        num_heads=[],
        patch_sizes=[],
        in_index=[0, 1, 2, 3, 4, 5],
        channels=32,
        dropout_ratio=0.1,
        num_classes=19,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


