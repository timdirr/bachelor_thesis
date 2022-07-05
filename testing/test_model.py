from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot, single_gpu_test
from mmseg.core.evaluation import get_palette
from mmseg.datasets import build_dataloader, build_dataset
from mmcv import Config
from mmseg.models import build_segmentor

config_file = 'timformer_testing.py'
checkpoint_file = 'model.pth'

cfg = Config.fromfile(config_file)
cfg.load_from = checkpoint_file

# Since we use only one GPU, BN is used instead of SyncBN
# cfg.norm_cfg = dict(type='BN', requires_grad=True)
# cfg.model.backbone.norm_cfg = cfg.norm_cfg
# cfg.model.decode_head.norm_cfg = cfg.norm_cfg

# Set up working dir to save files and logs.
cfg.work_dir = './save'

cfg.gpu_ids = range(1)
cfg.seed = 0

model = init_segmentor(config_file, checkpoint_file, device='cpu')

dataset = build_dataset(cfg.data.val)
loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        seed=cfg.seed,
        drop_last=True)
loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })

train_loader_cfg = {**loader_cfg, **cfg.data.get('val_dataloader', {})}
data_loader = build_dataloader(dataset, **train_loader_cfg)


# model.show_result(img, result, palette=get_palette('cityscapes'), show=True, opacity=0.5, out_file='output.png')

single_gpu_test(model, data_loader=data_loader, show=False, out_dir='save')
