import os
import shutil
import argparse
from mmseg_test import testing_or_eval
MY_PATHS = [
#    "/data/Code/timformer/training",
    "/data/Code/timformer/att_training",
#    "/data/Code/timformer/att_extra_training",
    "/data/Code/timformer/add_att_training",
    "/data/Code/timformer/pretrain_weight_testing/training",
]
for MY_PATH in MY_PATHS:
    print(MY_PATH)
    for dir in os.listdir(MY_PATH):
        pth_file = None
        conf_file = None
        for file in os.listdir(os.path.join(MY_PATH, dir)):
            if file.startswith("iter_300000.pth"):
                pth_file = file
            if file.endswith(".py"):
                conf_file = file
        if pth_file and conf_file:
            args = argparse.Namespace(aug_test=False, cfg_options=None, checkpoint=os.path.join(MY_PATH, dir, pth_file), config=os.path.join(MY_PATH, dir, conf_file),
                  eval=['mIoU'], eval_options=None, format_only=False, gpu_collect=False, gpu_id=0,
                  launcher='none', local_rank=0, opacity=0.5, options=None, out=None, show=False, show_dir=None,
                  tmpdir=None, work_dir=os.path.join('/data/Code/timformer/save_results', dir))
            testing_or_eval(args)
            if dir.startswith('6_Att') or dir.startswith('5_Att') or dir.startswith('4_Att') or dir.startswith('1_Att') or dir.startswith('4_top') or dir.startswith('5_top'):
                pth_file = "iter_272000.pth"
                args = argparse.Namespace(aug_test=False, cfg_options=None,
                                          checkpoint=os.path.join(MY_PATH, dir, pth_file),
                                          config=os.path.join(MY_PATH, dir, conf_file),
                                          eval=['mIoU'], eval_options=None, format_only=False, gpu_collect=False,
                                          gpu_id=0,
                                          launcher='none', local_rank=0, opacity=0.5, options=None, out=None,
                                          show=False, show_dir=None,
                                          tmpdir=None, work_dir=os.path.join('/data/Code/timformer/save_results', dir))
                testing_or_eval(args)
            if dir.startswith('3_top'):
                pth_file = "iter_240000.pth"
                args = argparse.Namespace(aug_test=False, cfg_options=None,
                                          checkpoint=os.path.join(MY_PATH, dir, pth_file),
                                          config=os.path.join(MY_PATH, dir, conf_file),
                                          eval=['mIoU'], eval_options=None, format_only=False, gpu_collect=False,
                                          gpu_id=0,
                                          launcher='none', local_rank=0, opacity=0.5, options=None, out=None,
                                          show=False, show_dir=None,
                                          tmpdir=None, work_dir=os.path.join('/data/Code/timformer/save_results', dir))
                testing_or_eval(args)
            if dir.startswith('pretrain_baseline'):
                pth_file = "iter_230000.pth"
                args = argparse.Namespace(aug_test=False, cfg_options=None,
                                          checkpoint=os.path.join(MY_PATH, dir, pth_file),
                                          config=os.path.join(MY_PATH, dir, conf_file),
                                          eval=['mIoU'], eval_options=None, format_only=False, gpu_collect=False,
                                          gpu_id=0,
                                          launcher='none', local_rank=0, opacity=0.5, options=None, out=None,
                                          show=False, show_dir=None,
                                          tmpdir=None, work_dir=os.path.join('/data/Code/timformer/save_results', dir))
                testing_or_eval(args)
            if dir.startswith('pretrain_0'):
                pth_file = "iter_230000.pth"
                args = argparse.Namespace(aug_test=False, cfg_options=None,
                                          checkpoint=os.path.join(MY_PATH, dir, pth_file),
                                          config=os.path.join(MY_PATH, dir, conf_file),
                                          eval=['mIoU'], eval_options=None, format_only=False, gpu_collect=False,
                                          gpu_id=0,
                                          launcher='none', local_rank=0, opacity=0.5, options=None, out=None,
                                          show=False, show_dir=None,
                                          tmpdir=None,
                                          work_dir=os.path.join('/data/Code/timformer/save_results', dir))
                testing_or_eval(args)


            # ausg = argparse.Namespace(aug_test=False, cfg_options=None, checkpoint='testing/model.pth', config='timformer.py',
            #           eval=['cityscapes'], eval_options=None, format_only=False, gpu_collect=False, gpu_id=0,
            #           launcher='none', local_rank=0, opacity=0.5, options=None, out=None, show=False, show_dir=None,
            #           tmpdir=None, work_dir='save_results')
