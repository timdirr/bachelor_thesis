import os
import shutil
import argparse
from mmseg_test import testing_or_eval
from change_keys import real_key
import json
MY_PATHS = [
    "/data/Code/timformer/att_training",
    "/data/Code/timformer/add_att_training",
    "/data/Code/timformer/pretrain_weight_testing/training",
    "/data/Code/timformer/pretrain_weight_testing/training_full",
]

with open("/data/Code/timformer/tf_sum/results.json", "r") as f:
    data = json.load(f)

with open("/data/Code/timformer/gathered_data/iters.json", "r") as f:
    used_iters = json.load(f)

for MY_PATH in MY_PATHS:
    print(MY_PATH)
    for dir in os.listdir(MY_PATH):
        pth_file = None
        conf_file = None
        for file in os.listdir(os.path.join(MY_PATH, dir)):
            if file.endswith(".py"):
                conf_file = file
        # for d in data[dir]:
        #     pth_file = "iter_" + str(d) + ".pth"
        #     if conf_file:
        #         args = argparse.Namespace(aug_test=False, cfg_options=None, checkpoint=os.path.join(MY_PATH, dir, pth_file), config=os.path.join(MY_PATH, dir, conf_file),
        #               eval=['mDice'], eval_options=None, format_only=False, gpu_collect=False, gpu_id=0,
        #               launcher='none', local_rank=0, opacity=0.5, options=None, out=None, show=True, show_dir=os.path.join('/data/Code/timformer/save_results/eval/mDice', dir, "iter_" + str(d)),
        #               tmpdir=None, work_dir=os.path.join('/data/Code/timformer/save_results/eval/mDice', dir, "iter_" + str(d)))
        #         testing_or_eval(args)
        pth_file = used_iters[real_key(dir)] + ".pth"
        if conf_file:
            args = argparse.Namespace(aug_test=False, cfg_options=None, checkpoint=os.path.join(MY_PATH, dir, pth_file), config=os.path.join(MY_PATH, dir, conf_file),
                  eval=['mIoU'], eval_options=None, format_only=False, gpu_collect=False, gpu_id=0,
                  launcher='none', local_rank=0, opacity=0.5, options=None, out=None, show=True, show_dir=os.path.join('/data/Code/timformer/save_results/eval/print_images', real_key(dir)),
                  tmpdir=None, work_dir=os.path.join('/data/Code/timformer/save_results/eval/print_images', real_key(dir)))
            testing_or_eval(args)
