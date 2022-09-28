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
    "/data/Code/timformer/pretrain_weight_testing/training_full",
]
DEST_PATH = "/data/Code/timformer/tf_sum"

for MY_PATH in MY_PATHS:
    print(MY_PATH)
    for dir in os.listdir(MY_PATH):
        for file in os.listdir(os.path.join(MY_PATH, dir, "tf_logs")):
            new_name = "events.out.tfevents." + dir + ".0"
            shutil.copyfile(os.path.join(MY_PATH, dir, "tf_logs", file), os.path.join(DEST_PATH, new_name))
