import os
import shutil
import argparse
from mmseg_test import testing_or_eval
import json
from change_keys import real_key
# MY_PATH = "/data/Code/timformer/save_results/eval"
MY_PATH = "/data/Code/timformer/save_results/eval/mIoU"
# MY_PATH = "/data/Code/timformer/save_results/eval/mDice"



full_data={}
for dir in os.listdir(MY_PATH):
    if dir == "mIoU":
        continue
    if dir == "mDice":
        continue
    # temp = {"averageScoreClasses" : 0}
    temp = {"mIoU" : 0}
    # temp = {"mDice" : 0}
    for iter in os.listdir(os.path.join(MY_PATH, dir)):
        for file in os.listdir(os.path.join(MY_PATH, dir, iter)):
            with open(os.path.join(MY_PATH, dir, iter, file), "r") as f:
                data = json.load(f)
                # if data['metric']['averageScoreClasses'] > temp['averageScoreClasses']:
                if data['metric']['mIoU'] > temp['mIoU']:
                # if data['metric']['mDice'] > temp['mDice']:
                    temp = data['metric']
                    temp['iter'] = iter
    # del temp['confMatrix']
    # del temp['priors']
    # del temp['perImageScores']
    full_data[real_key(dir)] = temp


# #mIoU
# mIoU = {}
# for key, value in full_data.items():
#     mIoU[key] = round(value['averageScoreClasses']* 100, 2)
#
# #class_mIoU
# class_mIoU = {}
# for key, value in full_data.items():
#     useless = []
#     for k, v in value['classScores'].items():
#         if str(v) == "nan":
#             useless.append(k)
#         else:
#             value['classScores'][k] = round(value['classScores'][k]* 100, 2)
#     for k in useless:
#         del value['classScores'][k]
#     class_mIoU[key] = value['classScores']

mAcc= {}
for key, value in full_data.items():
    mAcc[key] = round(value['mAcc']* 100, 2)

iter={}
for key, value in full_data.items():
    print(key)
    iter[key] =value['iter']


# mDice = {}
# for key, value in full_data.items():
#     mDice[key] = round(value['mDice'] * 100, 2)

# with open("/data/Code/timformer/gathered_data/mIoU.json", "w") as write_file:
#     json.dump(mIoU, write_file, indent=4)
# with open("/data/Code/timformer/gathered_data/class_mIoU.json", "w") as write_file2:
#     json.dump(class_mIoU, write_file2, indent=4)

with open("/data/Code/timformer/gathered_data/mAcc.json", "w") as write_file3:
    json.dump(mAcc, write_file3, indent=4)

# with open("/data/Code/timformer/gathered_data/mDice.json", "w") as write_file4:
#     json.dump(mDice, write_file4, indent=4)

with open("/data/Code/timformer/gathered_data/iters.json", "w") as write_file5:
    json.dump(iter, write_file5, indent=4)

