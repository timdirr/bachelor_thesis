import os
import shutil
import argparse
from mmseg_test import testing_or_eval
import json
import matplotlib.pyplot as plt
import numpy as np
MY_PATH = "/data/Code/timformer/save_results"
MY_KEYS = ['aAcc', 'mIoU', 'mAcc', 'IoU.road', 'IoU.sidewalk', 'IoU.building', 'IoU.wall', 'IoU.fence', 'IoU.pole', 'IoU.traffic light', 'IoU.traffic sign', 'IoU.vegetation', 'IoU.terrain',
'IoU.sky', 'IoU.person', 'IoU.rider', 'IoU.car', 'IoU.truck', 'IoU.bus', 'IoU.train', 'IoU.motorcycle', 'IoU.bicycle', 'Acc.road', 'Acc.sidewalk', 'Acc.building', 'Acc.wall', 'Acc.fence', 'Acc.pole', 'Acc.traffic light', 'Acc.traffic sign', 'Acc.vegetation', 'Acc.terrain', 'Acc.sky', 'Acc.person', 'Acc.rider', 'Acc.car', 'Acc.truck', 'Acc.bus', 'Acc.train',
'Acc.motorcycle', 'Acc.bicycle']

my_dict = {}
i = False
for group in os.listdir(MY_PATH):
    if not group.startswith('group'):
        continue
    group_dict = {}
    for dir in os.listdir(os.path.join(MY_PATH, group)):
        for file in os.listdir(os.path.join(MY_PATH, group, dir)):
            group_dict[dir] = json.load(open(os.path.join(MY_PATH, group, dir, file), 'r'))['metric']
    my_dict[group] = group_dict

my_plot_dict = {}
for group_key, group_value in my_dict.items():
    group_plot_dict = {}
    for k in MY_KEYS:
        group_plot_dict[k] = {}
    for member_key, member_value in group_value.items():
        for key, value in member_value.items():
            group_plot_dict[key][member_key] = value
    my_plot_dict[group_key] = group_plot_dict


for group_key, group_value in my_plot_dict.items():
    for metric_key, metric_value in group_value.items():
        xAxis = [key for key, value in metric_value.items()]
        yAxis = [value for key, value in metric_value.items()]
        fig, ax = plt.subplots()
        bar_x = np.arange(1, len(yAxis) + 1)
        bar_plot = plt.bar(bar_x, yAxis, tick_label=None)
        ax.set_xticklabels([])
        ymin = np.amin(yAxis)
        ymax = np.amax(yAxis)
        ydif = ymax - ymin
        plt.ylim([ymin - 0.6 * ydif, ymax + 0.1 * ydif])
        for idx, rect in enumerate(bar_plot):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.,(ymin - 0.5 * ydif),
                    xAxis[idx],
                    ha='center', va='bottom', rotation=90)
        plt.title(metric_key)
        if not os.path.exists(os.path.join(MY_PATH, 'plots', group_key)):
            os.mkdir(os.path.join(MY_PATH, 'plots', group_key))
        plt.savefig(os.path.join(MY_PATH, 'plots', group_key, ''.join([metric_key, '.png'])))
        plt.figure()



# my_dict = {}
# i = False
# for group in os.listdir(MY_PATH):
#     if not group.startswith('group'):
#         continue
#     for dir in os.listdir(os.path.join(MY_PATH, group)):
#         for file in os.listdir(os.path.join(MY_PATH, group, dir)):
#             my_dict[dir] = json.load(open(os.path.join(MY_PATH, group, dir, file), 'r'))['metric']
#
#
# my_plot_dict = {}
# for k in MY_KEYS:
#     my_plot_dict[k] = {}
# for member_key, member_value in my_dict.items():
#     for key, value in member_value.items():
#         my_plot_dict[key][member_key] = value
#
# for metric_key, metric_value in my_plot_dict.items():
#     xAxis = [key for key, value in metric_value.items()]
#     yAxis = [value for key, value in metric_value.items()]
#
#     fig, ax = plt.subplots()
#
#     bar_x = np.arange(1, len(yAxis) + 1)
#
#     bar_plot = plt.bar(bar_x, yAxis, tick_label=None)
#
#     ymin = np.amin(yAxis)
#     ymax = np.amax(yAxis)
#     ydif = ymax - ymin
#     plt.ylim([ymin - 0.6 * ydif, ymax + 0.1 * ydif])
#
#     for idx, rect in enumerate(bar_plot):
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width() / 2.,(ymin - 0.5 * ydif),
#                 xAxis[idx],
#                 ha='center', va='bottom', rotation=90)
#
#
#
#
#     plt.title(metric_key)
#
#     plt.savefig(os.path.join(MY_PATH, 'plots', ''.join([metric_key, '.png'])))
#     plt.figure()


print(json.dumps(my_plot_dict, sort_keys=True, indent=4))

