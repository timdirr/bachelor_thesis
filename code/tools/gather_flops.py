import os
import shutil
import argparse
from get_flops import running_main
from change_keys import real_key
import json
MY_PATHS = [
    # "/data/Code/timformer/att_training",
    "/data/Code/timformer/add_att_training",
    "/data/Code/timformer/pretrain_weight_testing/training",
    "/data/Code/timformer/pretrain_weight_testing/training_full",
    # "/data/Code/timformer/pretrain_weight_testing/configs/test"
    # "/data/Code/timformer/flops_testing/"
]
flops = {}
param = {}
for MY_PATH in MY_PATHS:
    print(MY_PATH)
    for dir in os.listdir(MY_PATH):
        if real_key(dir) != "U-Att-Small-4" and real_key(dir) != "All-MLP" and real_key(dir) != "All-MLP-Full" and real_key(dir) != "U-Att-Full-R-4" and real_key(dir) != "U-Att-Large-6":
        # if real_key(dir) != "U-Att-Small-4":
           continue
        print(dir)
        conf_file = None
        for file in os.listdir(os.path.join(MY_PATH, dir)):
            if file.endswith(".py"):
                conf_file = file
        if conf_file:
            args = argparse.Namespace(config=os.path.join(MY_PATH, dir, conf_file),shape=[1024,1024])
            dir = real_key(dir)
            flops[dir]={}
            param[dir]={}
            flops[dir],param[dir] = running_main(args)
print(json.dumps(flops, sort_keys=True,indent=4))
print(json.dumps(param, sort_keys=True,indent=4))

with open("/data/Code/timformer/gathered_data/flops_1024.json", "w") as write_file1:
    json.dump(flops, write_file1, indent=4)
with open("/data/Code/timformer/gathered_data/params_1024.json", "w") as write_file2:
    json.dump(param, write_file2, indent=4)