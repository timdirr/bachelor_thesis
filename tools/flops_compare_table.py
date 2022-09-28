import os
import shutil
import json
import math
import numpy as np
from contextlib import redirect_stdout

MY_PATH = "/data/Code/timformer/gathered_data/flops"
key_list = [
    # "U-Att-Full-0",
    # "U-Att-Full-1",
    # "U-Att-Full-2",
    # "U-Att-Full-3",
    # "U-Att-Full-4",
    # "U-Att-Full-5",
    # "U-Att-Full-6",
    # "U-Att-Full-R-5",
    "U-Att-Full-R-4",
    # "U-Att-Full-R-3",
    # "U-Att-Full-R-2",
    # "U-Att-Full-R-1",
    # "All-MLP (MiT-B0-Full)",
    "All-MLP-Full",
    # "U-Att-Small-0",
    # "U-Att-Small-1",
    # "U-Att-Small-2",
    # "U-Att-Small-3",
    "U-Att-Small-4",
    # "U-Att-Small-R-3",
    # "U-Att-Small-R-2",
    # "U-Att-Small-R-1",
    # "U-Att-Large-0",
    # "U-Att-Large-1",
    # "U-Att-Large-2",
    # "U-Att-Large-3",
    # "U-Att-Large-4",
    # "U-Att-Large-5",
    "U-Att-Large-6",
    "All-MLP",
]

def return_command(i):
    if i == 1:
        out = "one"
    elif i == 2:
        out = "two"
    elif i == 3:
        out = "three"
    elif i == 4:
        out = "four"
    elif i == 5:
        out = "five"
    elif i == 6:
        out = "six"
    elif i == 7:
        out = "seven"
    elif i == 8:
        out = "eigth"
    elif i == 9:
        out = "nine"
    else:
        out = "zero"
    # print("\\newcommand*{\\my" + out + "}{\makebox[\\xlength][c]{" + str(i) +"}}")
    return "\\my" + out + " "


def to_code_str(x):
    if x < 10 :
        first_d = int(x // 1)
        second_d = int(math.floor(x * 10) - first_d * 10)
        third_d = int(math.floor(x * 100) - first_d * 100 - second_d * 10)
        out = return_command(first_d) + "." + return_command(second_d) +  return_command(third_d)
    else:
        first_d = int(x// 10)
        second_d = int(math.floor(x) - first_d * 10)
        third_d = int(math.floor(x * 10) - first_d * 100 - second_d * 10)
        fourth_d = int(math.floor(x * 100) - first_d * 1000 - second_d * 100 - third_d * 10)
        out = return_command(first_d) + return_command(second_d) + "." + return_command(third_d) + return_command(fourth_d)
    return out



data = {}
for file in os.listdir(MY_PATH):
    f = open(os.path.join(MY_PATH, file), "r")
    key = file.replace("flops_", "").replace(".json", "")
    data[key] = {}
    data[key] = json.load(f)

new_data = {}
for key, value in data.items():
    for k, v in value.items():
        if k not in new_data:
            new_data[k] = {}
        new_data[k][key] = v

data = new_data


selected = {}
flops_keys = ['192', '256', '512', '768']
for key in key_list:
    print(key)
    selected[key] = {}
    for k in flops_keys:
        print(data[key][k])
        selected[key][k] = data[key][k]

# base_value = selected["All-MLP"]
# base_value = selected["All-MLP-Full"]

with open('save_results/flops_comparison.txt', 'w') as out_file:
    with redirect_stdout(out_file):
        for key, value in selected.items():
            print_str = key
            for k, v in value.items():
                print_str = print_str + " & " + to_code_str(v)
            print_str = print_str + " \\\\"
            print(print_str)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


# print(json.dumps(new_dict, sort_keys=True, indent=4))