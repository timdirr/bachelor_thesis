import os
import shutil
import json
import math
import numpy as np
from contextlib import redirect_stdout
key_list = [
    # "U-Att-Full-R-1",
    # "U-Att-Full-R-2",
    # "U-Att-Full-R-3",
    # "U-Att-Full-R-4",
    # "U-Att-Full-R-5",
    # "U-Att-Full-0",
    # "U-Att-Full-1",
    # "U-Att-Full-2",
    # "U-Att-Full-3",
    # "U-Att-Full-4",
    # "U-Att-Full-5",
    # "U-Att-Full-6",
    # "All-MLP (MiT-B0-Full)",
    # "All-MLP-Full",
    # "U-Att-Small-0",
    # "U-Att-Small-1",
    # "U-Att-Small-2",
    # "U-Att-Small-3",
    # "U-Att-Small-4",
    # "U-Att-Small-R-3",
    # "U-Att-Small-R-2",
    # "U-Att-Small-R-1",
    "U-Att-Large-0",
    "U-Att-Large-1",
    "U-Att-Large-2",
    "U-Att-Large-3",
    "U-Att-Large-4",
    "U-Att-Large-5",
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
    first_d = int(x// 10)
    second_d = int(math.floor(x) - first_d * 10)
    third_d = int(math.floor(x * 10) - first_d * 100 - second_d * 10)
    fourth_d = int(math.floor(x * 100) - first_d * 1000 - second_d * 100 - third_d * 10)
    # print(x)
    # print(first_d)
    # print(second_d)
    # print(third_d)
    # print(fourth_d)
    out = return_command(first_d) + return_command(second_d) + "." + return_command(third_d) + return_command(fourth_d)
    # print(out)
    return out






with open("/data/Code/timformer/gathered_data/class_mIoU.json", "r") as f:
    data = json.load(f)

selected = {}
for key in key_list:
    selected[key] = data[key]

new_dict = {}
for key, value in selected.items():
    for k, v in value.items():
        if k not in new_dict:
            new_dict[k] = {}
        new_dict[k][key] = v


with open('save_results/my_table_large.txt', 'w') as out_file:
    with redirect_stdout(out_file):
        for key, values in new_dict.items():
            print_str = key
            highest = np.zeros(len(values))
            i = 0
            for k, v in values.items():
                highest[i] = v
                i = i + 1
            max_idx = np.argmax(highest)
            # print(key)
            # print(values)
            i = 0
            for k, v in values.items():
                # print(v)
                if i == max_idx:
                    print_str = print_str + " & " + "\\textbf{" + to_code_str(v) + "}"
                else:
                    print_str = print_str + " & " + to_code_str(v)
                i = i + 1
            print_str = print_str + " \\\\"
            print(print_str)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


# print(json.dumps(new_dict, sort_keys=True, indent=4))