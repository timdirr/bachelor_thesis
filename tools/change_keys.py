import json
import math

# x = {
#     "1_Attention": 0.6181772280024473,
#     "2_Attention": 0.6382841429430443,
#     "3_Attention": 0.6312960218438345,
#     "4_Attention": 0.630076011843644,
#     "5_Attention": 0.6382649105590066,
#     "No_Attention": 0.626839732739859,
#     "SegFormer_Decoder": 0.6335252668179959,
#     "6_Attention": 0.644568363719526,
#     "1_top_att": 0.6486849395525927,
#     "2_top_att": 0.6448165071164159,
#     "3_top_att": 0.6439037249444235,
#     "4_top_att": 0.651984998121502,
#     "baseline_only_4": 0.6208898592118433,
#     "5_top_att": 0.6419231523815108,
#     "pretrain_0_att": 0.6198903353100322,
#     "pretrain_baseline": 0.6295797148569325,
#     "pretrain_4_att": 0.6620570610149116,
#     "pretrain_3_att": 0.6536491471374832,
#     "pretrain_2_att": 0.6410115538688278,
#     "pretrain_1_att": 0.6220135847056105,
#     "pretrain_1_rev": 0.6509468941213005,
#     "pretrain_2_rev": 0.6536030015673536,
#     "pretrain_3_rev": 0.6569852940279469,
#     "0_Att_full": 0.6386060972563506,
#     "1_Att_full": 0.6348821621272194,
#     "2_Att_full": 0.6438274841637234,
#     "6_Att_full": 0.6643328184856117,
#     "5_Att_full": 0.6618276781011314,
#     "3_Att_full": 0.6509769529818583,
#     "4_Att_full": 0.6642276223821396
# }
#
#
# def return_command(i):
#     if i == 1:
#         out = "one"
#     elif i == 2:
#         out = "two"
#     elif i == 3:
#         out = "three"
#     elif i == 4:
#         out = "four"
#     elif i == 5:
#         out = "five"
#     elif i == 6:
#         out = "six"
#     elif i == 7:
#         out = "seven"
#     elif i == 8:
#         out = "eigth"
#     elif i == 9:
#         out = "nine"
#     else:
#         out = "zero"
#     # print("\\newcommand*{\\my" + out + "}{\makebox[\\xlength][c]{" + str(i) +"}}")
#     return "\\my" + out + " "
#
#
# new_x = {}
# for k, v in x.items():
def real_key(k):
    if k.endswith("full"):
        new_k = "U-Att-Large-" + k[0]
    elif k.endswith("rev"):
        new_k = "U-Att-Small-R-" + k[9]
    elif k.endswith("baseline"):
        new_k = "All-MLP"
    elif k.startswith("pretrain"):
        new_k = "U-Att-Small-" + k[9]
    elif k.startswith("baseline"):
        new_k = "All-MLP (MiT-B0-Full)"
    elif k.endswith("top_att"):
        new_k = "U-Att-Full-R-" + k[0]
    elif k.endswith("No_Attention"):
        new_k = "U-Att-Full-0"
    elif k.endswith("Attention"):
        new_k = "U-Att-Full-" + k[0]
    else:
        new_k = "All-MLP-Full"
    return new_k
#     new_x[new_k] = round(100 * v, 2)
#
#     # new_kx = new_k.replace("-", "").replace("0", "O").replace("1", "A").replace("2", "B").replace("3", "C").replace("4", "D").replace("5", "E").replace("6", "F").replace(" ", "").replace("(", "").replace(")", "")
#     out_str = ""
#     first_d = int(new_x[new_k] // 10)
#     second_d = int(math.floor(new_x[new_k]) - first_d * 10)
#     third_d = int(math.floor(new_x[new_k] * 10) - first_d * 100 - second_d * 10)
#     fourth_d = int(math.floor(new_x[new_k] * 100) - first_d * 1000 - second_d * 100 - third_d * 10)
#     # print(new_x[new_k])
#     # print(first_d)
#     # print(second_d)
#     # print(third_d)
#     # print(fourth_d)
#     print(new_k)
#     print(return_command(first_d) + return_command(second_d) + "." + return_command(third_d) + return_command(fourth_d))
#
#     # print("\\newcommand*{\\" + new_kx + "}{\makebox[\\xlength][c]{" + str(new_x[new_k]) +"}}")
#
# print(json.dumps(new_x, sort_keys=True, indent=4))