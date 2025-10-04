import os
import shutil
import argparse
from mmseg_test import testing_or_eval
import json
from change_keys import real_key
from PIL import Image
MY_PATH = "/data/Code/timformer/save_results/eval/print_images"
img_number = "0026"
img_name = "munster_00" + img_number + "_000019_leftImg8bit.png"


for dir in os.listdir(MY_PATH):
    if dir == "combined_images":
        continue
    shutil.copyfile(os.path.join(MY_PATH, dir, img_name), os.path.join(MY_PATH, "combined_images", img_number + "_" + dir + ".png"))


background = Image.open(os.path.join("/data/Code/timformer/data/cityscapes/gtFine/val/munster", "munster_00" + img_number + "_000019_gtFine_color.png"))
overlay = Image.open(os.path.join("/data/Code/timformer/data/cityscapes/leftImg8bit/val/munster", img_name))
width, heigth = background.size
overlay = overlay.resize((width//4, heigth//4))
overlay = overlay.resize((width, heigth))
overlay = overlay.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.5)
new_img.save(os.path.join(MY_PATH, "combined_images", img_number + "_ground_truth.png"),"PNG")

for file in os.listdir(os.path.join(MY_PATH, "combined_images")):
    if not file.startswith(img_number):
        continue
    im = Image.open(os.path.join(MY_PATH, "combined_images", file))
    width, height = im.size
    if not file.startswith("0026"):
        left = width / 4 + 80
        top = 10
        right = 3 * width / 4 - 80
        bottom = height - 150
    else:
        left = width / 4 + 300
        top = 200
        right = 3 * width / 4 - 80
        bottom = height - 180
    new_img = im.crop((left, top, right, bottom))
    new_img = new_img.resize(((width // 2 - 160),height - 160))
    new_img.save(os.path.join(MY_PATH, "combined_images", file), "PNG")




