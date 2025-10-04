import os
import shutil
MY_PATH = "/data/Code/timformer/training"
for dir in os.listdir(MY_PATH):
    check = False
    for file in os.listdir(os.path.join(MY_PATH, dir)):
        if file.startswith("iter_16000"):
            check = True
            break
    if not check:
        shutil.rmtree(os.path.join(MY_PATH, dir))
