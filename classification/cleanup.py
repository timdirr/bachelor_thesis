import os
import shutil
MY_PATH = "/data/Code/classification_task/"
for file in os.listdir(MY_PATH):
    if file.endswith(".tar"):
        os.remove(os.path.join(MY_PATH, file))
