from PIL import Image, ImageDraw, ImageFont
import os

MY_PATH = "/data/Code/timformer/save_results/eval/print_images"

def image_key(k):
    if k.startswith("U-Att-Large-"):
        att = int(k[12])
        new_k = "U-Att-Large-"
        i = 0
        for i in range(0, att):
            new_k = new_k + "T"
        for i in range(att, 6):
            new_k = new_k + "U"

    elif k.startswith("U-Att-Small-R-"):
        att = int(k[14])
        new_k = "U-Att-Small-"
        i = 0
        for i in range(0, 4 - att):
            new_k = new_k + "U"
        for i in range(4 - att, 4):
            new_k = new_k + "T"

    elif k.startswith("U-Att-Small-"):
        att = int(k[12])
        new_k = "U-Att-Small-"
        i = 0
        for i in range(0, att):
            new_k = new_k + "T"
        for i in range(att, 4):
            new_k = new_k + "U"


    elif k.startswith("U-Att-Full-R-"):
        att = int(k[13])
        new_k = "U-Att-Full-"
        i = 0
        for i in range(0, 6 - att):
            new_k = new_k + "U"
        for i in range(6 - att, 6):
            new_k = new_k + "T"

    elif k.startswith("U-Att-Full-"):
        att = int(k[11])
        new_k = "U-Att-Full-"
        i = 0
        for i in range(0, att):
            new_k = new_k + "T"
        for i in range(att, 6):
            new_k = new_k + "U"
    elif k.startswith("All-MLP (MiT-B0-Full)"):
        new_k = "All-MLP*"
    elif k.startswith("ground_truth"):
        new_k = "Ground Truth"
    else:
        new_k = k
    return new_k

for file in os.listdir(os.path.join(MY_PATH, "combined_images")):
    im = Image.open(os.path.join(MY_PATH, "combined_images", file))
    width, height = im.size
    I1 = ImageDraw.Draw(im)
    myFont = ImageFont.truetype('/data/Code/timformer/tools/cambria.ttc', 65)
    w, h = I1.textsize(image_key((file[5:-4])), font=myFont)
    I1.text(((width-w)/2, 700), image_key((file[5:-4])), font=myFont, fill=(255, 255, 255))
    im.save(os.path.join(MY_PATH, "combined_images", file), "PNG")