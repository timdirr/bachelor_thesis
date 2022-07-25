import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import scipy.ndimage

WHITE = [255, 255, 255]
BLUE = [0, 0, 255]
GREEN = [0, 255, 0]
RED = [255, 0, 0]

start = np.zeros([4, 4, 3])
start[1, 1] = RED

up_one = np.zeros([8, 8, 3])

for y_val in range(0, start.shape[0]):
    for x_val in range(0, start.shape[1]):
        new_x = x_val * 2
        new_y = y_val * 2
        x_min = max(0, new_x - 1)
        x_max = min(new_x + 1, up_one.shape[0])
        y_min = max(0, new_y - 1)
        y_max = min(new_y + 1, up_one.shape[0])
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                for i in range(0, 3):
                    up_one[x, y, i] = max(start[x_val, y_val, i], up_one[x, y, i])

up_one[2, 2] += GREEN

up_two = np.zeros([16, 16, 3])

for y_val in range(0, up_one.shape[0]):
    for x_val in range(0, up_one.shape[1]):
        new_x = x_val * 2
        new_y = y_val * 2
        x_min = max(0, new_x - 1)
        x_max = min(new_x + 1, up_two.shape[0])
        y_min = max(0, new_y - 1)
        y_max = min(new_y + 1, up_two.shape[0])
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                for i in range(0, 3):
                    up_two[x, y, i] = max(up_two[x, y, i], up_one[x_val, y_val, i])

up_two[4, 4] += BLUE

up_three = np.zeros([16, 16, 3])

for y_val in range(0, up_two.shape[0]):
    for x_val in range(0, up_two.shape[1]):
        new_x = x_val
        new_y = y_val
        x_min = max(0, new_x - 1)
        x_max = min(new_x + 1, up_three.shape[0] - 1)
        y_min = max(0, new_y - 1)
        y_max = min(new_y + 1, up_three.shape[0] - 1)
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                for i in range(0, 3):
                    up_three[x, y, i] = max(up_three[x, y, i], up_two[x_val, y_val, i])

# data = scipy.ndimage.zoom(up_two, 16, order=0)
# print(up_one)
# img = Image.fromarray(data, 'RGB')
# img.save('test.png')
# img.show()
up_three[4, 4] = BLUE

plt.imshow(up_three, interpolation='nearest')
plt.show()

start = np.zeros([4, 4, 3])
start[1, 1] = RED

up_one = np.zeros([8, 8, 3])

for y_val in range(0, start.shape[0]):
    for x_val in range(0, start.shape[1]):
        new_x = x_val * 2
        new_y = y_val * 2
        x_min = max(0, new_x - 1)
        x_max = min(new_x + 1, up_one.shape[0])
        y_min = max(0, new_y - 1)
        y_max = min(new_y + 1, up_one.shape[0])
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                for i in range(0, 3):
                    up_one[x, y, i] = max(start[x_val, y_val, i], up_one[x, y, i])

up_one[2, 2] += GREEN

up_two = np.zeros([16, 16, 3])

for y_val in range(0, up_one.shape[0]):
    for x_val in range(0, up_one.shape[1]):
        new_x = x_val * 2
        new_y = y_val * 2
        x_min = max(0, new_x - 1)
        x_max = min(new_x + 1, up_two.shape[0])
        y_min = max(0, new_y - 1)
        y_max = min(new_y + 1, up_two.shape[0])
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                for i in range(0, 3):
                    up_two[x, y, i] = max(up_two[x, y, i], up_one[x_val, y_val, i])

up_two[4, 4] += BLUE

up_three = np.zeros([16, 16, 3])

for y_val in range(0, up_two.shape[0]):
    for x_val in range(0, up_two.shape[1]):
            for i in range(0, 3):
                up_three[x_val, y_val, i] = max(up_three[x, y, i], up_two[x_val, y_val, i])

# data = scipy.ndimage.zoom(up_two, 16, order=0)
# print(up_one)
# img = Image.fromarray(data, 'RGB')
# img.save('test.png')
# img.show()
up_three[4, 4] = BLUE

plt.imshow(up_three, interpolation='nearest')
plt.show()