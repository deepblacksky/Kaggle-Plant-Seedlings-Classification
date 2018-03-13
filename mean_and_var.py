"""Compute all data Mean and Var
"""
from PIL import Image
import numpy as np
import config


def get_files(dir):
    import os
    if not os.path.exists(dir):
        return []
    if os.path.isfile(dir):
        return [dir]
    result = []
    for subdir in os.listdir(dir):
        sub_path = os.path.join(dir, subdir)
        result += get_files(sub_path)
    return result


r = 0
g = 0
b = 0
r_2 = 0
g_2 = 0
b_2 = 0

total = 0
files = get_files(config.TRAIN_DATA)
count = len(files)

for i, image_path in enumerate(files):
    print('Process:%d/%d' % (i, count))
    img = Image.open(image_path)
    img = np.asarray(img)
    img = img.astype('float32') / 255.0
    total += img.shape[0] * img.shape[1]

    r += img[:, :, 0].sum()
    g += img[:, :, 1].sum()
    b += img[:, :, 2].sum()

    r_2 += (img[:, :, 0] ** 2).sum()
    g_2 += (img[:, :, 1] ** 2).sum()
    b_2 += (img[:, :, 2] ** 2).sum()

r_mean = r / total
g_mean = g / total
b_mean = b / total

r_var = r_2 / total - r_mean ** 2
g_var = g_2 / total - g_mean ** 2
b_var = b_2 / total - b_mean ** 2

print("Mean: is %s" % ([r_mean, g_mean, b_mean]))
print("Var: is %s" % ([r_var, g_var, b_var]))
# Mean: is [0.34165438114647267, 0.30459320399278472, 0.23276843071882697]
# Var: is [0.016283569476552281, 0.018187192886824563, 0.024280603503659182]
