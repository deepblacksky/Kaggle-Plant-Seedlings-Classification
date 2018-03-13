"""Some process image function
"""

import os
from glob import glob
import random
import numpy as np
import cv2
import config


def center_crop(img, target_size):
    """新的resize方法, 支持等比裁剪, 中心裁剪
    # Arguments
        img: image array
        target_size: resize target size
    # Return
        resize image array
    """
    w, h = img.size
    if w > h:
        return img.crop(((w - h) // 2, 0, (w + h) // 2, h)).resize([target_size, target_size])
    else:
        return img.crop((0, (h - w) // 2, w, (h + w) // 2)).resize([target_size, target_size])


def random_crop(img, target_size):
    """新的resize方法, 支持等比裁剪, 随机裁剪
    # Arguments
        img: image array
        target_size: resize target size
    # Return
        resize image array
    """
    w, h = img.size
    l, t, r, b = 0, 0, w, h
    offset = abs(w - h)
    if w > h:
        l = random.randint(0, offset)
        r = l + h
    else:
        t = random.randint(0, offset)
        b = t + w
    img = img.crop((l, t, r, b)).resize([target_size, target_size])
    return img


def my_standardize(x):
    """用全局的mean和Var来标准化图片
        数据来自mean_and_var.py
        # Mean: is [0.34165438114647267, 0.30459320399278472, 0.23276843071882697]
        # Var: is [0.016283569476552281, 0.018187192886824563, 0.024280603503659182]
    """

    if x.dtype != np.float32:
        x = x.astype(np.float32)
    scale = 1 / 255.0
    x = x * scale

    if len(x.shape) == 3:
        # mean:
        x[:, :, 0] -= 0.34165438114647267
        x[:, :, 1] -= 0.30459320399278472
        x[:, :, 2] -= 0.23276843071882697

        # var
        x[:, :, 0] /= 0.016283569476552281
        x[:, :, 1] /= 0.018187192886824563
        x[:, :, 2] /= 0.024280603503659182
    elif len(x.shape) == 4:
        # mean:
        x[:, :, :, 0] -= 0.34165438114647267
        x[:, :, :, 1] -= 0.30459320399278472
        x[:, :, :, 2] -= 0.23276843071882697

        # var
        x[:, :, :, 0] /= 0.016283569476552281
        x[:, :, :, 1] /= 0.018187192886824563
        x[:, :, :, 2] /= 0.024280603503659182
    else:
        raise Exception('image x format error')
    return x


# Preprocessing the images

def create_mask_for_plant(image):
    """产生植物图片的掩码
    """

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def segment_plant(image):
    """按照mask分割植物图片
    """

    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask=mask)

    return output


def sharpen_image(image):
    """锐化图片
    """

    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)

    return image_sharp


def preprocessing_images():
    for sub_dir in os.listdir(config.TEST_DATA):
        new_sub_dir = os.path.join(os.path.join('./data_new/test/', sub_dir))
        if not os.path.exists(new_sub_dir):
            os.makedirs(new_sub_dir)
        sub_path = os.path.join(config.TEST_DATA, sub_dir)
        for i, image_path in enumerate(os.listdir(sub_path)):
            print(i)
            image = cv2.imread(os.path.join(sub_path, image_path))
            image_segmented = segment_plant(image)
            image_sharpen = sharpen_image(image_segmented)
            cv2.imwrite(os.path.join(new_sub_dir, image_path), image_sharpen)


if __name__ == '__main__':
    preprocessing_images()
