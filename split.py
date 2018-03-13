"""Split Data to Train and Val
"""
import os


ALL_DATA = './data/train/'
VAL_DATA = './data/val'
all_data = os.listdir(ALL_DATA)
for sub_dir in all_data:
    temp_dir = os.path.join(ALL_DATA, sub_dir)
    val_sub_dir = os.path.join(VAL_DATA, sub_dir)
    all_image = os.listdir(temp_dir)
    val_data = all_image[:50]
    if not os.path.exists(val_sub_dir):
        os.makedirs(val_sub_dir)
    for img in val_data:
        img_dir = os.path.join(temp_dir, img)
        img_val_dir = os.path.join(val_sub_dir, img)
        with open(img_dir, 'rb') as old:
            with open(img_val_dir, 'wb') as new:
                new.write(old.read())
                print('write finish % s' % img)
        os.remove(img_dir)
print('complete!')
