"""Some Traing Config
"""

import os


TRAIN_DATA = './data/train/'
VAL_DATA = './data/val/'
TEST_DATA = './data/test/'

# TRAIN_DATA = './data_new/train/'
# VAL_DATA = './data_new/val/'
# TEST_DATA = './data_new/test/'

NUM_CLASSES = len(os.listdir(TRAIN_DATA))
# 
# IMAGE_SIZE = 350
IMAGE_SIZE = 299
EPOCHS = 100
lr = 1e-3
# BATCH_SIZE = 16
BATCH_SIZE = 32

TRAIN_NUM = 0
for sub_dir in os.listdir(TRAIN_DATA):
    sub_dir = os.path.join(TRAIN_DATA, sub_dir)
    TRAIN_NUM += len(os.listdir(sub_dir))
print(TRAIN_NUM)    #4174
VAL_NUM = 0
for sub_dir in os.listdir(VAL_DATA):
    sub_dir = os.path.join(VAL_DATA, sub_dir)
    VAL_NUM += len(os.listdir(sub_dir))
print(VAL_NUM)  #600
