""" Data Folder
    ./data
        ./train
            ./class01
            ./class02
            ...
        ./val
            ./class01
            ./class02
            ...
        ./test
            ./test_sub
                xxxx.png
                xxxx.png
                ...
    Model: densenet
"""

import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm

from keras import optimizers
from keras.engine import Model
from keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

import image_generator
import config

df_test = pd.read_csv('./data/sample_submission.csv')

WEIGHTS_DIR = './params/densenet121_f'
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)

WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'weights_densenet121_f_299.h5')

# datagen = image_generator.ImageDataGenerator(horizontal_flip=True,
#                                              vertical_flip=True,
#                                              width_shift_range=0.1,
#                                              height_shift_range=0.1)

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    config.TRAIN_DATA,
    target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=True)

validate_datagen = ImageDataGenerator(rescale=1. / 255)

validate_generator = validate_datagen.flow_from_directory(
    config.VAL_DATA,
    target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=True)

callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1),
             ModelCheckpoint(WEIGHTS_PATH, monitor='val_loss',
                             save_best_only=True, verbose=1),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto',
                               epsilon=0.0001, cooldown=0, min_lr=0)]
#  TensorBoard(log_dir='./log/xception_v3', write_images=True)]

input_tensor = Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))

base_model = DenseNet121(input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
                      input_tensor=input_tensor,
                      include_top=False, weights='imagenet', pooling='avg')

# Only train Dense layers
# for layer in base_model.layers:
#     layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(12, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)
model = Model(inputs=input_tensor, outputs=predictions)

for layer in model.layers:
    layer.W_regularizer = l2(1e-2)
    layer.trainable = True

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(
    lr=config.lr), metrics=['accuracy'])

# print(model.summary())

# ----- TRAINING -----
model.fit_generator(train_generator,
                    steps_per_epoch=config.TRAIN_NUM // config.BATCH_SIZE,
                    validation_data=validate_generator,
                    validation_steps=config.VAL_NUM // config.BATCH_SIZE,
                    callbacks=callbacks,
                    epochs=config.EPOCHS,
                    verbose=1)

# ------ TESTING ------
label_map = validate_generator.class_indices
print(label_map)

# No augmentation on test dataset
test_datagen = ImageDataGenerator(rescale=1. / 255)
generator = test_datagen.flow_from_directory(
    config.TEST_DATA,
    target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode=None,  # No labels for test dataset
    shuffle=False)  # Do not shuffle data to keep labels in same order

if os.path.isfile(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)

p_test = model.predict_generator(generator, verbose=1)

preds = []
for i in range(len(p_test)):
    pos = np.argmax(p_test[i])
    preds.append(list(label_map.keys())[list(label_map.values()).index(pos)])

df_test['species'] = preds
df_test.to_csv('./results/submission_densenet121_f_299.csv', index=False)
