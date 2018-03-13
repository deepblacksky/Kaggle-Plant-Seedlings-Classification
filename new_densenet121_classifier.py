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
    Model: Densenet121
"""

import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm

from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

# lr decay schedule

TRAIN_NUM = 4174
VAL_NUM = 600
BATCH_SIZE = 16

df_test = pd.read_csv('./data/sample_submission.csv')

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-4
    if epoch > 80:
        lr *= 1e-1
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 140:
        lr *= 1e-3
    print('Learning rate: ', lr)
    return lr


# data generator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
    './data/train/',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)

val_generator = val_datagen.flow_from_directory(
    './data/val/',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True)

# pretrain dense layer
# to avoid large gradient to destroy the pretrained model
# build model
tensorboard = TensorBoard('./log/new_densenet121')

basic_model = DenseNet121(input_shape=(224, 224, 3),
                       include_top=False, weights='imagenet', pooling='avg')

for layer in basic_model.layers:
    layer.trainable = False

# build top
x = basic_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
pred = Dense(12, activation='softmax')(x)

model=Model(inputs=basic_model.input, outputs=pred)
model.compile(optimizer=optimizers.RMSprop(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator,
                    steps_per_epoch=(TRAIN_NUM // BATCH_SIZE),
                    validation_data=val_generator,
                    validation_steps=(VAL_NUM // BATCH_SIZE),
                    callbacks=[tensorboard],
                    epochs=40,
                    verbose=1)

# train with whole model
for layer in model.layers:
    layer.W_regularizer=l2(1e-2)
    layer.trainable=True

model.compile(optimizer=optimizers.RMSprop(lr_schedule(0)),
              loss='categorical_crossentropy', metrics=['accuracy'])

# call backs
WEIGHTS_DIR='./params/new_densenet/'
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)

WEIGHTS_PATH=os.path.join(WEIGHTS_DIR, 'weights_new_densenet121.h5')
checkpointer=ModelCheckpoint(filepath=WEIGHTS_PATH, verbose=1,
                               save_best_only=True)


lr=LearningRateScheduler(lr_schedule)

model.fit_generator(train_generator,
                    steps_per_epoch=(TRAIN_NUM // BATCH_SIZE),
                    validation_data=val_generator,
                    validation_steps=(VAL_NUM // BATCH_SIZE),
                    epochs=150,
                    callbacks=[checkpointer, tensorboard, lr],
                    initial_epoch=40,
                    verbose=1)

# ------ TESTING ------
label_map = val_generator.class_indices
print(label_map)

# save model
model.save('./model/new_densenet121.h5')

# No augmentation on test dataset
test_datagen = ImageDataGenerator(rescale=1. / 255)
generator = test_datagen.flow_from_directory(
    './data/test/',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode=None,  # No labels for test dataset
    shuffle=False)  # Do not shuffle data to keep labels in same order

if os.path.isfile(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)

p_test_1 = model.predict_generator(generator, verbose=1)

# np.save('densenet.npy', p_test_1)

test_datagen_2 = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

generator_2 = test_datagen_2.flow_from_directory(
    './data/test/',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode=None,  # No labels for test dataset
    shuffle=False)  # Do not shuffle data to keep labels in same order

p_test_2 = model.predict_generator(generator_2, verbose=1)

p_test = (p_test_1 + p_test_2) / 2.0

np.save('densenet_2.npy', p_test)

preds = []
for i in range(len(p_test)):
    pos = np.argmax(p_test[i])
    preds.append(list(label_map.keys())[list(label_map.values()).index(pos)])

df_test['species'] = preds
df_test.to_csv('./results/submission_new_densenet121_2.csv', index=False)

