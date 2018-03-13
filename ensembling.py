import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm

from keras import optimizers
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Average, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import MobileNet, Xception, InceptionResNetV2, InceptionV3, DenseNet121, DenseNet169
from keras.preprocessing.image import ImageDataGenerator

import config


def ensemble(models, input_tensor):

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    model = Model(inputs=input_tensor, outputs=y, name='ensemble')

    return model

# mobile
input_tensor = Input(shape=(299, 299, 3))

# base_model = MobileNet(input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
#                        input_tensor=input_tensor,
#                        include_top=False, weights='imagenet', pooling='avg')
# x = base_model.output
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.5)(x)
# predictions = Dense(12, activation='softmax')(x)
# # model = Model(inputs=base_model.input, outputs=predictions)
# model_mobilenet = Model(inputs=input_tensor, outputs=predictions)


# xception
base_model_2 = Xception(input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
                      input_tensor=input_tensor,
                      include_top=False, weights='imagenet', pooling='avg')
x_2 = base_model_2.output
# x = GlobalAveragePooling2D()(x)
x_2 = Dense(256, activation='relu')(x_2)
x_2 = Dropout(0.5)(x_2)
predictions_2 = Dense(12, activation='softmax')(x_2)
# model = Model(inputs=base_model.input, outputs=predictions)
model_xception = Model(inputs=input_tensor, outputs=predictions_2)

# InceptionResnetV2
base_model_3 = InceptionResNetV2(input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
                               input_tensor=input_tensor,
                               include_top=False, weights='imagenet', pooling='avg')
x_3 = base_model_3.output
# x = GlobalAveragePooling2D()(x)
x_3 = Dense(256, activation='relu')(x_3)
x_3 = Dropout(0.5)(x_3)
predictions_3 = Dense(12, activation='softmax')(x_3)
# model = Model(inputs=base_model.input, outputs=predictions)
model_inceptionresnetv2 = Model(inputs=input_tensor, outputs=predictions_3)

# InceptionV3
base_model_4 = InceptionV3(input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
                      input_tensor=input_tensor,
                      include_top=False, weights='imagenet', pooling='avg')
x_4 = base_model_4.output
# x = GlobalAveragePooling2D()(x)
x_4 = Dense(256, activation='relu')(x_4)
x_4 = Dropout(0.5)(x_4)
predictions_4 = Dense(12, activation='softmax')(x_4)
# model = Model(inputs=base_model.input, outputs=predictions)
model_inceptionv3 = Model(inputs=input_tensor, outputs=predictions_4)

#densenet121
# base_model_5 = DenseNet121(input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
#                       input_tensor=input_tensor,
#                       include_top=False, weights='imagenet', pooling='avg')
# x_5 = base_model_5.output
# x_5 = GlobalAveragePooling2D()(x_5)
# x_5 = Dense(256, activation='relu')(x_5)
# x_5 = Dropout(0.5)(x_5)
# predictions_5 = Dense(12, activation='softmax')(x_5)
# # model = Model(inputs=base_model.input, outputs=predictions)
# model_densenet121 = Model(inputs=input_tensor, outputs=predictions_5)

#densenet169
# base_model_6 = DenseNet169(input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
#                       input_tensor=input_tensor,
#                       include_top=False, weights='imagenet', pooling='avg')
# x_6 = base_model_6.output
# x_6 = GlobalAveragePooling2D()(x_6)
# x_6 = Dense(256, activation='relu')(x_6)
# x_6 = Dropout(0.5)(x_6)
# predictions_6 = Dense(12, activation='softmax')(x_6)
# # model = Model(inputs=base_model.input, outputs=predictions)
# model_densenet169 = Model(inputs=input_tensor, outputs=predictions_6)


# ensembling
# model_mobilenet.load_weights('./params/mobilenet_f/weights_mobilenet_f_n.h5')
model_xception.load_weights('./params/xception_f/weights_xception_f_299.h5')
model_inceptionresnetv2.load_weights(
    './params/inceptionresnetv2_f/weights_inceptionresnetv2_f_299.h5')
model_inceptionv3.load_weights('./params/inceptionV3_f/weights_inceptionV3_f_299.h5')
# model_densenet121.load_weights('./params/densenet121_f/weights_densenet121_f_n.h5')
# model_densenet169.load_weights('./params/densenet169_f/weights_densenet169_f_n.h5')
# models = [model_mobilenet, model_xception, model_inceptionresnetv2, model_inceptionv3, model_densenet121]
# models = [model_xception, model_densenet121]
models = [model_xception, model_inceptionv3, model_inceptionresnetv2]

ensemble_model = ensemble(models, input_tensor)

label_map = {'Fat Hen': 5, 'Sugar beet': 11, 'Loose Silky-bent': 6, 'Cleavers': 2, 'Shepherds Purse': 9,
             'Common wheat': 4, 'Charlock': 1, 'Common Chickweed': 3, 'Small-flowered Cranesbill': 10, 'Black-grass': 0,
             'Maize': 7, 'Scentless Mayweed': 8}
df_test = pd.read_csv('./data/sample_submission.csv')

# No augmentation on test dataset
test_datagen = ImageDataGenerator(rescale=1. / 255)
generator = test_datagen.flow_from_directory(
    config.TEST_DATA,
    target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode=None,  # No labels for test dataset
    shuffle=False)  # Do not shuffle data to keep labels in same order

p_test = ensemble_model.predict_generator(generator, verbose=1)

preds = []
for i in range(len(p_test)):
    pos = np.argmax(p_test[i])
    preds.append(list(label_map.keys())[list(label_map.values()).index(pos)])

df_test['species'] = preds
df_test.to_csv('./results/submission_ensemble_f_299.csv', index=False)
