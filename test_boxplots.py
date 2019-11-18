# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 18/11/2019
"""
Post-processing program to perform statistical analysis and generate boxplots

Requirements
----------
Testing images must be located in the folder dir_test_img = "../MSK_Simplified_Knee/slices_png/test"
Testing segmentations (ground truth) must be located in the folder "../MSK_Simplified_Knee/slices_png_{}/test"
        with {} replaced by "bone_femur", "bone_tibia", "bone_patella", "cartilage_femur", "cartilage_tibia" or "cartilage_patella"
Model weights must be found as "models/model_{}.h5"
        with {} replaced by "bone_femur", "bone_tibia", "bone_patella", "cartilage_femur", "cartilage_tibia" or "cartilage_patella"

Return
----------
Two files are generated :
- "boxplot_all_dice.pdf"     : boxplot of the Dice coefficient for all the bones and cartilages (femur, tibia and patella)
- "boxplot_all_accuracy.pdf" : boxplot of the accuracy measure for all the bones and cartilages (femur, tibia and patella)
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, AveragePooling2D, Dense, Dropout, Flatten, Lambda, MaxPool2D, Conv2DTranspose, UpSampling2D, Concatenate, Add
from tensorflow.keras import regularizers, optimizers
from keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import skimage
from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import train_test_split

def combine_generator(gen1, gen2):
    while True:
        yield(gen1.next(), gen2.next())

# Defining loss function
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred, smooth=1):
    return 1-dice_coef(y_true, y_pred, smooth)

# Defining all components of the neural network
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

# Creating the network architecture
def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPool2D((2, 2)) (c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPool2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPool2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPool2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = Concatenate()([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = Concatenate()([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = Concatenate()([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = Concatenate(axis=3)([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
  
## Hyperparameters
seed = 1234567
input_size = 128
filters = 32
dropout = 0
batch_normalization = 1
batch_size = 8
steps_per_epoch = 250
epochs = 10
learning_rate = 1e-3
loss_function = dice_coef_loss
optimizer = optimizers.Adam(lr=learning_rate)
dir_test_img = "../MSK_Simplified_Knee/slices_png/test"

## Model
input_img = Input((input_size, input_size, 1), name='img')
model = get_unet(input_img, n_filters=filters, dropout=dropout, batchnorm=batch_normalization)

model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])
#model.summary()

# Image preprocessing
data_gen_args_img2 = dict(rescale=1./255)
image_datagen2 = image.ImageDataGenerator(**data_gen_args_img2)

#tab_experiments = ["bone_femur", "bone_tibia", "bone_patella", "cartilage_femur", "cartilage_tibia", "cartilage_patella"]
tab_experiments = ["bone_femur", "bone_tibia", "bone_patella", "cartilage_femur", "cartilage_patella"]
tab_dice_all = np.array([0.]*10*len(tab_experiments)).reshape((len(tab_experiments), 10))
tab_accuracy_all = np.array([0.]*10*len(tab_experiments)).reshape((len(tab_experiments), 10))

for i in range(0, len(tab_experiments)):
    # Importing data
    filename_weights = "models/model_{}.h5".format(tab_experiments[i])
    dir_test_mask = "../MSK_Simplified_Knee/slices_png_{}/test".format(tab_experiments[i])

    # Loading weights
    model.load_weights(filename_weights)

    image_generator_test = image_datagen2.flow_from_directory(dir_test_img, batch_size=batch_size, class_mode=None, target_size=(input_size, input_size), color_mode="grayscale", seed=seed)
    mask_generator_test = image_datagen2.flow_from_directory(dir_test_mask, batch_size=batch_size, class_mode=None, target_size=(input_size, input_size), color_mode="grayscale", seed=seed)
    test_generator = combine_generator(image_generator_test, mask_generator_test)

    # Generating segmentation and evaluate Dice coefficient and accuracy
    for j in range(0, 10):
        test_generator0 = next(test_generator)
        try:
            loss, accuracy = model.evaluate(test_generator0[0], test_generator0[1], batch_size=batch_size, verbose=1)
            tab_dice_all[i, j] = 1-loss
            tab_accuracy_all[i, j] = accuracy
        except:
            print("Problem with batch_size: {} != {}".format(len(test_generator0[0]), len(test_generator0[1])))
            j -= 1

    print("Experiment = ", tab_experiments[i])
    print("Dice = ", np.mean(tab_dice_all[i]), np.std(tab_dice_all[i]), np.median(tab_dice_all[i]), tab_dice_all[i])
    print("Accuracy = ", np.mean(tab_accuracy_all[i]), np.std(tab_accuracy_all[i]), np.median(tab_accuracy_all[i]), tab_accuracy_all[i])

# Plotting dice coefficient boxplots
red_diamond = dict(markerfacecolor='r', markeredgewidth=0, marker='D')
blue_circle = dict(markerfacecolor='b', markeredgewidth=0, marker='o')
fig, ax = plt.subplots()
ax.set_title('Statistical Tests: Dice coefficient')
ax.boxplot([tab_dice_all[i] for i in range(0, len(tab_experiments))], vert=False, flierprops=red_diamond, showmeans=True, meanprops=blue_circle)
plt.yticks([i+1 for i in range(0, len(tab_experiments))], tab_experiments)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("boxplot_all_dice.pdf", format="pdf")
plt.clf()

# Plotting accuracy boxplots
red_diamond = dict(markerfacecolor='r', markeredgewidth=0, marker='D')
blue_circle = dict(markerfacecolor='b', markeredgewidth=0, marker='o')
fig, ax = plt.subplots()
ax.set_title('Statistical Tests: Accuracy')
ax.boxplot([tab_accuracy_all[i] for i in range(0, len(tab_experiments))], vert=False, flierprops=red_diamond, showmeans=True, meanprops=blue_circle)
plt.yticks([i+1 for i in range(0, len(tab_experiments))], tab_experiments)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("boxplot_all_accuracy.pdf", format="pdf")
plt.clf()
