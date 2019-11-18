# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 18/11/2019
"""
Main program to train and test the U-net architecture

Requirements
----------
Training images must be located in the folder dir_train_img = "MSK_Simplified_Knee/slices_png/train"
Training segmentations (ground truth) must be located in the folder dir_train_mask = "MSK_Simplified_Knee/slices_png_mask/train"
Testing images must be located in the folder dir_test_img = "MSK_Simplified_Knee/slices_png/test"
Testing segmentations (ground truth) must be located in the folder dir_test_mask = "MSK_Simplified_Knee/slices_png_mask/test"

Return
----------
Several files are generated :
- "data_augmentation.pdf" : plot of 2 input images and their segmentation after the data augmentation process
- "model.h5"              : the model weights for post-training segmentation
- "learning_curve.pdf"    : plot of the curve containing the loss functions
- "train_prediction.pdf"  : plot a comparison of the ground truth and the generated segmentation for an image in the training set
- "test_prediction.pdf"   : plot a comparison of the ground truth and the generated segmentation for an image in the testing set
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
  
# Hyperparameters
seed = 123456
input_size = 128
filters = 32
dropout = 0
batch_normalization = 1
batch_size = 8
steps_per_epoch = 250
epochs = 10
learning_rate = 1e-3
#loss_function = "binary_crossentropy"
loss_function = dice_coef_loss
optimizer = optimizers.Adam(lr=learning_rate)
#optimizer = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=True)
filename_weights = "model.h5"

# Importing data
dir_train_img = "MSK_Simplified_Knee/slices_png/train"
dir_train_mask = "MSK_Simplified_Knee/slices_png_mask/train"

dir_test_img = "MSK_Simplified_Knee/slices_png/test"
dir_test_mask = "MSK_Simplified_Knee/slices_png_mask/test"

# Data augmentation
data_gen_args_img = dict(rescale=1./255,
                     rotation_range=5,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     zoom_range=0,
                     #horizontal_flip=True,
                     fill_mode="constant",
                     cval=0,
                     validation_split=0.15)

image_datagen = image.ImageDataGenerator(**data_gen_args_img)

image_generator_train = image_datagen.flow_from_directory(dir_train_img, batch_size=batch_size, class_mode=None, target_size=(input_size, input_size), color_mode="grayscale", seed=seed, subset="training")
mask_generator_train = image_datagen.flow_from_directory(dir_train_mask, batch_size=batch_size, class_mode=None, target_size=(input_size, input_size), color_mode="grayscale", seed=seed, subset="training")
train_generator = combine_generator(image_generator_train, mask_generator_train)

image_generator_val = image_datagen.flow_from_directory(dir_train_img, batch_size=batch_size, class_mode=None, target_size=(input_size, input_size), color_mode="grayscale", seed=seed, subset="validation")
mask_generator_val = image_datagen.flow_from_directory(dir_train_mask, batch_size=batch_size, class_mode=None, target_size=(input_size, input_size), color_mode="grayscale", seed=seed, subset="validation")
val_generator = combine_generator(image_generator_val, mask_generator_val)

# Plotting data augmentation
train_generator0 = next(train_generator)
plt.figure(figsize=(10, 10))

plt.subplot('221')
plt.imshow(train_generator0[0][0, :, :, 0], cmap='gray')
plt.title('Input')
plt.colorbar()

plt.subplot('222')
plt.imshow(train_generator0[1][0, :, :, 0].squeeze())
plt.title('Ground Truth')
plt.colorbar()

plt.subplot('223')
plt.imshow(train_generator0[0][1, :, :, 0], cmap='gray')
plt.title('Input')
plt.colorbar()

plt.subplot('224')
plt.imshow(train_generator0[1][1, :, :, 0].squeeze())
plt.title('Ground Truth')
plt.colorbar()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("data_augmentation.pdf", format="pdf")
plt.clf()

# Model
input_img = Input((input_size, input_size, 1), name='img')
model = get_unet(input_img, n_filters=filters, dropout=dropout, batchnorm=batch_normalization)

model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])
model.summary()

callbacks = [
    EarlyStopping(patience=20, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-6, verbose=1),
    ModelCheckpoint(filename_weights, verbose=1, save_best_only=True, save_weights_only=True)
    #TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)
]

# Training
tps = time.time()
results = model.fit_generator(
                            train_generator,
                            validation_data=val_generator,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=steps_per_epoch,
                            epochs=epochs, callbacks=callbacks, verbose=1)
print("Execution time = ", time.time()-tps)

# Plotting learning curve
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="o", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend();

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("learning_curve.pdf", format="pdf")
plt.clf()

def plot_sample(X, y, preds, binary_preds, filename, ix=None):
    if ix is None:
        ix = np.random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(X[ix, ..., 0], cmap='gray')
    if has_mask:
        ax[0, 0].contour(y[ix].squeeze(), colors='r', levels=[0.5])
    ax[0, 0].set_title('Input')

    ax[0, 1].imshow(y[ix].squeeze())
    ax[0, 1].set_title('Ground Truth')

    ax[1, 0].imshow(preds[ix].squeeze(), vmin=0, vmax=1, cmap="jet")
    if has_mask:
        ax[1, 0].contour(y[ix].squeeze(), colors='w', levels=[0.5])
    ax[1, 0].set_title('Predicted Segmentation')
    
    ax[1, 1].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[1, 1].contour(y[ix].squeeze(), colors='r', levels=[0.5])
    ax[1, 1].set_title('Binary Predicted Segmentation');

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, format="pdf")
    plt.clf()


# Image preprocessing for final tests
data_gen_args_img2 = dict(rescale=1./255)

image_datagen2 = image.ImageDataGenerator(**data_gen_args_img2)

image_generator_train = image_datagen2.flow_from_directory(dir_train_img, batch_size=batch_size, class_mode=None, target_size=(input_size, input_size), color_mode="grayscale", seed=seed)
mask_generator_train = image_datagen2.flow_from_directory(dir_train_mask, batch_size=batch_size, class_mode=None, target_size=(input_size, input_size), color_mode="grayscale", seed=seed)
train_generator = combine_generator(image_generator_train, mask_generator_train)

image_generator_test = image_datagen2.flow_from_directory(dir_test_img, batch_size=batch_size, class_mode=None, target_size=(input_size, input_size), color_mode="grayscale", seed=seed)
mask_generator_test = image_datagen2.flow_from_directory(dir_test_mask, batch_size=batch_size, class_mode=None, target_size=(input_size, input_size), color_mode="grayscale", seed=seed)
test_generator = combine_generator(image_generator_test, mask_generator_test)

train_generator0 = next(train_generator)
test_generator0 = next(test_generator)

# Loading weights
model.load_weights(filename_weights)

# Generating segmentation
pred_train = model.predict(train_generator0[0], batch_size=batch_size, verbose=1)
pred_test = model.predict(test_generator0[0], batch_size=batch_size, verbose=1)

# Threshold predictions
pred_train_bin = (pred_train > 0.5).astype(np.uint8)
pred_test_bin = (pred_test > 0.5).astype(np.uint8)

# Plotting comparison between ground truth and generated segmentation for the training and the testing sets
print(model.evaluate(train_generator0[0], train_generator0[1], batch_size=batch_size, verbose=1))
plot_sample(train_generator0[0], train_generator0[1], pred_train, pred_train_bin, "train_prediction.pdf")

print(model.evaluate(test_generator0[0], test_generator0[1], batch_size=batch_size, verbose=1))
plot_sample(test_generator0[0], test_generator0[1], pred_test, pred_test_bin, "test_prediction.pdf")
