from __future__ import absolute_import
from __future__ import print_function
from keras.models import Model
from keras.layers import Lambda, add, core, concatenate, merge, Flatten, Dense, Input, Dropout, Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, AveragePooling2D,Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.utils import np_utils
import keras
from keras.applications import imagenet_utils
from keras import backend as K
import h5py
import tensorflow as tf
from keras.initializers import RandomNormal


def unet(input_shape=(None, None, 3)):

    inputs = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    #
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
   
    up1 = UpSampling2D(size=(2, 2))(conv4)
    up1 = concatenate([conv3,up1])
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    #
    up2 = UpSampling2D(size=(2, 2))(conv5)
    up2 = concatenate([conv2,up2])
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    #
    up3 = UpSampling2D(size=(2, 2))(conv6)
    up3 = concatenate([conv1,up3])
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    conv8 = Conv2D(1, (1, 1), activation='sigmoid')(conv7) 
    model = Model(inputs=inputs, outputs=conv8)
    return model
# 1: conv 2:Dropout 3: conv 4: maxpooling2D 5:conv 6: Dropout 7: conv 8: maxpooling 
# 9: conv 10: Dropout 11: conv 12: maxpooling 13:conv 14: Dropout 15:conv 16; upsampling
# 17: concate 18: conv 19: Dropout 20: conv 21:upsampling 22: concatenate 23: conv
# 24: Dropout 25: conv 26: upsampling 27: concatenate 28: conv 29: Dropout 30: conv 31: conv


def endecoder(input_shape=(None, None, 3)):

    inputs = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    #
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
   
    up1 = UpSampling2D(size=(2, 2))(conv4)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    #
    up2 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    #
    up3 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    conv8 = Conv2D(1, (1, 1), activation='sigmoid')(conv7) 
    model = Model(inputs=inputs, outputs=conv8)
    return model
# 1: conv 2: drop 3: conv 4: pooling 5: conv 6: drop 7: conv 8: pooling 9: conv 10: drop 11: conv
# 12: pooling 13: conv 14: drop 15: conv 16: upsamping 17: conv 18: drop 19: conv 20: upsampling
# 21: conv 22: drop 23: conv 24: upsamping 25: conv 26: drop 27: conv 28: conv


def DUNet(input_shape=(None, None, 3)):

    dilated_conv_kernel_initializer = RandomNormal(stddev=0.01)

    inputs = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    #
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
   
    dilate1 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(conv4)
    dilate2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilate1)
    dilate3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilate2)
    dilate4 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilate3)

    up1 = UpSampling2D(size=(2, 2))(dilate4)
    up1 = concatenate([conv3,up1])
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    #
    up2 = UpSampling2D(size=(2, 2))(conv5)
    up2 = concatenate([conv2,up2])
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    #
    up3 = UpSampling2D(size=(2, 2))(conv6)
    up3 = concatenate([conv1,up3])
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    dilate1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(conv7)
    dilate2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilate1)
    dilate3 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilate2)
    dilate4 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(dilate3)

   
    conv8 = Conv2D(1, (1, 1), activation='sigmoid')(dilate4) 
    model = Model(inputs=inputs, outputs=conv8)
    return model


