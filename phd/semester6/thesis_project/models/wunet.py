#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:42:23 2024

@author: masuareb
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import models, layers, regularizers
from unittest import TestCase
from .common import conv_block, gating_signal, conv_bn_relu, convolutional_block, res_conv_block, attention_block  

def UNet(input_shape=(256, 256, 1), batch_norm=True, dropout_rate=0.0):
    sh1, sh2, sh3 = input_shape
    in_256 = layers.Input((sh1, sh2, sh3))
    in_128 = layers.Input((sh1//2, sh2//2, sh3))
    in_64 = layers.Input((sh1//4, sh2//4, sh3))
    in_32 = layers.Input((sh1//8, sh2//8, sh3))
    in_16 = layers.Input((sh1//16, sh2//16, sh3))

    # Encoder Path: 256 -> 128 -> 64 -> 32 -> 16 -> 8
    # Encoder Block1 : convolutional + max pool : 256 -> 128
    dn_128 = convolutional_block(in_256, 64, batch_norm, dropout_rate)
    on_128 = convolutional_block(in_128, 64, batch_norm, dropout_rate)
    pl_64 = layers.MaxPooling2D(pool_size=(2,2))(dn_128)
    pl_64 = layers.Add()([pl_64, on_128])
    # Encoder Block2 : convolutional + max pool : 128 -> 64
    dn_64 = convolutional_block(pl_64, 128, batch_norm, dropout_rate)
    on_64 = convolutional_block(in_64, 128, batch_norm, dropout_rate)
    pl_32 = layers.MaxPooling2D(pool_size=(2,2))(dn_64)
    pl_32 = layers.Add()([pl_32, on_64])
    # Encoder Block3 : convolutional + max pool : 64 -> 32
    dn_32 = convolutional_block(pl_32, 256, batch_norm, dropout_rate)
    on_32 = convolutional_block(in_32, 256, batch_norm, dropout_rate)
    pl_16 = layers.MaxPooling2D(pool_size=(2,2))(dn_32)
    pl_16 = layers.Add()([pl_16, on_32])
    # Encoder Block4 : convolutional + max pool : 32 -> 16
    dn_16 = convolutional_block(pl_16, 512, batch_norm, dropout_rate)
    on_16 = convolutional_block(in_16, 512, batch_norm, dropout_rate)
    pl_8 = layers.MaxPooling2D(pool_size=(2,2))(dn_16)
    pl_8 = layers.Add()([pl_8, on_16])
    # Bottleneck
    dn_8 = convolutional_block(pl_8, 1024, batch_norm, dropout_rate)
    # Decoder path: 8 -> 16 -> 32 -> 64 -> 128 -> 256
    # Decoder Block4 : upsample + concat + convolutional : 8 -> 16
    up_16 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(dn_8)
    ct_16 = layers.concatenate([up_16, dn_16], axis=3)
    dc_16 = convolutional_block(ct_16, 512, batch_norm, dropout_rate)
    # Decoder Block3 : upsample + concat + convolutional : 16 -> 32
    up_32 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(dc_16)
    ct_32 = layers.concatenate([up_32, dn_32], axis=3)
    dc_32 = convolutional_block(ct_32, 256, batch_norm, dropout_rate)
    # Decoder Block2 : upsample + concat + convolutional : 32 -> 64
    up_64 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(dc_32)
    ct_64 = layers.concatenate([up_64, dn_64], axis=3)
    dc_64 = convolutional_block(ct_64, 128, batch_norm, dropout_rate)
    # Decoder Block1 : upsample + concat + convolutional : 64 -> 128
    up_128 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(dc_64)
    ct_128 = layers.concatenate([up_128, dn_128], axis=3)
    dc_128 = convolutional_block(ct_128, 64, batch_norm, dropout_rate)
    # Classification layer
    outputs = layers.Conv2D(1, kernel_size=(1,1))(dc_128)
    outputs = layers.BatchNormalization(axis=3)(outputs)
    outputs = layers.Activation('sigmoid')(outputs)

    model = keras.Model([in_256, in_128, in_64, in_32, in_16], outputs, name='UNet')
    return model

def Attention_UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet,

    '''
    # network structure
    FILTER_NUM = 64  # number of basic filters for the first layer
    FILTER_SIZE = 3  # size of the convolutional filter
    UP_SAMP_SIZE = 2  # size of upsampling filters
    INPUT_W, INPUT_H, INPUT_C = input_shape

    inputs_128 = layers.Input(input_shape, dtype=tf.float32)
    # In this case we are using 128 as dimension base (not 256 because the variable naming starts at 128 instead of 256)
    inputs_64 = layers.Input((INPUT_W//2, INPUT_H//2, INPUT_C))
    inputs_32 = layers.Input((INPUT_W//4, INPUT_H//4, INPUT_C))
    inputs_16 = layers.Input((INPUT_W//8, INPUT_H//8, INPUT_C))
    inputs_8 = layers.Input((INPUT_W//16, INPUT_H//16, INPUT_C))
    
    
    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    outp_64 = conv_block(inputs_64, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm) 
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)
    pool_64 = layers.Add()([pool_64, outp_64]) # Concatenate wavelet convolution output
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    outp_32 = conv_block(inputs_32, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)
    pool_32 = layers.Add()([pool_32, outp_32])
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    outp_16 = conv_block(inputs_16, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm) 
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32) 
    pool_16 = layers.Add()([pool_16, outp_16])
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    outp_8 = conv_block(inputs_8, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2, 2))(conv_16)
    pool_8 = layers.Add()([pool_8, outp_8])
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8 * FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8 * FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4 * FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4 * FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2 * FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2 * FILTER_NUM)
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1, 1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  # Change to softmax for multichannel

    # Model integration
    model = models.Model([inputs_128, inputs_64, inputs_32, inputs_16, inputs_8], conv_final, name="Attention_UNet")
    return model


def Attention_ResUNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Rsidual UNet, with attention

    '''
    # network structure
    FILTER_NUM = 64  # number of basic filters for the first layer
    FILTER_SIZE = 3  # size of the convolutional filter
    UP_SAMP_SIZE = 2  # size of upsampling filters
    # input data
    INPUT_W, INPUT_H, INPUT_C = input_shape
    # dimension of the image depth
    axis = 3
    inputs_128 = layers.Input(input_shape, dtype=tf.float32)
    # In this case we are using 128 as dimension base (not 256 because the variable naming starts at 128 instead of 256)
    inputs_64 = layers.Input((INPUT_W//2, INPUT_H//2, INPUT_C))
    inputs_32 = layers.Input((INPUT_W//4, INPUT_H//4, INPUT_C))
    inputs_16 = layers.Input((INPUT_W//8, INPUT_H//8, INPUT_C))
    inputs_8 = layers.Input((INPUT_W//16, INPUT_H//16, INPUT_C))

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = res_conv_block(inputs_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    outp_64 = res_conv_block(inputs_64, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)
    pool_64 = layers.Add()([pool_64, outp_64])
    # DownRes 2
    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    outp_32 = res_conv_block(inputs_32, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)
    pool_32 = layers.Add()([pool_32, outp_32])
    # DownRes 3
    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    outp_16 = res_conv_block(inputs_16, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm) 
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)
    pool_16 = layers.Add()([pool_16, outp_16])
    # DownRes 4
    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    outp_8 = res_conv_block(inputs_8, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2, 2))(conv_16)
    pool_8 = layers.Add()([pool_8, outp_8])
    # DownRes 5, convolution only
    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8 * FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8 * FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=axis)
    up_conv_16 = res_conv_block(up_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4 * FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4 * FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=axis)
    up_conv_32 = res_conv_block(up_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2 * FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2 * FILTER_NUM)
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=axis)
    up_conv_64 = res_conv_block(up_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=axis)
    up_conv_128 = res_conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers

    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1, 1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=axis)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  # Change to softmax for multichannel

    # Model integration
    model = models.Model([inputs_128, inputs_64, inputs_32, inputs_16, inputs_8], conv_final, name="Attention_ResUNet")
    return model
if __name__ == '__main__':
    data256 = tf.random.uniform((1, 256, 256, 1))
    data128 = tf.random.uniform((1, 128, 128, 1))
    data64 = tf.random.uniform((1, 64, 64, 1))
    data32 = tf.random.uniform((1, 32, 32, 1))
    data16 = tf.random.uniform((1, 16, 16, 1))
    model = UNet(input_shape=(256, 256, 1))
    print(model([data256, data128, data64, data32, data16]).shape)
    print(model.summary())
