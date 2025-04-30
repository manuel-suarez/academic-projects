#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:42:23 2024

@author: masuareb
"""

import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
import unittest

#%% Define metrics and losses
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def IoU_coef(y_true, y_pred):
    T = K.flatten(y_true)
    P = K.flatten(y_pred)
    intersection = K.sum(T * P)
    return (intersection + 1.0) / (K.sum(T) + K.sum(T) - intersection + 1.0)

def IoU_loss(y_true, y_pred):
    return -IoU_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return - dice_coef(y_true, y_pred)

#%% Define auxiliary blocks
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, name, filters, batch_norm=True, dropout_rate=0.0, **kwargs):
        super(EncoderBlock, self).__init__(name=name, **kwargs)
        # Parameters
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        # Layers
        self.block1_conv1 = layers.Conv2D(filters, (3, 3), padding="same")
        self.batch_norm1  = layers.BatchNormalization(axis=3)
        self.block1_relu1 = layers.Activation("relu")
        self.block1_conv2 = layers.Conv2D(filters, (3, 3), padding="same")
        self.batch_norm2  = layers.BatchNormalization(axis=3)
        self.block1_relu2 = layers.Activation("relu")
        self.block1_drop  = layers.Dropout(dropout_rate)

    def call(self, inputs):
        x = self.block1_conv1(inputs)
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = self.block1_relu1(x)

        x = self.block1_conv2(x)
        if self.batch_norm:
            x = self.batch_norm2(x)
        x = self.block1_relu2(x)

        if self.dropout_rate > 0:
            x = self.block1_drop(x)

        return x

class DecoderUpBlock(tf.keras.layers.Layer):
    def __init__(self, name, filters, batch_norm=True, dropout_rate=0.0, **kwargs):
        super(DecoderUpBlock, self).__init__(name=name, **kwargs)
        # Parameters
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        # Layers
        self.upsample = layers.UpSampling2D(size=(2, 2), data_format="channels_last")
        # concatenation
        self.upblock4_conv1 = layers.Conv2D(filters, (3, 3), padding="same")
        self.batch_norm1 = layers.BatchNormalization(axis=3)
        self.upblock4_relu1 = layers.Activation("relu")

        self.upblock4_conv2 = layers.Conv2D(filters, (3, 3), padding="same")
        self.batch_norm2 = layers.BatchNormalization(axis=3)
        self.upblock4_relu2 = layers.Activation("relu")

        self.upblock4_dropout = layers.Dropout(dropout_rate)

    def call(self, inputs):
        x, skips = inputs
        x = self.upsample(x)
        x = layers.concatenate([x, skips])
        x = self.upblock4_conv1(x)
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = self.upblock4_relu1(x)

        x = self.upblock4_conv2(x)
        if self.batch_norm:
            x = self.batch_norm2(x)
        x = self.upblock4_relu2(x)

        if self.dropout_rate > 0:
            x = self.upblock4_dropout(x)

        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, name="encoder", batch_norm=True, dropout_rate=0.0, **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        # Blocks
        self.block1 = EncoderBlock("encoder_block1", filters=64, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.block2 = EncoderBlock("encoder_block2", filters=128, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.block3 = EncoderBlock("encoder_block3", filters=256, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.block4 = EncoderBlock("encoder_block4", filters=512, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.block5 = EncoderBlock("encoder_block5", filters=1024, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.pool   = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs):
        x1 = self.block1(inputs)
        x2 = self.block2(self.pool(x1))
        x3 = self.block3(self.pool(x2))
        x4 = self.block4(self.pool(x3))
        x5 = self.block5(self.pool(x4))

        return x5, [x1, x2, x3, x4]

class Decoder(tf.keras.layers.Layer):
    def __init__(self, name="decoder", batch_norm=True, dropout_rate=0.0, **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        # Blocks
        self.upblock4 = DecoderUpBlock("decoder_upblock4", filters=512, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.upblock3 = DecoderUpBlock("decoder_upblock3", filters=256, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.upblock2 = DecoderUpBlock("decoder_upblock2", filters=128, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.upblock1 = DecoderUpBlock("decoder_upblock1", filters=64, batch_norm=batch_norm, dropout_rate=dropout_rate)

    def call(self, inputs):
        x, skips = inputs
        x = self.upblock4([x, skips[3]])
        x = self.upblock3([x, skips[2]])
        x = self.upblock2([x, skips[1]])
        x = self.upblock1([x, skips[0]])

        return x

class SegmentationLayer(tf.keras.layers.Layer):
    def __init__(self, name="segmentation", **kwargs):
        super(SegmentationLayer, self).__init__(name=name, **kwargs)
        self.conv = layers.Conv2D(2, kernel_size=(1, 1))
        self.batch = layers.BatchNormalization(axis=3)
        self.activ = layers.Activation('sigmoid')

    def call(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.batch(outputs)
        outputs = self.activ(outputs)

        return outputs

#%% Define networks architectures
class UNet(tf.keras.Model):
    def __init__(self, input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
        super(UNet, self).__init__()
        # Input layer
        self.inputs = layers.Input(input_shape, dtype=tf.float32)
        # Encoder layer
        self.encoder = Encoder(batch_norm=batch_norm, dropout_rate=dropout_rate)
        # Decoder layer
        self.decoder = Decoder(batch_norm=batch_norm, dropout_rate=dropout_rate)
        # Segmentation Block (1*1 convolutional layer)
        self.segmentation = SegmentationLayer()

    def call(self, inputs, **kwargs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return self.segmentation(x)

#%% Define unit tests
class TestEncoder(unittest.TestCase):
    def test_block1(self):
        input = tf.random.uniform((1, 128, 128, 3))
        model = Encoder().block1
        self.assertEqual(model(input).shape, (1, 128, 128, 64))

    def test_block2(self):
        input = tf.random.uniform((1, 64, 64, 64))
        model = Encoder().block2
        self.assertEqual(model(input).shape, (1, 64, 64, 128))

    def test_block3(self):
        input = tf.random.uniform((1, 32, 32, 128))
        model = Encoder().block3
        self.assertEqual(model(input).shape, (1, 32, 32, 256))

    def test_block4(self):
        input = tf.random.uniform((1, 16, 16, 256))
        model = Encoder().block4
        self.assertEqual(model(input).shape, (1, 16, 16, 512))

    def test_block5(self):
        input = tf.random.uniform((1, 8, 8, 512))
        model = Encoder().block5
        self.assertEqual((1, 8, 8, 1024), model(input).shape)

    def test_blocks(self):
        input = tf.random.uniform((1, 128, 128, 3))
        maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        model = Encoder()
        output1 = model.block1(input)
        self.assertEqual((1, 128, 128, 64), output1.shape)
        output2 = model.block2(maxpool(output1))
        self.assertEqual((1, 64, 64, 128), output2.shape)
        output3 = model.block3(maxpool(output2))
        self.assertEqual((1, 32, 32, 256), output3.shape)
        output4 = model.block4(maxpool(output3))
        self.assertEqual((1, 16, 16, 512), output4.shape)
        output5 = model.block5(maxpool(output4))
        self.assertEqual((1, 8, 8, 1024), output5.shape)

    def test_encoder(self):
        input = tf.random.uniform((1, 128, 128, 3))
        model = Encoder()
        output, skips = model(input)
        self.assertEqual(4, len(skips))
        self.assertEqual((1, 128, 128,   64), skips[0].shape)
        self.assertEqual((1,  64,  64,  128), skips[1].shape)
        self.assertEqual((1,  32,  32,  256), skips[2].shape)
        self.assertEqual((1,  16,  16,  512), skips[3].shape)
        self.assertEqual((1,   8,   8, 1024), output.shape)

class TestDecoder(unittest.TestCase):
    def test_upblock4(self):
        input = tf.random.uniform((1, 8, 8, 1024))
        skip = tf.random.uniform((1, 16, 16, 512))
        model = Decoder().upblock4
        self.assertEqual((1, 16, 16, 512), model([input, skip]).shape)

    def test_upblock3(self):
        input = tf.random.uniform((1, 16, 16, 512))
        skip = tf.random.uniform((1, 32, 32, 256))
        model = Decoder().upblock3
        self.assertEqual((1, 32, 32, 256), model([input, skip]).shape)

    def test_upblock2(self):
        input = tf.random.uniform((1, 32, 32, 256))
        skip = tf.random.uniform((1, 64, 64, 128))
        model = Decoder().upblock2
        self.assertEqual((1, 64, 64, 128), model([input, skip]).shape)

    def test_upblock1(self):
        input = tf.random.uniform((1, 64, 64, 128))
        skip = tf.random.uniform((1, 128, 128, 64))
        model = Decoder().upblock1
        self.assertEqual((1, 128, 128, 64), model([input, skip]).shape)

    def test_blocks(self):
        inputs = tf.random.uniform((1, 128, 128, 3))
        encoder = Encoder()
        outputs, skips = encoder(inputs)
        decoder = Decoder()
        output1 = decoder.upblock4([outputs, skips[3]])
        self.assertEqual((1, 16, 16, 512), output1.shape)
        output2 = decoder.upblock3([output1, skips[2]])
        self.assertEqual((1, 32, 32, 256), output2.shape)
        output3 = decoder.upblock2([output2, skips[1]])
        self.assertEqual((1, 64, 64, 128), output3.shape)
        output4 = decoder.upblock1([output3, skips[0]])
        self.assertEqual((1, 128, 128, 64), output4.shape)

    def test_decoder(self):
        inputs = tf.random.uniform((1, 128, 128, 3))
        encoder = Encoder()
        encoder_outputs = encoder(inputs)
        decoder = Decoder()
        decoder_outputs = decoder(encoder_outputs)
        self.assertEqual((1, 128, 128, 64), decoder_outputs.shape)

class TestUNet(unittest.TestCase):
    def test_unet(self):
        data = tf.random.uniform((1, 128, 128, 3))
        model = UNet(input_shape=(128, 128, 3))
        self.assertEqual(model(data).shape, (1, 128, 128, 2))

#%% Main code
if __name__ == "__main__":
    unittest.main()