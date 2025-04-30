import unittest
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from unet import Encoder, DecoderUpBlock, SegmentationLayer


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, name, filters, size, **kwargs):
        super(AttentionBlock, self).__init__(name=name, **kwargs)
        self.conv_x = layers.Conv2D(filters, kernel_size=1, strides=2)
        self.conv_g = layers.Conv2D(filters, kernel_size=1)
        self.relu = layers.Activation('relu')
        self.sigmoid = layers.Activation('sigmoid')
        self.conv_ag = layers.Conv2D(filters, kernel_size=(1, 1))
        self.upsample = layers.UpSampling2D(size=(size, size), interpolation='bilinear')

    def call(self, inputs):
        x, g = inputs
        wx = self.conv_x(x)
        wg = self.conv_g(g)
        ag = layers.add([wx, wg])
        sigma_1 = self.relu(ag)
        ag = self.conv_ag(sigma_1)
        sigma_2 = self.sigmoid(ag)
        alpha = self.upsample(sigma_2)
        x_hat = layers.multiply([x, alpha])

        return x_hat

class DecoderAttention(tf.keras.layers.Layer):
    def __init__(self, name="decoder", batch_norm=True, dropout_rate=0.0, **kwargs):
        super(DecoderAttention, self).__init__(name=name, **kwargs)
        # Attention blocks
        self.attblock4 = AttentionBlock("decoder_attnblock4", filters=512, size=2)
        self.attblock3 = AttentionBlock("decoder_attnblock3", filters=256, size=2)
        self.attblock2 = AttentionBlock("decoder_attnblock2", filters=128, size=2)
        self.attblock1 = AttentionBlock("decoder_attnblock1", filters= 64, size=2)
        # Upsampling blocks
        self.upblock4 = DecoderUpBlock("decoder_upblock4", filters=512, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.upblock3 = DecoderUpBlock("decoder_upblock3", filters=256, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.upblock2 = DecoderUpBlock("decoder_upblock2", filters=128, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.upblock1 = DecoderUpBlock("decoder_upblock1", filters= 64, batch_norm=batch_norm, dropout_rate=dropout_rate)

    def call(self, inputs):
        pass

class UNetAttention(tf.keras.Model):
    def __init__(self, input_shape, num_classes=1, dropout_rate=0.0, batch_norm=True):
        super(UNetAttention, self).__init__()
        # Input layer
        self.inputs = layers.Input(input_shape, dtype=tf.float32)
        # Encoder layer
        self.encoder = Encoder(batch_norm=batch_norm, dropout_rate=dropout_rate)
        # Decoder with attention layer
        self.decoder = DecoderAttention(batch_norm=batch_norm, dropout_rate=dropout_rate)
        # Segmentation Block (1*1 convolutional layer)
        self.segmentation = SegmentationLayer()

    def call(self, inputs, **kwargs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return self.segmentation(x)

class TestDecoderAttention(unittest.TestCase):
    def test_attblock4(self):
        g = tf.random.uniform((1, 8, 8, 1024))
        x = tf.random.uniform((1, 16, 16, 512))
        model = DecoderAttention().attblock4
        self.assertEqual((1, 16, 16, 512), model([x, g]).shape)

    def test_attblock3(self):
        g = tf.random.uniform((1, 16, 16, 512))
        x = tf.random.uniform((1, 32, 32, 256))
        model = DecoderAttention().attblock3
        self.assertEqual((1, 32, 32, 256), model([x, g]).shape)

    def test_attblock2(self):
        g = tf.random.uniform((1, 32, 32, 256))
        x = tf.random.uniform((1, 64, 64, 128))
        model = DecoderAttention().attblock2
        self.assertEqual((1, 64, 64, 128), model([x, g]).shape)

    def test_attblock1(self):
        g = tf.random.uniform((1, 64, 64, 128))
        x = tf.random.uniform((1, 128, 128, 64))
        model = DecoderAttention().attblock1
        self.assertEqual((1, 128, 128, 64), model([x, g]).shape)