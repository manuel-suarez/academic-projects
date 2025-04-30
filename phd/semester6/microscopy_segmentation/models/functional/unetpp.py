from tensorflow import keras
from tensorflow.keras import layers
from .unet import convolutional_block

def UNetPP(input_shape=(256, 256, 1), batch_norm=True, dropout_rate=0.0):
    inputs = layers.Input(input_shape)
    # Backbone: j=0, i=0,1,2,3,4
    # X i=0, j=0
    x0_0 = convolutional_block(inputs, 64, batch_norm, dropout_rate)
    p0_0 = layers.MaxPooling2D(pool_size=(2,2))(x0_0)
    # X i=1, j=0
    x1_0 = convolutional_block(p0_0, 128, batch_norm, dropout_rate)
    p1_0 = layers.MaxPooling2D(pool_size=(2,2))(x1_0)
    # X i=2, j=0
    x2_0 = convolutional_block(p1_0, 256, batch_norm, dropout_rate)
    p2_0 = layers.MaxPooling2D(pool_size=(2,2))(x2_0)
    # X i=3, j=0
    x3_0 = convolutional_block(p2_0, 512, batch_norm, dropout_rate)
    p3_0 = layers.MaxPooling2D(pool_size=(2,2))(x3_0)
    # X i=4, j=0
    x4_0 = convolutional_block(p3_0, 1024, batch_norm, dropout_rate)
    # Upsampling + residual + convolutional
    # Intermediate layer j=1
    # X i=0, j=1
    u1_0 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(x1_0)
    c0_1 = layers.concatenate([u1_0, x0_0], axis=3)
    x0_1 = convolutional_block(c0_1, 64, batch_norm, dropout_rate)
    # X i=1, j=1
    u2_0 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(x2_0)
    c1_1 = layers.concatenate([u2_0, x1_0], axis=3)
    x1_1 = convolutional_block(c1_1, 128, batch_norm, dropout_rate)
    # X i=2, j=1
    u3_0 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(x3_0)
    c2_1 = layers.concatenate([u3_0, x2_0], axis=3)
    x2_1 = convolutional_block(c2_1, 256, batch_norm, dropout_rate)
    # X i=3, j=1
    u4_0 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(x4_0)
    c3_1 = layers.concatenate([u4_0, x3_0], axis=3)
    x3_1 = convolutional_block(c3_1, 512, batch_norm, dropout_rate)
    # Intermediate layer j=2
    # X i=0, j=2
    u1_1 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(x1_1)
    c0_2 = layers.concatenate([u1_1, x0_0, x0_1], axis=3)
    x0_2 = convolutional_block(c0_2, 64, batch_norm, dropout_rate)
    # X i=1, j=2
    u2_1 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(x2_1)
    c1_2 = layers.concatenate([u2_1, x1_0, x1_1], axis=3)
    x1_2 = convolutional_block(c1_2, 128, batch_norm, dropout_rate)
    # X i=2, j=2
    u3_1 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(x3_1)
    c2_2 = layers.concatenate([u3_1, x2_0, x2_1], axis=3)
    x2_2 = convolutional_block(c2_2, 256, batch_norm, dropout_rate)
    # Intermediate layer j=3
    # X i=0, j=3
    u1_2 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(x1_2)
    c0_3 = layers.concatenate([u1_2, x0_0, x0_1, x0_2], axis=3)
    x0_3 = convolutional_block(c0_3, 64, batch_norm, dropout_rate)
    # X i=1, j=3
    u2_2 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(x2_2)
    c1_3 = layers.concatenate([u2_2, x1_0, x1_1, x1_2], axis=3)
    x1_3 = convolutional_block(c1_3, 128, batch_norm, dropout_rate)
    # Final layer j=4
    # X i=0, j=4
    u1_3 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(x1_3)
    c0_4 = layers.concatenate([u1_3, x0_0, x0_1, x0_2, x0_3], axis=3)
    x0_4 = convolutional_block(c0_4, 64, batch_norm, dropout_rate)
    # Classification layer
    outputs = layers.Conv2D(1, kernel_size=(1,1))(x0_4)
    outputs = layers.BatchNormalization(axis=3)(outputs)
    outputs = layers.Activation('sigmoid')(outputs)

    model = keras.Model(inputs, outputs, name='UnetPP')
    return model

if __name__ == '__main__':
    model = UNetPP((256, 256, 1))
    print(model.summary())