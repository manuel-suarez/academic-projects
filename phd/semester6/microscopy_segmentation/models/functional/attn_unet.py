from tensorflow import keras
from tensorflow.keras import layers
from unet import convolutional_block

def AttentionUNet(input_shape=(256, 256, 1), batch_norm=True, dropout_rate=0.0):
    inputs = layers.Input(input_shape)
    # Encoder Path: 256 -> 128 -> 64 -> 32 -> 16 -> 8
    # Encoder Block1 : convolutional + max pool : 256 -> 128
    x_128 = convolutional_block(inputs, 64, batch_norm, dropout_rate)
    p_64 = layers.MaxPooling2D(pool_size=(2,2))(x_128)
    # Encoder Block2 : convolutional + max pool : 128 -> 64
    x_64 = convolutional_block(p_64, 128, batch_norm, dropout_rate)
    p_32 = layers.MaxPooling2D(pool_size=(2,2))(x_64)
    # Encoder Block3 : convolutional + max pool : 64 -> 32
    x_32 = convolutional_block(p_32, 256, batch_norm, dropout_rate)
    p_16 = layers.MaxPooling2D(pool_size=(2,2))(x_32)
    # Encoder Block4 : convolutional + max pool : 32 -> 16
    x_16 = convolutional_block(p_16, 512, batch_norm, dropout_rate)
    pl_8 = layers.MaxPooling2D(pool_size=(2,2))(x_16)
    # Bottleneck
    g_8 = convolutional_block(pl_8, 1024, batch_norm, dropout_rate)
    # Decoder path: 8 -> 16 -> 32 -> 64 -> 128 -> 256
    # Decoder Block4 : upsample + concat + convolutional : 8 -> 16
    theta_x_16 = layers.Conv2D(filters=512, kernel_size=(1,1), strides=(2,2), padding='same')(x_16)
    phi_g_16 = layers.Conv2D(filters=512, kernel_size=(1,1), strides=(1,1), padding='same')(g_8)
    concat_xg_16 = layers.add([phi_g_16, theta_x_16])
    act_xg_16 = layers.Activation('relu')(concat_xg_16)
    psi_16 = layers.Conv2D(filters=1, kernel_size=(1,1), padding='same')(act_xg_16)
    sigmoid_xg_16 = layers.Activation('sigmoid')(psi_16)
    upsample_psi_16 = layers.UpSampling2D(size=(2,2), data_format='channels_last')(sigmoid_xg_16)
    y_16 = layers.multiply([x_16, upsample_psi_16])
    g_16 = convolutional_block(y_16, 512, batch_norm, dropout_rate)
    # Decoder Block3 : upsample + concat + convolutional : 16 -> 32
    theta_x_32 = layers.Conv2D(filters=256, kernel_size=(1,1), strides=(2,2), padding='same')(x_32)
    phi_g_32 = layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding='same')(g_16)
    concat_xg_32 = layers.add([phi_g_32, theta_x_32])
    act_xg_32 = layers.Activation('relu')(concat_xg_32)
    psi_32 = layers.Conv2D(filters=1, kernel_size=(1,1), padding='same')(act_xg_32)
    sigmoid_xg_32 = layers.Activation('sigmoid')(psi_32)
    upsample_psi_32 = layers.UpSampling2D(size=(2,2), data_format='channels_last')(sigmoid_xg_32)
    y_32 = layers.multiply([x_32, upsample_psi_32])
    g_32 = convolutional_block(y_32, 256, batch_norm, dropout_rate)
    # Decoder Block2 : upsample + concat + convolutional : 32 -> 64
    theta_x_64 = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='same')(x_64)
    phi_g_64 = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(g_32)
    concat_xg_64 = layers.add([phi_g_64, theta_x_64])
    act_xg_64 = layers.Activation('relu')(concat_xg_64)
    psi_64 = layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same')(act_xg_64)
    sigmoid_xg_64 = layers.Activation('sigmoid')(psi_64)
    upsample_psi_64 = layers.UpSampling2D(size=(2, 2), data_format='channels_last')(sigmoid_xg_64)
    y_64 = layers.multiply([x_64, upsample_psi_64])
    g_64 = convolutional_block(y_64, 256, batch_norm, dropout_rate)
    # Decoder Block1 : upsample + concat + convolutional : 64 -> 128
    theta_x_128 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same')(x_128)
    phi_g_128 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(g_64)
    concat_xg_128 = layers.add([phi_g_128, theta_x_128])
    act_xg_128 = layers.Activation('relu')(concat_xg_128)
    psi_128 = layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same')(act_xg_128)
    sigmoid_xg_128 = layers.Activation('sigmoid')(psi_128)
    upsample_psi_128 = layers.UpSampling2D(size=(2, 2), data_format='channels_last')(sigmoid_xg_128)
    y_128 = layers.multiply([x_128, upsample_psi_128])
    g_128 = convolutional_block(y_128, 256, batch_norm, dropout_rate)
    # Classification layer
    outputs = layers.Conv2D(1, kernel_size=(1,1))(g_128)
    outputs = layers.BatchNormalization(axis=3)(outputs)
    outputs = layers.Activation('sigmoid')(outputs)

    model = keras.Model(inputs, outputs, name='UNet')
    return model

if __name__ == '__main__':
    model = AttentionUNet((256, 256, 1))
    print(model.summary())