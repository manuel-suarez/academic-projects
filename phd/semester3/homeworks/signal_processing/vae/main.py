# Setup
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import StrMethodFormatter
import numpy as np
from numpy.random import random, randint, randn
from zernike import RZern

# Generación de los polinomios
# Definimos parámetros iniciales
order = 6           # Orden de los polinomios a generar
cart = RZern(order) # Generador de polinomios
dim = 128           # Tamaño de dimensiones de imagen
L, K = dim, dim     # Tamaño de cada imagen
num = 30000         # Tamaño conjunto de entrenamiento
num_test = 5000     # Tamaño conjunto de prueba
num_epochs = 50     # Número de épocas para el entrenamiento
latent_dim = 10      # Dimensión del espacio latente de la red VAE
# Definimos el grid para la generación de las señales
ddx = np.linspace(-1.0, 1.0, K)
ddy = np.linspace(-1.0, 1.0, L)
xv, yv = np.meshgrid(ddx, ddy)
cart.make_cart_grid(xv, yv)
# Generamos 30,000 señales variando los parámetros del polinomio a generar
Z = []
for i in range(num):
  # Variamos los parámetros para el polinomio
  c = np.random.normal(size=cart.nk)
  # Generamos polinomio
  Phi = cart.eval_grid(c, matrix=True)
  # Reemplazamos NaN con 0
  p = np.nan_to_num(Phi, False, 0)
  #p = Phi
  # Reescalamos a 0-1 (necesario para que la red calcule correctamente las métricas)
  # Verificar con el Dr. si esto sería necesario en este caso
  p_scaled = (p - np.min(p)) / (np.max(p) - np.min(p))
  # Agregamos a conjunto de resultados
  Z.append(p_scaled)

# Desplegamos una muestra
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    # print(Z[i*3 + j])
    ax[i,j].imshow(Z[i*3 + j])
    ax[i,j].set_title(f"Señal {i*3+j}")

fig.tight_layout()
plt.savefig('figure_1.png')

fig, ax = plt.subplots(nrows=3, ncols=3, subplot_kw={"projection": "3d"})
for i in range(3):
  for j in range(3):
    ax[i, j].plot_surface(ddx, ddy, Z[i * 3 + j], cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)
    ax[i, j].set_title(f"Señal {i * 3 + j}")

fig.tight_layout()
plt.savefig('figure_2.png')

# Generamos las derivadas direccionales para cada imagen
Dx = []
Dy = []
for img in Z:
  img_dy, img_dx = np.gradient(img)
  Dx.append(img_dx)
  Dy.append(img_dy)

# Desplegamos una muestra de los gradientes en X
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    ax[i,j].imshow(Dx[i*3 + j])
    ax[i,j].set_title(f"Dx {i*3+j}")

fig.tight_layout()
plt.savefig('figure_3.png')

# Desplegamos una muestra de los gradientes en Y
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    #print(Dy[i*3 + j])
    ax[i,j].imshow(Dy[i*3 + j])
    ax[i,j].set_title(f"Dy {i*3+j}")

fig.tight_layout()
plt.savefig('figure_4.png')

# VAE
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Flatten, Reshape, Dropout, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.utils.vis_utils import plot_model

# Dimensión de la imagen de entrada (el polinomio) utilizado en el entrenamiento y pruebas
INPUT_DIM     = (128,128,1)
# Utilizamos dos canales de entrada para representar las derivadas parciales del polinomio
GRADIENT_DIM  = (128,128,2)
# Dimensión del espacio latente
LATENT_DIM    = 150
BATCH_SIZE    = 384
R_LOSS_FACTOR = 100000  # 10000
EPOCHS        = 50
INITIAL_EPOCH = 0

steps_per_epoch = num//BATCH_SIZE


class Sampler(keras.Model):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def __init__(self, latent_dim, **kwargs):
    super(Sampler, self).__init__(**kwargs)
    self.latent_dim = latent_dim
    self.model = self.sampler_model()
    self.built = True

  def get_config(self):
    config = super(Sampler, self).get_config()
    config.update({"units": self.units})
    return config

  def sampler_model(self):
    '''
    input_dim is a vector in the latent (codified) space
    '''
    input_data = layers.Input(shape=self.latent_dim)
    z_mean = Dense(self.latent_dim, name="z_mean")(input_data)
    z_log_var = Dense(self.latent_dim, name="z_log_var")(input_data)

    self.batch = tf.shape(z_mean)[0]
    self.dim = tf.shape(z_mean)[1]

    epsilon = tf.keras.backend.random_normal(shape=(self.batch, self.dim))
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    model = keras.Model(input_data, [z, z_mean, z_log_var])
    return model

  def call(self, inputs):
    '''
    '''
    return self.model(inputs)

class Encoder(keras.Model):
    def __init__(self, input_dim, output_dim, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
                 use_batch_norm=True, use_dropout=True, **kwargs):
      '''
      '''
      super(Encoder, self).__init__(**kwargs)

      self.input_dim = input_dim
      self.output_dim = output_dim
      self.encoder_conv_filters = encoder_conv_filters
      self.encoder_conv_kernel_size = encoder_conv_kernel_size
      self.encoder_conv_strides = encoder_conv_strides
      self.n_layers_encoder = len(self.encoder_conv_filters)
      self.use_batch_norm = use_batch_norm
      self.use_dropout = use_dropout

      self.model = self.encoder_model()
      self.built = True

    def get_config(self):
      config = super(Encoder, self).get_config()
      config.update({"units": self.units})
      return config

    def encoder_model(self):
      '''
      '''
      encoder_input = layers.Input(shape=self.input_dim, name='encoder')
      x = encoder_input

      for i in range(self.n_layers_encoder):
        x = Conv2D(filters=self.encoder_conv_filters[i],
                   kernel_size=self.encoder_conv_kernel_size[i],
                   strides=self.encoder_conv_strides[i],
                   padding='same',
                   name='encoder_conv_' + str(i), )(x)
        if self.use_batch_norm:
          x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
          x = Dropout(rate=0.25)(x)

      self.last_conv_size = x.shape[1:]
      x = Flatten()(x)
      encoder_output = Dense(self.output_dim)(x)
      model = keras.Model(encoder_input, encoder_output)
      return model

    def call(self, inputs):
      '''
      '''
      return self.model(inputs)


class Decoder(keras.Model):
  def __init__(self, input_dim, input_conv_dim,
               decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides,
               use_batch_norm=True, use_dropout=True, **kwargs):

    '''
    '''
    super(Decoder, self).__init__(**kwargs)

    self.input_dim = input_dim
    self.input_conv_dim = input_conv_dim

    self.decoder_conv_t_filters = decoder_conv_t_filters
    self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
    self.decoder_conv_t_strides = decoder_conv_t_strides
    self.n_layers_decoder = len(self.decoder_conv_t_filters)

    self.use_batch_norm = use_batch_norm
    self.use_dropout = use_dropout

    self.model = self.decoder_model()
    self.built = True

  def get_config(self):
    config = super(Decoder, self).get_config()
    config.update({"units": self.units})
    return config

  def decoder_model(self):
    '''
    '''
    decoder_input = layers.Input(shape=self.input_dim, name='decoder')
    x = Dense(np.prod(self.input_conv_dim))(decoder_input)
    x = Reshape(self.input_conv_dim)(x)

    for i in range(self.n_layers_decoder):
      x = Conv2DTranspose(filters=self.decoder_conv_t_filters[i],
                          kernel_size=self.decoder_conv_t_kernel_size[i],
                          strides=self.decoder_conv_t_strides[i],
                          padding='same',
                          name='decoder_conv_t_' + str(i))(x)
      if i < self.n_layers_decoder - 1:

        if self.use_batch_norm:
          x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
          x = Dropout(rate=0.25)(x)
      else:
        x = Activation('sigmoid')(x)
    decoder_output = x
    model = keras.Model(decoder_input, decoder_output)
    return model

  def call(self, inputs):
    '''
    '''
    return self.model(inputs)

class VAE(keras.Model):
    def __init__(self, r_loss_factor=1, summary=False, **kwargs):
      super(VAE, self).__init__(**kwargs)

      self.r_loss_factor = r_loss_factor

      # Architecture
      self.input_dim = GRADIENT_DIM
      self.latent_dim = LATENT_DIM
      # Utilizamos un número mayor de capas convolucionales para obtener mejor
      # las características del gradiente de entrada
      self.encoder_conv_filters = [64, 64, 64, 64]
      self.encoder_conv_kernel_size = [3, 3, 3, 3]
      self.encoder_conv_strides = [2, 2, 2, 2]
      self.n_layers_encoder = len(self.encoder_conv_filters)

      self.decoder_conv_t_filters = [64, 64, 64, 1]
      self.decoder_conv_t_kernel_size = [3, 3, 3, 3]
      self.decoder_conv_t_strides = [2, 2, 2, 2]
      self.n_layers_decoder = len(self.decoder_conv_t_filters)

      self.use_batch_norm = True
      self.use_dropout = True

      self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
      self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
      self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
      self.mae = tf.keras.losses.MeanAbsoluteError()

      # Encoder
      self.encoder_model = Encoder(input_dim=self.input_dim,
                                   output_dim=self.latent_dim,
                                   encoder_conv_filters=self.encoder_conv_filters,
                                   encoder_conv_kernel_size=self.encoder_conv_kernel_size,
                                   encoder_conv_strides=self.encoder_conv_strides,
                                   use_batch_norm=self.use_batch_norm,
                                   use_dropout=self.use_dropout)
      self.encoder_conv_size = self.encoder_model.last_conv_size
      if summary:
        self.encoder_model.summary()

      # Sampler
      self.sampler_model = Sampler(latent_dim=self.latent_dim)
      if summary:
        self.sampler_model.summary()

      # Decoder
      self.decoder_model = Decoder(input_dim=self.latent_dim,
                                   input_conv_dim=self.encoder_conv_size,
                                   decoder_conv_t_filters=self.decoder_conv_t_filters,
                                   decoder_conv_t_kernel_size=self.decoder_conv_t_kernel_size,
                                   decoder_conv_t_strides=self.decoder_conv_t_strides,
                                   use_batch_norm=self.use_batch_norm,
                                   use_dropout=self.use_dropout)
      if summary: self.decoder_model.summary()

      self.built = True

    @property
    def metrics(self):
      return [self.total_loss_tracker,
              self.reconstruction_loss_tracker,
              self.kl_loss_tracker, ]

    @tf.function
    def train_step(self, data):
      '''
      '''
      # Desestructuramos data ya que contiene los dos inputs (gradientes, integral)
      gradients, integral = data[0]
      with tf.GradientTape() as tape:
        # predict
        x = self.encoder_model(gradients)
        z, z_mean, z_log_var = self.sampler_model(x)
        pred = self.decoder_model(z)

        # loss
        r_loss = self.r_loss_factor * self.mae(integral, pred)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = r_loss + kl_loss

      # gradient
      grads = tape.gradient(total_loss, self.trainable_weights)
      # train step
      self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

      # compute progress
      self.total_loss_tracker.update_state(total_loss)
      self.reconstruction_loss_tracker.update_state(r_loss)
      self.kl_loss_tracker.update_state(kl_loss)
      return {"loss": self.total_loss_tracker.result(),
              "reconstruction_loss": self.reconstruction_loss_tracker.result(),
              "kl_loss": self.kl_loss_tracker.result(), }

    @tf.function
    def generate(self, z_sample):
      '''
      We use the sample of the N(0,I) directly as
      input of the deterministic generator.
      '''
      return self.decoder_model(z_sample)

    @tf.function
    def codify(self, images):
      '''
      For an input image we obtain its particular distribution:
      its mean, its variance (unvertaintly) and a sample z of such distribution.
      '''
      x = self.encoder_model.predict(images)
      z, z_mean, z_log_var = self.sampler_model(x)
      return z, z_mean, z_log_var

    # implement the call method
    @tf.function
    def call(self, inputs, training=False):
      '''
      '''
      tmp1, tmp2 = self.encoder_model.use_Dropout, self.decoder_model.use_Dropout
      if not training:
        self.encoder_model.use_Dropout, self.decoder_model.use_Dropout = False, False

      x = self.encoder_model(inputs)
      z, z_mean, z_log_var = self.sampler_model(x)
      pred = self.decoder_model(z)

      self.encoder_model.use_Dropout, self.decoder_model.use_Dropout = tmp1, tmp2
      return pred
vae = VAE(r_loss_factor=R_LOSS_FACTOR, summary=True)
vae.summary()
vae.compile(optimizer=keras.optimizers.Adam())

# Reemplazamos el conjunto de prueba del ejemplo MNIST a los gradientes
# de los polinomios generados

# Convertimos la lista a tensor y agregamos la dimensión del canal (1)
Dtf_x = tf.expand_dims(tf.convert_to_tensor(Dx, dtype=tf.float32), axis=-1)
Dtf_y = tf.expand_dims(tf.convert_to_tensor(Dy, dtype=tf.float32), axis=-1)
# Combinamos los gradientes en un tensor de 2 canales de acuerdo con la especificación
# de la entrada del encoder INPUT_DIM
Dtf = tf.keras.layers.Concatenate(axis=3)([Dtf_x, Dtf_y])
Ztf = tf.expand_dims(tf.convert_to_tensor(Z, dtype=tf.float32), axis=-1)

# Visualizamos el primer dato para verificar que el Tensor se haya creado correctamente
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
ax[0].imshow(Ztf[0,:,:,0])
ax[1].imshow(Dtf[0,:,:,0])
ax[2].imshow(Dtf[0,:,:,1])
fig.tight_layout()
plt.savefig('figure_5.png')

from tensorflow.keras.callbacks import ModelCheckpoint
filepath = 'best_weight_model.h5'
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')
callbacks = [checkpoint]

vae.fit([Dtf, Ztf],
        batch_size      = BATCH_SIZE,
        epochs          = EPOCHS,
        initial_epoch   = INITIAL_EPOCH,
        steps_per_epoch = steps_per_epoch,
        callbacks       = callbacks)
vae.save_weights("final_weights_model.h5")