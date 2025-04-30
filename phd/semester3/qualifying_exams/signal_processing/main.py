# Import
import numpy as np
import tensorflow as tf
from numpy.random import random
from zernike import RZern
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Flatten, Reshape, Dropout, BatchNormalization, Activation, LeakyReLU
from matplotlib import pyplot as plt

# Usamos un mismo conjunto de polinomios para el entrenamiento con variación de parámetros
order = 6           # Orden de los polinomios a generar
cart = RZern(order) # Generador de polinomios
dim = 128           # Tamaño de dimensiones de imagen
L, K = dim, dim     # Tamaño de cada imagen
num = 20000         # Tamaño conjunto de entrenamiento
num_test = 5000     # Tamaño conjunto de prueba
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
  #p_scaled = (p - np.min(p)) / (np.max(p) - np.min(p))
  # Agregamos a conjunto de resultados
  Z.append(p)

# Desplegamos una muestra
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    # print(Z[i*3 + j])
    print(f"Señal {i*3+j}, máx: {np.max(Z[i*3+j])}, min: {np.min(Z[i*3+j])}")
    ax[i,j].imshow(Z[i*3 + j])
    ax[i,j].set_title(f"Señal {i*3+j}")

fig.tight_layout()
plt.savefig(f"figura1.png")
plt.close()

# Definimos función W para el envolvimiento de fase del polinomio
def W(p):
  return np.arctan2(np.sin(p), np.cos(p))

# En vez de generar las derivadas direccionales ahora aplicamos el envolvimiento de fase del polinomio
WZ = []
for img in Z:
  WZ.append(W(img))

# Desplegamos una muestra de los polinomios con envolvimiento de fase
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    print(f"Señal {i*3+j}, máx: {np.max(WZ[i*3+j])}, min: {np.min(WZ[i*3+j])}")
    ax[i,j].imshow(WZ[i*3 + j])
    ax[i,j].set_title(f"W(Señal {i*3+j})")

fig.tight_layout()
plt.savefig(f"figura2.png")
plt.close()

# Generamos las derivadas direccionales para cada imagen sin envoltura
Dx = []
Dy = []
for img in Z:
  img_dy, img_dx = np.gradient(img)
  # Aplicamos W a los gradientes para eliminar las transiciones
  Dx.append(img_dx)
  Dy.append(img_dy)

# Desplegamos una muestra de los gradientes en X
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    ax[i,j].imshow(Dx[i*3 + j])
    ax[i,j].set_title(f"Dx {i*3+j}")

fig.tight_layout()
plt.savefig(f"figura3.png")
plt.close()

# Desplegamos una muestra de los gradientes en Y
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    #print(Dy[i*3 + j])
    ax[i,j].imshow(Dy[i*3 + j])
    ax[i,j].set_title(f"Dy {i*3+j}")

fig.tight_layout()
plt.savefig(f"figura4.png")
plt.close()

# Generamos las derivadas direccionales para cada imagen sobre la fase envuelta
DWx = []
DWy = []
for img in WZ:
  img_dy, img_dx = np.gradient(img)
  # Aplicamos W a los gradientes para eliminar las transiciones
  DWx.append(W(img_dx))
  DWy.append(W(img_dy))

# Desplegamos una muestra de los gradientes en X
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    ax[i,j].imshow(DWx[i*3 + j])
    ax[i,j].set_title(f"DWx {i*3+j}")

fig.tight_layout()
plt.savefig(f"figura5.png")
plt.close()

# Desplegamos una muestra de los gradientes en Y
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    #print(Dy[i*3 + j])
    ax[i,j].imshow(DWy[i*3 + j])
    ax[i,j].set_title(f"DWy {i*3+j}")

fig.tight_layout()
plt.savefig(f"figura6.png")
plt.close()

# Generación de conjunto de polinomios de prueba
# Generamos imágenes de prueba
Z_test = []
for i in range(num_test):
  # Variamos los parámetros para el polinomio
  c = np.random.normal(size=cart.nk)
  # Generamos polinomio
  Phi = cart.eval_grid(c, matrix=True)
  # Reemplazamos NaN con 0
  p = np.nan_to_num(Phi, False, 0)
  # Reescalamos a 0-1 (necesario para que la red calcule correctamente las métricas)
  #p_scaled = (p - np.min(p)) / (np.max(p) - np.min(p))
  Z_test.append(p)
# Aplicamos función de fase
WZ_test = []
for img in Z_test:
    WZ_test.append(W(img))
# Generamos las derivadas direccionales para cada imagen
Dx_test = []
Dy_test = []
for img in Z_test:
  img_dy, img_dx = np.gradient(img)
  Dx_test.append(img_dx)
  Dy_test.append(img_dy)

DWx_test = []
DWy_test = []
for img in WZ_test:
  img_dy, img_dx = np.gradient(img)
  DWx_test.append(W(img_dx))
  DWy_test.append(W(img_dy))


# Conversión a tensores
# Convertimos a tensores
Dtf_x_test = tf.expand_dims(tf.convert_to_tensor(DWx_test, dtype=tf.float32), axis=-1)
Dtf_y_test = tf.expand_dims(tf.convert_to_tensor(DWy_test, dtype=tf.float32), axis=-1)
# print(Dtf_x.shape)
# print(Dtf_y.shape)
Dtf_test = tf.keras.layers.Concatenate(axis=3)([Dtf_x_test, Dtf_y_test])
Ztf_test = tf.expand_dims(tf.convert_to_tensor(Z_test, dtype=tf.float32), axis=-1)

# Definimos parámetros iniciales
def train_model(scale_factor, fn_activation):
    '''
    :param scale_factor: Factor de escala para obtener el rango dinámico de los polinomios
    :param fn_activation: Función de activación del decodificador
    :return:
    '''
    print(80*"=")
    print(f"Factor de escala: {scale_factor}, Función de activación: {fn_activation}")
    print(80 * "=")

    # Arquitectura del Autoencoder variacional

    # Dimensión de la imagen de entrada (el polinomio) utilizado en el entrenamiento y pruebas
    INPUT_DIM     = (128,128,1)
    # Utilizamos dos canales de entrada para representar las derivadas parciales del polinomio
    GRADIENT_DIM  = (128,128,2)
    # Dimensión del espacio latente
    LATENT_DIM    = 150
    BATCH_SIZE    = 384
    R_LOSS_FACTOR = 100000  # 10000
    EPOCHS        = 100
    INITIAL_EPOCH = 0

    steps_per_epoch = num//BATCH_SIZE

    # Generador de muestras del espacio latente
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

    # Arquitectura del codificador
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

    # Decodificador
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
            x = Activation('tanh')(x)
        decoder_output = x
        model = keras.Model(decoder_input, decoder_output)
        return model

      def call(self, inputs):
        '''
        '''
        return self.model(inputs)

    # Autoencoder variacional
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
          pred = scale_factor * self.decoder_model(z)

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

    # Preparación de los datos de entrada

    # Convertimos la lista a tensor y agregamos la dimensión del canal (1)
    Dtf_Wx = tf.expand_dims(tf.convert_to_tensor(DWx, dtype=tf.float32), axis=-1)
    Dtf_Wy = tf.expand_dims(tf.convert_to_tensor(DWy, dtype=tf.float32), axis=-1)
    # Combinamos los gradientes en un tensor de 2 canales de acuerdo con la especificación
    # de la entrada del encoder INPUT_DIM
    Dtf = tf.keras.layers.Concatenate(axis=3)([Dtf_Wx, Dtf_Wy])
    Ztf = tf.expand_dims(tf.convert_to_tensor(Z, dtype=tf.float32), axis=-1)
    # Visualizamos el primer dato para verificar que el Tensor se haya creado correctamente
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    for i in range(3):
        ax[i,0].imshow(Ztf[i,:,:,0])
        ax[i,0].set_title(f"Z {i}")
        ax[i,1].imshow(Dtf[i,:,:,0])
        ax[i,1].set_title(f"DWx {i}")
        ax[i,2].imshow(Dtf[i,:,:,1])
        ax[i,2].set_title(f"DWy {i}")
    fig.tight_layout()
    plt.savefig(f"scale_{scale_factor}_activation_{fn_activation}_tensor_entrada.png")
    plt.close()

    # Instanciación de la VAE
    vae = VAE(r_loss_factor=R_LOSS_FACTOR)

    from tensorflow.keras.callbacks import ModelCheckpoint
    filepath = 'best_weight_model_vae2.h5'
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='min')
    callbacks = [checkpoint]

    # Entrenamiento de la red
    vae.compile(optimizer=keras.optimizers.Adam())
    history = vae.fit([Dtf, Ztf],
            batch_size      = BATCH_SIZE,
            epochs          = EPOCHS,
            initial_epoch   = INITIAL_EPOCH,
            steps_per_epoch = steps_per_epoch,
            callbacks       = callbacks)
    # Resultados del entrenamiento

    # Plot training & validation loss values
    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['reconstruction_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Loss', 'Reconstruction'], loc='upper left')
    plt.savefig(f"scale_{scale_factor}_activation_{fn_activation}_training_results.png")
    plt.close()

    vae.save_weights("final_weights_model_vae2.h5")

    def plot_latent_space(vae, input_size=(28,28,1), n=30, figsize=15,  scale=1., latents_start=[0,1]):
        # display a n*n 2D manifold of digits
        canvas = np.zeros((input_size[0]*n, input_size[1]*n, input_size[2]))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        z_sample = np.random.normal(0,1,(1,vae.latent_dim))
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample[0][latents_start[0]], z_sample[0][latents_start[1]]=xi,yi
                x_decoded = scale_factor * vae.generate(z_sample)
                img = x_decoded[0].numpy().reshape(input_size)
                canvas[i*input_size[0] : (i + 1)*input_size[0],
                       j*input_size[1] : (j + 1)*input_size[1],
                       : ] = img

        plt.figure(figsize=(figsize, figsize))
        start_range    = input_size[0] // 2
        end_range      = n*input_size[0] + start_range
        pixel_range    = np.arange(start_range, end_range, input_size[0])
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[{}]".format(latents_start[0]))
        plt.ylabel("z[{}]".format(latents_start[1]))
        plt.imshow(canvas[:,:,0])
        plt.savefig(f"scale_{scale_factor}_activation_{fn_activation}_latent_space.png")
        plt.close()
    plot_latent_space(vae, input_size=INPUT_DIM, n = 6, latents_start=[20,30], scale=3)

    # Predicción de resultados
    # Visualizamos un conjunto de predicciones
    num_vis = 4
    fig, ax = plt.subplots(nrows=num_vis, ncols=4, figsize=(10, 10))
    for i in range(num_vis):
      # Obtenemos la predicción
      data = tf.expand_dims(Dtf_test[i], axis=0)
      x  = vae.encoder_model(data)
      z, z_mean, z_log_var = vae.sampler_model(x)
      x_decoded = scale_factor * vae.decoder_model(z)
      #digit = x_decoded[0].reshape(digit_size, digit_size)
      print(np.min(x_decoded), np.max(x_decoded))
      print(i, np.min(Ztf_test[i]), np.max(Ztf_test[i]))

      # Desplegamos
      ax[i, 0].imshow(Dtf_test[i,:,:,0])
      ax[i, 0].set_title('Gradiente X')
      ax[i, 1].imshow(Dtf_test[i,:,:,1])
      ax[i, 1].set_title('Gradiente Y')
      # ax[i, 2].imshow(pred[0,:,:,0])
      ax[i, 2].imshow(x_decoded[0,:,:,0])
      ax[i, 2].set_title('Predicción')
      ax[i, 3].imshow(Ztf_test[i,:,:,0])
      ax[i, 3].set_title('Polinomio')
    fig.tight_layout()
    plt.savefig(f"scale_{scale_factor}_activation_{fn_activation}_prediction_results.png")
    plt.close()

ACTIVATIONS = ['sigmoid','tanh']
SCALE_FACTORS = [1, 5, 10, 20, 40]
for scale_factor in SCALE_FACTORS:
    for activation in ACTIVATIONS:
        train_model(scale_factor=scale_factor, fn_activation=activation)