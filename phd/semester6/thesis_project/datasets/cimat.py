import sys
import yaml
import numpy as np
import pandas as pd
import tensorflow.keras
from skimage.io import imread


class DataGenerator(tensorflow.keras.utils.Sequence):


    '''
    Contructor to load images by batch
    @param key Basename of dataset of images
    @param featurePath Path of dataset of images
    @param labelPath Path of dataset of label images (Ground-truth)
    @param featureChannels Name of directories for each image defined as channels
    @param featureExt File extension of dataset of image
    @param labelExt File extension of label images
    @param batch_size Batch size used during the training
    @param dim Size of the images to read.
    '''
    def __init__(self, keys, featurePath, labelPath, featureChannels, 
         featureExt, labelExt, batch_size=32, dim=(32,32,1)):
        
        self.keys = keys 
        self.featurePath = featurePath
        self.labelPath = labelPath
        self.featureChannels = featureChannels
        self.featureExt = featureExt
        self.labelExt = labelExt
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.keys) / self.batch_size))


    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.keys[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.keys))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        #'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = []
        Y = []

        # Generate data
        for i, key in enumerate(list_IDs_temp):

            x = np.zeros( self.dim, dtype=np.float32 )

            for j , feature in enumerate( self.featureChannels ):

                filename = self.featurePath + feature + "/" + key + self.featureExt
                z = imread(filename, as_gray=True).astype(np.float32)

                if z.shape[0] == self.dim[0] and z.shape[1] == self.dim[1]:

                    x[...,j] = z

            filename = self.labelPath + key + self.labelExt

            y = np.zeros( (x.shape[0], x.shape[1], 1) )

            z = imread(filename, as_gray=True).astype(np.float32)/255.0

            if z.shape[0] == self.dim[0] and z.shape[1] == self.dim[1]:            
                y[...,0] = z
            
            X.append( x )
            Y.append( y )

        return np.array(X), np.array(Y)


def CreateDatasets(plot_images = True):
    # Configure datasets
    trainingFile = sys.argv[1]
    crossFile = sys.argv[2]
    featurePath = sys.argv[3]
    labelPath = sys.argv[4]
    outputPath = sys.argv[5]
    configFile = sys.argv[6]

    config = yaml.safe_load(open(configFile, 'r'))

    trainSet = pd.read_csv(trainingFile)
    crossSet = pd.read_csv(crossFile)

    training_generator = DataGenerator(trainSet["key"],
                                       featurePath,
                                       labelPath,
                                       config["channel_configuration"],
                                       config["feature_ext"],
                                       config["label_ext"],
                                       batch_size=config["batch_size"],
                                       dim=config["input_shape"])
    validation_generator = DataGenerator(crossSet["key"],
                                         featurePath,
                                         labelPath,
                                         config["channel_configuration"],
                                         config["feature_ext"],
                                         config["label_ext"],
                                         batch_size=config["batch_size"],
                                         dim=config["input_shape"])
    
    return training_generator, validation_generator
