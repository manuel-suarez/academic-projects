import os
import numpy as np
from skimage.io import imread
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(
        self,
        keys,
        features_path,
        labels_path,
        features_channels,
        feature_ext,
        label_ext,
        batch_size,
        dimensions,
    ):
        super().__init__()
        self.keys = keys
        self.features_path = features_path
        self.labels_path = labels_path
        self.features_channels = features_channels
        self.feature_ext = feature_ext
        self.label_ext = label_ext
        self.batch_size = batch_size
        self.dims = dimensions
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.keys) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_Ids = [self.keys[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_Ids)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.keys))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_Ids):
        X = []
        Y = []

        # Generate data
        for i, key in enumerate(list_Ids):
            x = np.zeros(self.dims, dtype=np.float32)
            # Load features
            for j, feature in enumerate(self.features_channels):
                filename = os.path.join(self.features_path, feature, key + self.feature_ext)
                z = imread(filename, as_gray=True).astype(np.float32)

                if z.shape[0] == self.dims[0] and z.shape[1] == self.dims[1]:
                    x[...,j] = z
            # Load label
            filename = os.path.join(self.labels_path, key + self.label_ext)
            y = np.zeros( (x.shape[0], x.shape[1], 1))
            z = imread(filename, as_gray=True).astype(np.float32)/255.0

            if z.shape[0] == self.dims[0] and z.shape[1] == self.dims[1]:
                y[...,0] = z

            X.append(x)
            Y.append(y)

        return np.array(X), np.array(Y)


