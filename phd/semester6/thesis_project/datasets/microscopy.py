import os
import cv2
import pywt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

SIZE = 256
class Dataset:
    """Dataset. Read images"""
    def __init__(
            self,
            images_dir,
            masks_dir,
            wave_name = None
    ):
        self.ids = ["_".join(fname.split('.')[0].split('_')[1:]) for fname in os.listdir(images_dir)]
        """
        image_127_2_2.tif  image_157_2_2.tif  image_39_2_2.tif   image_69_2_2.tif  image_99_2_2.tif
        image_127_2_3.tif  image_157_2_3.tif  image_39_2_3.tif   image_69_2_3.tif  image_99_2_3.tif

        mask_127_2_2.tif  mask_157_2_2.tif  mask_39_2_2.tif   mask_69_2_2.tif  mask_99_2_2.tif
        mask_127_2_3.tif  mask_157_2_3.tif  mask_39_2_3.tif   mask_69_2_3.tif  mask_99_2_3.tif
        """
        self.images_fps =[os.path.join(images_dir, f"image_{image_id}.tif") for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, f"mask_{image_id}.tif") for image_id in self.ids]
        self.wave_name = wave_name

    def wavedec2(self, image, name, level):
        c = pywt.wavedec2(image, name, mode='periodization', level=level)
        # normalize each coefficient array independently for better visibility
        c[0] /= np.abs(c[0]).max()
        return c[0]

    def __getitem__(self, index):
        # read data
        image = cv2.imread(self.images_fps[index], 0)/255.0 # Grayscale
        mask = cv2.imread(self.masks_fps[index], 0)/255.0
        # compute the 2D DWT
        if self.wave_name is not None:
            c1 = self.wavedec2(image, self.wave_name, 1)
            c2 = self.wavedec2(image, self.wave_name, 2)
            c3 = self.wavedec2(image, self.wave_name, 3)
            c4 = self.wavedec2(image, self.wave_name, 4)

        image = np.expand_dims(image, -1)
        mask = np.expand_dims(mask, -1)
        if self.wave_name is not None:
            c1 = np.expand_dims(c1, -1)
            c2 = np.expand_dims(c2, -1)
            c3 = np.expand_dims(c3, -1)
            c4 = np.expand_dims(c4, -1)
        #print(image.shape, mask.shape, c1.shape, c2.shape, c3.shape, c4.shape)

        if self.wave_name is not None:
            return image, c1, c2, c3, c4, mask
        else:
            return image, mask

    def __len__(self):
        return len(self.ids)

class Dataloader(tf.keras.utils.Sequence):
    """Load data from dataset and form batches"""
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, index):
        # collect batch data
        start = index * self.batch_size
        stop = (index + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            sample = self.dataset[j]
            #print(j, len(sample), len(sample[0]), sample[0][0].shape, sample[1].shape)
            data.append(sample)
        #print(len(data))

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        #print(len(batch))
        #for i in range(len(batch)):
        #    print(batch[i].shape)
        X = batch[:-1]
        y = batch[-1]
        #print(len(X))
        #print(len(y))
        return X, y

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

def visualize(figure_name, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig(figure_name)
    plt.close()

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1
def CreateDatasets(plot_images = True):
    home_dir = os.path.expanduser('~')
    data_dir = os.path.join(home_dir, 'data', 'microscopy-dataset')
    train_dir = os.path.join(data_dir, 'training2', 'train')
    val_dir = os.path.join(data_dir, 'training2', 'val')

    
    SIZE = 256

    # Use image generators to load images from disk
    seed = 24
    batch_size = 8

    # Parameters for model
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 1

    # Using sequences
    train_dataset = Dataset(os.path.join(train_dir, 'images'), os.path.join(train_dir, 'masks'), 'db1')
    val_dataset = Dataset(os.path.join(val_dir, 'images'), os.path.join(val_dir, 'masks'), 'db1')
    if plot_images:
        # Visualize image
        image, _, _, _, _, mask = train_dataset[5]
        print(image.shape, mask.shape)
        visualize('train_images.png', image=image, mask=mask)
        image, _, _, _, _, mask = val_dataset[5]
        visualize('val_images.png', image=image, mask=mask)
        train_dataloader = Dataloader(train_dataset, batch_size=8, shuffle=True)
        val_dataloader = Dataloader(val_dataset, batch_size=8, shuffle=False)
    #
    return train_dataloader, val_dataloader
