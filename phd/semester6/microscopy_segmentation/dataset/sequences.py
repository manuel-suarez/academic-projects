import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

SIZE = 256
class Dataset:
    """Dataset. Read images"""
    def __init__(
            self,
            images_dir,
            masks_dir
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

    def __getitem__(self, index):
        # read data
        image = cv2.imread(self.images_fps[index], 0)/255.0 # Grayscale
        mask = cv2.imread(self.masks_fps[index], 0)/255.0
        mask = np.expand_dims(mask, -1)
        #print(image.shape, mask.shape)

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
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return batch

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)