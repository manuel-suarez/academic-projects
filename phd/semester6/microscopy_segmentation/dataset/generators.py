import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(data_dir, batch_size, seed):
    img_data_gen_args = dict(rescale = 1/255.,
                             rotation_range=90,
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             shear_range=0.5,
                             zoom_range=0.3,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='reflect')
    mask_data_gen_args = dict(rescale = 1/255.,
                              rotation_range=90,
                              width_shift_range=0.3,
                              height_shift_range=0.3,
                              shear_range=0.5,
                              zoom_range=0.3,
                              horizontal_flip=True,
                              vertical_flip=True,
                              fill_mode='reflect',
                              preprocessing_function=lambda x: np.where(x>0, 1, 0).astype(x.dtype))
    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    num_train_imgs = len(os.listdir(os.path.join(data_dir,'train_images','train')))
    image_generator = image_data_generator.flow_from_directory(os.path.join(data_dir,'train_images'),
                                                               seed=seed,
                                                               batch_size=batch_size,
                                                               class_mode=None) # Binary
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    mask_generator = mask_data_generator.flow_from_directory(os.path.join(data_dir,'train_masks'),
                                                             seed=seed,
                                                             batch_size=batch_size,
                                                             color_mode='grayscale',
                                                             class_mode=None) # Binary

    valid_img_generator = image_data_generator.flow_from_directory(os.path.join(data_dir,'val_images'),
                                                                   seed=seed,
                                                                   batch_size=batch_size,
                                                                   class_mode=None)
    valid_mask_generator = mask_data_generator.flow_from_directory(os.path.join(data_dir,'val_masks'),
                                                                   seed=seed,
                                                                   batch_size=batch_size,
                                                                   color_mode='grayscale',
                                                                   class_mode=None)
    test_img_generator = image_data_generator.flow_from_directory(os.path.join(data_dir,'test_images'),
                                                                  seed=seed,
                                                                  batch_size=32,
                                                                  class_mode=None)
    test_mask_generator = mask_data_generator.flow_from_directory(os.path.join(data_dir,'test_masks'),
                                                                  seed=seed,
                                                                  batch_size=32,
                                                                  color_mode='grayscale',
                                                                  class_mode=None)
    train_generator = zip(image_generator, mask_generator)
    val_generator = zip(valid_img_generator, valid_mask_generator)
    return num_train_imgs, train_generator, val_generator