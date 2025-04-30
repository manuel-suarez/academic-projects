import os
import cv2
import numpy
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

SIZE = 256

def create_datasets(image_dir, mask_dir):
    image_dataset = []
    mask_dataset = []

    images = os.listdir(image_dir)
    for i, image_name in tqdm(enumerate(images)):
        if image_name.split('.')[1] == 'tif':
            image = cv2.imread(os.path.join(image_dir, image_name), 0)
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            image_dataset.append(numpy.array(image))

    masks = os.listdir(mask_dir)
    for i, image_name in tqdm(enumerate(masks)):
        if image_name.split('.')[1] == 'tif':
            image = cv2.imread(os.path.join(mask_dir, image_name), 0)
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            mask_dataset.append(numpy.array(image))

    # Normalize images
    image_dataset = numpy.array(image_dataset)/255.
    mask_dataset = numpy.expand_dims((numpy.array(mask_dataset)),3)/255.
    print(image_dataset.shape, mask_dataset.shape)
    print(image_dataset[0].shape, mask_dataset[0].shape)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.10, random_state=0)

    import random
    import numpy as np
    image_number = random.randint(0, len(X_train))
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
    plt.subplot(122)
    plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
    plt.show()

    # Parameters for model
    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH = X_train.shape[2]
    IMG_CHANNELS = 1
    num_labels = 1
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    return X_train, X_test, y_train, y_test
