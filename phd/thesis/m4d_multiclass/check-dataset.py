import os
import numpy as np
from skimage.io import imread

classes = {"sea surface": 0, "oil spill": 1, "look-alike": 2, "ship": 3, "land": 4}

home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, "data", "krestenitis_v3")

for dname in ["train", "test"]:
    dataset_dir = os.path.join(data_dir, dname)
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels_1D")

    ids = os.listdir(images_dir)
    images_fps = [
        os.path.join(images_dir, image_id.split(".")[0] + ".jpg") for image_id in ids
    ]
    labels_fps = [
        os.path.join(labels_dir, image_id.split(".")[0] + ".png") for image_id in ids
    ]

    # Open labels and count class pixel values
    total_pixels = 0
    pixels_per_class = {class_name: 0 for class_name in classes}
    for label_fname in labels_fps:
        # Open label
        label = imread(label_fname)
        # Count pixels per class
        for class_name in classes:
            pixels_per_class[class_name] += np.sum(label == classes[class_name])
        # Count total pixels
        total_pixels += label.shape[0] * label.shape[1]

    print(f"{dname} statistics")
    print(f"\tTotal de pixeles: {total_pixels}")
    print("\tPixeles por clase:")
    total = 0
    for class_name in classes:
        percentage = pixels_per_class[class_name] / total_pixels * 100
        total += pixels_per_class[class_name]
        print(
            f"\t\t{class_name}: {pixels_per_class[class_name]}, porcentaje: {percentage:.2f}"
        )
    print(f"\t\tTotal: {total}, {total / total_pixels * 100:.2f}")
