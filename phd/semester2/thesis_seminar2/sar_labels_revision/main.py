import os
import cv2
import random
import shutil
import numpy as np

# Directories config
data_path = '/home/est_posgrado_manuel.suarez/data/projects/consorcio-ia/data/oil_spills_17/augmented_dataset'
labels_path = 'labels'
features_path = 'features/ORIGIN'

# Get dir list files
labels_files = [(filename, np.sum(cv2.imread(os.path.join(data_path, labels_path, filename), cv2.IMREAD_GRAYSCALE)))
                for filename in os.listdir(os.path.join(data_path, labels_path))
                if os.path.isfile(os.path.join(data_path, labels_path, filename))]
print(len(labels_files))
print(labels_files[1])

# Get segmentations mask with all 0's or all 1's (255)
allZeros = [f for f in labels_files if f[1] == 0]
allOnes = [f for f in labels_files if f[1] == 224*224*255]
print(len(allZeros), len(allOnes))

# Shuffle
# Reordenamos
random.shuffle(allZeros)
random.shuffle(allOnes)
print(len(allZeros), len(allOnes))

# Get a 80-20 division for training and validation
zeros_train = allZeros[:8000]
zeros_val = allZeros[8001:10001]
print(len(zeros_train), len(zeros_val))
ones_train = allOnes[:8000]
ones_val = allOnes[8001:10001]
print(len(ones_train), len(ones_val))

# Create directories
for feat_dir in ['training', 'validation']:
    os.mkdir(feat_dir)
    for class_dir in ['spill', 'no_spill']:
        os.mkdir(os.path.join(feat_dir, class_dir))
# Copy files to respective directories
for item in zeros_train:
    fname, fext = os.path.splitext(item[0])
    shutil.copyfile(os.path.join(data_path, features_path, f"{fname}.tiff"), os.path.join('training', 'no_spill', f"{fname}.tiff"))
for item in ones_train:
    fname, fext = os.path.splitext(item[0])
    shutil.copyfile(os.path.join(data_path, features_path, f"{fname}.tiff"), os.path.join('training', 'spill', f"{fname}.tiff"))
for item in zeros_val:
    fname, fext = os.path.splitext(item[0])
    shutil.copyfile(os.path.join(data_path, features_path, f"{fname}.tiff"), os.path.join('validation', 'no_spill', f"{fname}.tiff"))
for item in ones_val:
    fname, fext = os.path.splitext(item[0])
    shutil.copyfile(os.path.join(data_path, features_path, f"{fname}.tiff"), os.path.join('validation', 'spill', f"{fname}.tiff"))

print('Done')