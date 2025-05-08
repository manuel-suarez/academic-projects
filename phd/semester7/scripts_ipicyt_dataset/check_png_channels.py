import os

from PIL import Image
from skimage.io import imread

base_path = os.path.expanduser("~")
data_path = os.path.join(base_path, "data", "cimat", "dataset-cimat")
mask_path = os.path.join(data_path, "mask_png")

Image.MAX_IMAGE_PIXELS = None
for mask_name in os.listdir(mask_path):
    mask = imread(os.path.join(mask_path, mask_name))
    print(mask_name, mask.shape, mask.dtype, mask.min(), mask.max())
