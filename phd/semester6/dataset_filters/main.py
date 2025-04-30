import os
import cv2
import rasterio
import numpy as np
from rasterio.plot import show
from matplotlib import pyplot as plt
from PIL import Image
from skimage import io
from cv2.ximgproc import guidedFilter
from skimage.filters import roberts, sobel, scharr, prewitt, farid

Image.MAX_IMAGE_PIXELS = None
home_dir = os.path.expanduser("~")
work_dir = os.path.expanduser(".")
data_dir = os.path.join(home_dir, "data", "dataset-cimat")
tiff_dir = os.path.join(data_dir, "tiff")

if os.getenv('SLURM_ARRAY_JOB_ID', '') != '':
    slurm_array_job_id = int(os.getenv('SLURM_ARRAY_JOB_ID'))
    slurm_array_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    print(f"SLURM_ARRAY_JOB_ID: {slurm_array_job_id}")
    print(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")

#print(os.listdir(tiff_dir))
#exit(0)
def load_files(fname):
    img_tiff = rasterio.open(os.path.join(tiff_dir, fname)).read(1)

    return img_tiff

def apply_edge_filters(img_tiff):
    print("\tRoberts")
    img_roberts = roberts(img_tiff)
    print("\tSobel")
    img_sobel = sobel(img_tiff)
    print("\tScharr")
    img_scharr = scharr(img_tiff)
    print("\tPrewitt")
    img_prewitt = prewitt(img_tiff)
    print("\tFarid")
    img_farid = farid(img_tiff)
    return img_roberts, img_sobel, img_scharr, img_prewitt, img_farid

def apply_edge_preservingfilter(img_tiff):
    print("\tEdge preserving")
    img_edge = cv2.edgePreservingFilter(img_tiff, flags=1, sigma_s=60, sigma_r=0.4)
    #print("\tDetail enhance")
    #img_detail = cv2.detailEnhance(img_tiff, sigma_s=10, sigma_r=0.15)
    return img_edge,#, img_detail

def apply_guided_filter(img_tiff):
    print("\tGuided Filter")
    img_guided = guidedFilter(img_tiff,13,70)
    return img_guided,

def plot_image_and_filters(work_dir, fname, names, images):
    #fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 10))
    # Display of original image, mask and segmentation
    for i, (name, image) in enumerate(zip(names, images)):
        fig = plt.figure()
        im = plt.imshow(image, cmap="gray")
        #axes[i].set_axis_off()
        #axes[i].title.set_text(titles[i])
        fig.colorbar(im, fraction=0.070, pad=0.04)
        plt.tight_layout()
        fig.suptitle(fname.split(".")[0], fontsize=16)
        # plt.show()
        plt.savefig(os.path.join(work_dir,'results',fname.split('.')[0],f"{name}.png"))

def plot_image(work_dir, fname):
    img_tiff = load_files(fname)
    images = apply_edge_filters(img_tiff)
    names = ["roberts","sobel","scharr","prewitt","farid"]
    plot_image_and_filters(work_dir, fname, names, images)
    images = apply_edge_preservingfilter(img_tiff)
    names = ["edge"]
    plot_image_and_filters(work_dir, fname, names, images)
    images = apply_guided_filter(img_tiff)
    names = ["guided"]
    plot_image_and_filters(work_dir, fname, names, images)

# Process file according to array task ID
if os.getenv('SLURM_ARRAY_TASK_ID', '') == '':
    fname = os.listdir(tiff_dir)[0]
else:
    fname = os.listdir(tiff_dir)[int(os.getenv('SLURM_ARRAY_TASK_ID'))]
print(fname)
# Make directory result based on filename
os.makedirs(os.path.join(work_dir, "results", fname.split('.')[0]), exist_ok=True)
plot_image(work_dir, fname)
