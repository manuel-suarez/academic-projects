import os, glob
import rasterio
import numpy as np
from PIL import Image
from skimage.io import imread
from matplotlib import pyplot as plt

Image.MAX_IMAGE_PIXELS = None

home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, "data", "cimat", "dataset-cimat")
app_dir = os.path.join(home_dir, "data", "cimat", "ml_application")
out_dir = os.path.join(app_dir, "out")
res_dir = os.path.join(app_dir, "results")
dwh_dir = os.path.join("dwh")
os.makedirs("figures", exist_ok=True)


def plot_image_and_mask(filename):
    # fname is the date, so we need to search for ASAR filename
    asar_fname = os.path.join(out_dir, filename, f"{filename}.png")
    label_fname = os.path.join(res_dir, f"{filename}_not_wind", f"{filename}_mask.png")
    var_fname = os.path.join(out_dir, f"{filename}", f"{filename}_var.tif")
    dwh_fname = os.path.join(dwh_dir, "png", f"{filename}.png")

    image = imread(asar_fname, as_gray=True)
    label = imread(label_fname, as_gray=True).astype(np.float32)
    var = imread(var_fname, as_gray=True)
    if os.path.exists(dwh_fname):
        dwh = imread(dwh_fname, as_gray=True)
    print("Label array: ", np.unique(label), label.shape, label.dtype)
    # Binarize
    binary_image = np.where(label > 0, 1, 0)

    print(fname, image.shape, label.shape, binary_image.shape)
    print(image.dtype, label.dtype, binary_image.dtype)
    # mask = np.ma.masked_values(label, 0.0)

    if os.path.exists(dwh_fname):
        fig, ax = plt.subplots(1, 5, figsize=(26, 6))
    else:
        fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    ax[0].set_title("SAR origin")
    im0 = ax[0].imshow(image, cmap="gray")
    fig.colorbar(im0, ax=ax[0])

    ax[1].set_title("Segmentation label")
    im1 = ax[1].imshow(label, cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
    fig.colorbar(im1, ax=ax[1])
    # Masked overlay
    ax[2].set_title("Mask")
    ax[2].imshow(image, cmap="gray")
    mask = np.ma.masked_where(binary_image == 0, binary_image, copy=True)
    im2 = ax[2].imshow(
        mask,
        cmap="viridis",
        alpha=0.8,
        vmin=0.0,
        vmax=1.0,
        interpolation="none",
    )
    fig.colorbar(im2, ax=ax[2])

    ax[3].set_title("Variance")
    ax[3].imshow(var, cmap="gray")

    if os.path.exists(dwh_fname):
        ax[4].set_title("Segmentation")
        ax[4].imshow(dwh, cmap="gray")

    fig.suptitle(f"{fname}, {asar_fname[0].split('/')[-1]}")

    plt.savefig(os.path.join("figures", filename + ".png"))
    plt.close()


slurm_ntasks = os.getenv("SLURM_NTASKS", 1)
slurm_procid = os.getenv("SLURM_PROCID", 1)
slurm_task_pid = os.getenv("SLURM_TASK_PID", 1)
print("SLURM_NTASKS: ", slurm_ntasks)
print("SLURM_PROCID: ", slurm_procid)
print("SLURM_TASK_PID: ", slurm_task_pid)

# Open image according to slurm proc id
index = int(slurm_procid) - 1
files_dir = glob.glob(os.path.join(out_dir, "ASA_*"))
# print(files_dir)
print(len(files_dir))
# print("Index of image: ", index)
fname = files_dir[index]
fname = fname.split("/")[-1]
print("Image: ", fname)
#
## Data
plot_image_and_mask(fname)
