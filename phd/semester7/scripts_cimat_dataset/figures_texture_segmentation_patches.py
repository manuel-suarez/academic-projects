import os
import argparse
from glob import glob
from skimage.io import imread
from matplotlib import pyplot as plt

# Path configuration
home_path = os.path.expanduser("~")
data_path = os.path.join(home_path, "data", "cimat")
feat_path = os.path.join(data_path, "dataset-cimat", "segmentation", "features")
label_path = os.path.join(data_path, "dataset-cimat", "segmentation", "labels")
figures_path = os.path.join(
    data_path, "dataset-cimat", "segmentation", "figures", "texture"
)

parser = argparse.ArgumentParser(
    prog="FiguresTexturePatches", description="Generate figures for texture patches"
)
parser.add_argument("--filename")
args = parser.parse_args()
# print(args)

# Traverse features and label path to get patches and generate figures
fname = args.filename.split(".")[0]

# Get image and label patches
image_patches = glob(os.path.join(feat_path, "origin", f"{fname}_*.tif"))
label_patches = glob(os.path.join(label_path, f"{fname}_*.png"))
print(f"Image name: {fname}")
print(f"Image patches: {len(image_patches)}, label patches: {len(label_patches)}")
if len(image_patches) != len(label_patches):
    print("Error on patches num")
# Get texture patches
for texture_path in os.listdir(os.path.join(feat_path, "texture")):
    texture_patches = glob(
        os.path.join(feat_path, "texture", texture_path, f"{fname}_*.tif")
    )
    print(f"Texture {texture_path} patches: {len(texture_patches)}")
    if len(image_patches) != len(texture_patches):
        print(f"Error on texture {texture_path} patches num")

# Num of patches are ok now we proceed to traverse all image patches to verify that the
# corresponding label and texture patches exists
for image_patch in image_patches:
    # Extract patch filename
    patch_name = image_patch.split("/")[-1]
    print(patch_name)
    # Check if label and texture features exists
    if not os.path.exists(os.path.join(label_path, patch_name.split(".")[0] + ".png")):
        print(f"Label doesn't exists for patch name: {patch_name}")
    for texture_path in os.listdir(os.path.join(feat_path, "texture")):
        if not os.path.exists(
            os.path.join(feat_path, "texture", texture_path, patch_name)
        ):
            print(
                f"Texture {texture_path} image doesn't exists for patch name: {patch_name}"
            )

    # Now generate figure with image, label and texture patches

    # Save figure with image and mask patches
    fig, ax = plt.subplots(2, 6, figsize=(12, 8))
    image = imread(image_patch, as_gray=True)
    ax[0, 0].imshow(image, cmap="gray")
    ax[0, 0].set_title("Image patch")
    label = imread(
        os.path.join(label_path, patch_name.split(".")[0] + ".png"), as_gray=True
    )
    ax[1, 0].imshow(label, cmap="gray")
    ax[1, 0].set_title("Mask patch")
    for index, texture_path in enumerate(
        os.listdir(os.path.join(feat_path, "texture"))
    ):
        texture = imread(
            os.path.join(feat_path, "texture", texture_path, patch_name), as_gray=True
        )
        ax[index % 2, index % 5 + 1].imshow(texture, cmap="gray")
        ax[index % 2, index % 5 + 1].set_title(f"Texture {texture_path}")
    fig.tight_layout()
    fig.suptitle(patch_name)
    plt.savefig(os.path.join(figures_path, patch_name.split(".")[0] + ".png"))
    plt.close()

print("Done!")
