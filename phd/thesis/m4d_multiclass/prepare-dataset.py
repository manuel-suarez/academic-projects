import os
import numpy as np
from tqdm import tqdm
from itertools import product
from skimage.io import imread, imsave

classes = {"sea surface": 0, "oil spill": 1, "look-alike": 2, "ship": 3, "land": 4}

home_dir = os.path.expanduser("~")
src_dir = os.path.join(home_dir, "data", "krestenitis_v0")
dst_dir = os.path.join(home_dir, "data", "krestenitis_v4")

patch_size = 320
for dname in ["train", "test"]:
    dataset_dir = os.path.join(src_dir, dname)
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    labels_1D_dir = os.path.join(dataset_dir, "labels_1D")

    os.makedirs(os.path.join(dst_dir, dname, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, dname, "labels"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, dname, "labels_1D"), exist_ok=True)

    images_ids = os.listdir(images_dir)

    # Open images-labels and divide in patches of 160x160 keeping only the patches where
    # at least exists 1 pixel of oil spill
    print(f"Patchifying {dname} images")
    for image_id in tqdm(images_ids):
        # Open label
        image_fname = os.path.join(images_dir, image_id.split(".")[0] + ".jpg")
        label_fname = os.path.join(labels_dir, image_id.split(".")[0] + ".png")
        label_1D_fname = os.path.join(labels_1D_dir, image_id.split(".")[0] + ".png")
        image = imread(image_fname)
        label = imread(label_fname)
        label_1D = imread(label_1D_fname)

        for index, (i, j) in enumerate(
            product(
                range(image.shape[0] // patch_size),
                range(image.shape[1] // patch_size),
            )
        ):
            ii = i * (patch_size)
            jj = j * (patch_size)
            if ii + patch_size > image.shape[0]:
                ii = ii - (image.shape[0] - (ii + patch_size))
            if jj + patch_size > image.shape[1]:
                jj = jj - (image.shape[1] - (jj + patch_size))

            # print(index, ii, jj)
            # Get label patch
            label_patch = label[ii : ii + patch_size, jj : jj + patch_size]
            label_1D_patch = label_1D[ii : ii + patch_size, jj : jj + patch_size]
            # If has at least one oil spill pixel we get the image patch and save
            # if np.sum(label_patch == classes["oil spill"]) > 0:
            # We will use all the patches

            # Get image patch
            image_patch = image[ii : ii + patch_size, jj : jj + patch_size, :]

            # Check label and patch dimensions
            if label_patch.shape != (patch_size, patch_size, 3):
                print("Error on label shape: ", label_patch.shape)
                exit(0)
            if label_1D_patch.shape != (patch_size, patch_size):
                print("Error on label 1D shape: ", label_1D_patch.shape)
                exit(0)
            if image_patch.shape != (patch_size, patch_size, 3):
                print("Error on image shape: ", image_patch.shape)
                exit(0)

            # Save image and label patches
            imsave(
                os.path.join(
                    dst_dir,
                    dname,
                    "images",
                    image_id.split(".")[0] + f"_{index+1}.jpg",
                ),
                image_patch,
                check_contrast=False,
            )
            imsave(
                os.path.join(
                    dst_dir,
                    dname,
                    "labels",
                    image_id.split(".")[0] + f"_{index+1}.png",
                ),
                label_patch,
                check_contrast=False,
            )
            imsave(
                os.path.join(
                    dst_dir,
                    dname,
                    "labels_1D",
                    image_id.split(".")[0] + f"_{index+1}.png",
                ),
                label_1D_patch,
                check_contrast=False,
            )

    # images_to_erase = 11 if dname == "train" else 4
    # print("Erase images: ", images_to_erase)
    ## Now we delete 11 images from train and 4 from test to get same size dataset from CNN hybrid networks
    ## We are reading the file names and get its pixel oil spill count
    # images_ids = os.listdir(os.path.join(dst_dir, dname, "images"))
    ## Load images and weights
    # images_weights = {}
    # print("Loading weights")
    # for image_id in tqdm(images_ids):
    #    image_fname = os.path.join(
    #        dst_dir, dname, "images", image_id.split(".")[0] + ".jpg"
    #    )
    #    label_fname = os.path.join(
    #        dst_dir, dname, "labels_1D", image_id.split(".")[0] + ".png"
    #    )
    #    image = imread(image_fname)
    #    label = imread(label_fname)
    #    images_weights[image_id] = np.sum(label == classes["oil spill"])
    ## Get minimum images and delete from disk and from dictionary
    # while images_to_erase > 0:
    #    images = []
    #    weights = []
    #    for image_key in images_weights:
    #        images.append(image_key)
    #        weights.append(images_weights[image_key])
    #    idx_to_erase = np.argmax(np.array(weights))
    #    image_id_to_erase = images[idx_to_erase]

    #    # Delete from dictionary
    #    del images_weights[image_id_to_erase]
    #    # Delete from filesystem
    #    os.remove(
    #        os.path.join(
    #            dst_dir, dname, "images", image_id_to_erase.split(".")[0] + ".jpg"
    #        )
    #    )
    #    os.remove(
    #        os.path.join(
    #            dst_dir, dname, "labels_1D", image_id_to_erase.split(".")[0] + ".png"
    #        )
    #    )

    #    images_to_erase -= 1
