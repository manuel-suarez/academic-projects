import os
import shutil
import numpy as np
from skimage.io import imread, imsave
import albumentations as A
import matplotlib.pyplot as plt
from tqdm import tqdm


def to_uint8(img):
    """
    Convierte imagen float (0-1) o float32/64 a uint8.
    """
    if img.dtype in [np.float32, np.float64]:
        return np.clip(img * 255, 0, 255).astype(np.uint8)
    return img.astype(np.uint8)


def compute_dataset_statistics(dataset_dir):
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels_1D")
    total_petroleum = 0
    total_background = 0
    image_files = sorted(os.listdir(images_dir))
    stats = {}
    for filename in image_files:
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".png")
        label = imread(label_path, as_gray=True)
        petroleum_pixels = np.sum(label == 1)
        total_pixels = label.size
        background_pixels = total_pixels - petroleum_pixels
        total_petroleum += petroleum_pixels
        total_background += background_pixels
        stats[filename] = {
            "petroleum_pixels": petroleum_pixels,
            "background_pixels": background_pixels,
            "petroleum_ratio": petroleum_pixels / total_pixels,
        }
    print("=== Estadísticas del Dataset ===")
    print("Total de imágenes:", len(image_files))
    print("Total de píxeles de petróleo:", total_petroleum)
    print("Total de píxeles de fondo:", total_background)
    avg_ratio = total_petroleum / (total_petroleum + total_background)
    print("Ratio promedio de petróleo:", avg_ratio)
    return stats


# Binary is for binary oil spill classification (all other classes will be converted to 0-background)
def apply_augmentation(
    dataset_dir, output_dir, binary=False, threshold=0.05, aug_pipeline=None
):
    if aug_pipeline is None:
        aug_pipeline = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=1.0),
                A.RandomBrightnessContrast(p=0.2),
            ]
        )
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    output_labels_1D_dir = os.path.join(output_dir, "labels_1D")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(output_labels_1D_dir, exist_ok=True)

    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    labels_1D_dir = os.path.join(dataset_dir, "labels_1D")
    image_files = sorted(os.listdir(images_dir))

    augmented_examples = []

    for filename in tqdm(image_files):
        # print("Processing file: ", filename)
        image_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".png")
        label_1D_path = os.path.join(
            labels_1D_dir, os.path.splitext(filename)[0] + ".png"
        )
        image = imread(image_path)
        # print("Image shape: ", image.shape)
        label = imread(label_path)
        # print("Label shape: ", label.shape)
        label_1D = imread(label_1D_path)
        # print("Label 1D shape: ", label_1D.shape)
        # print(
        #    f"label : {label.dtype}, {label.min()}, {label.max()}\nimage : {image.dtype}, {image.min()}, {image.max()}"
        # )
        # label = (label == 1).astype(np.int64)

        # Copy original image and label
        shutil.copy(
            os.path.join(images_dir, filename),
            os.path.join(output_images_dir, filename),
        )
        shutil.copy(
            os.path.join(labels_dir, os.path.splitext(filename)[0] + ".png"),
            os.path.join(output_labels_dir, os.path.splitext(filename)[0] + ".png"),
        )
        shutil.copy(
            os.path.join(labels_1D_dir, os.path.splitext(filename)[0] + ".png"),
            os.path.join(output_labels_1D_dir, os.path.splitext(filename)[0] + ".png"),
        )

        petroleum_pixels = np.sum(label_1D == 1)
        total_pixels = label.size
        ratio = petroleum_pixels / total_pixels

        if ratio >= threshold:
            masks = [label[:, :, 0], label[:, :, 1], label[:, :, 2], label_1D]
            augmented = aug_pipeline(image=image, masks=masks)
            aug_image = augmented["image"]
            aug_label_R, aug_label_G, aug_label_B, aug_label_1D = augmented["masks"]
            aug_label = np.stack([aug_label_R, aug_label_G, aug_label_B], axis=-1)
            #    print("Aug label shape: ", aug_label.shape)

            aug_filename = "aug_" + filename
            aug_mask_filename = os.path.splitext(aug_filename)[0] + ".png"

            imsave(
                os.path.join(output_images_dir, aug_filename),
                aug_image,
                check_contrast=False,
            )
            imsave(
                os.path.join(output_labels_dir, aug_mask_filename),
                aug_label,
                check_contrast=False,
            )
            imsave(
                os.path.join(output_labels_1D_dir, aug_mask_filename),
                aug_label_1D,
                check_contrast=False,
            )

            if len(augmented_examples) < 5:
                augmented_examples.append(
                    ((image, label, label_1D), (aug_image, aug_label, aug_label_1D))
                )
    return augmented_examples


def visualize_examples(examples, output_dir):
    num_examples = len(examples)
    print(num_examples)
    # plt.figure(figsize=(10, 4 * num_examples))
    fig, axs = plt.subplots(num_examples, 6, figsize=(20, 4 * num_examples))
    for i, (
        (image, label, label_1D),
        (aug_image, aug_label, aug_label_1D),
    ) in enumerate(examples):
        axs[i, 0].imshow(image, cmap="gray")
        axs[i, 0].set_title(f"Imagen {i+1}")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(label)
        axs[i, 1].set_title(f"Label {i+1}")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(label_1D, cmap="gray")
        axs[i, 2].set_title(f"Label 1D {i+1}")
        axs[i, 2].axis("off")

        axs[i, 3].imshow(aug_image, cmap="gray")
        axs[i, 3].set_title(f"Augmented Imagen {i+1}")
        axs[i, 3].axis("off")

        axs[i, 4].imshow(aug_label)
        axs[i, 4].set_title(f"Augmented Label {i+1}")
        axs[i, 4].axis("off")

        axs[i, 5].imshow(aug_label_1D, cmap="gray")
        axs[i, 5].set_title(f"Augmented Label 1D {i+1}")
        axs[i, 5].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir), exist_ok=True)
    comp_path = os.path.join(output_dir, "augmentation_examples.png")
    plt.savefig(comp_path)
    plt.close()


dataset_version = "v4"


def main():
    base_dir = os.path.expanduser("~")
    data_dir = os.path.join(base_dir, "data")
    for dataset_dir in ["train", "test"]:
        input_dir = os.path.join(
            data_dir, f"krestenitis_{dataset_version}/{dataset_dir}"
        )
        output_dir = os.path.join(
            data_dir, f"krestenitis_{dataset_version}_augmented/{dataset_dir}"
        )
        threshold = 0.01  # 5% de petróleo mínimo para aplicar augmentación

        os.makedirs(output_dir, exist_ok=True)

        print("Estadísticas del dataset original:")
        original_stats = compute_dataset_statistics(input_dir)

        augmented_examples = apply_augmentation(
            input_dir, output_dir, threshold=threshold
        )

        if augmented_examples:
            visualize_examples(augmented_examples, output_dir)
        else:
            print(
                "No se encontraron imágenes que cumplan el umbral para aplicar augmentación."
            )

        print("Estadísticas del nuevo dataset (con aumentaciones):")
        new_stats = compute_dataset_statistics(output_dir)

        print(f"Nuevo dataset guardado en: {output_dir}")


if __name__ == "__main__":
    main()
