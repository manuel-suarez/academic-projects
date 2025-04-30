import os
import argparse
from matplotlib import pyplot as plt
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from dataset import SatelliteDataset, features_dirs
from skimage.io import imread

# Definimos el modelo U-Net con un backbone preentrenado (ResNet)
arch_name = "unet"
encoder_name = "senet154"
print("Arch+decoder: ", arch_name, encoder_name)

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


# Función para cargar el modelo con los pesos
def load_model(
    weights_folder, arch_name="unet", encoder_name="resnet152", in_channels=3
):
    archs = {
        "unet": smp.Unet,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "linknet": smp.Linknet,
        "manet": smp.MAnet,
    }

    arch = archs[arch_name]

    model = arch(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
    ).to(device)

    model.load_state_dict(
        torch.load(weights_folder, map_location=device, weights_only=True)
    )

    return model


# Definir las épocas y cargar los pesos de cada una
epochs = [40, 50, 60]


# Función para generar máscara con el modelo cargado
def generate_mask(model, image):
    # Generar máscara
    with torch.no_grad():
        output = model(image)  # Image comes from dataset so it's a tensor already
    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    return mask


# Función para visualizar y guardar comparación
def visualize_comparison(figure_name, image_name, images_path, label, generated_masks):
    print(images_path, image_name, figure_name)
    img = imread(os.path.join(images_path, image_name))
    label = torch.squeeze(label)

    # Crear la figura con subplots para las máscaras generadas en cada época
    fig, axes = plt.subplots(1, len(generated_masks) + 2, figsize=(15, 5))

    # Mostrar la imagen original
    axes[0].imshow(img)
    axes[0].set_title("Imagen Original")
    axes[0].axis("off")

    # Mostrar las máscaras generadas
    for i, (epoch, mask) in enumerate(generated_masks.items()):
        axes[i + 1].imshow(mask, cmap="gray")
        axes[i + 1].set_title(f"Máscara Epoca {epoch}")
        axes[i + 1].axis("off")

    # Mostrar la máscara ruidosa
    axes[-1].imshow(label, cmap="gray")
    axes[-1].set_title("Máscara Ruidosa")
    axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig(figure_name)
    plt.close()


def main(
    arch_name,
    encoder_name,
    channels_configuration,
    weights_folder,
    output_dir,
    images_folder,
    mask_folder,
    base_folder,
    feat_folders,
):
    output_path = os.path.join(
        output_dir,
        arch_name,
        encoder_name,
        channels_configuration,
    )
    os.makedirs(output_path, exist_ok=True)

    # Crear el dataset y dataloader
    dataset = SatelliteDataset(
        img_dir=images_folder,
        mask_dir=mask_folder,
        base_dir=base_folder,
        feat_dirs=feat_folders,
        return_names=True,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Generar máscaras para cada época
    # Ejemplo de nombre para archivo de pesos: unet_senet154_0.1.2.13.14.15_40_weights.pth
    for image, label, image_name in dataloader:
        image, label = image.to(device), label.to(device)
        image_name = image_name[0]

        generated_masks = {}
        for epoch in epochs:
            weights_path = os.path.join(
                weights_folder,
                f"{arch_name}_{encoder_name}_{channels_configuration}_{epoch}_weights.pth",
            )
            # Cargar el modelo
            model = load_model(
                weights_path,
                arch_name=arch_name,
                encoder_name=encoder_name,
                in_channels=len(feat_folders),
            )

            model.load_state_dict(
                torch.load(weights_path, map_location=device, weights_only=True)
            )

            # Generar la máscara para esta época
            generated_masks[epoch] = generate_mask(model, image)

        # Visualizar y guardar comparación
        output_filename = os.path.join(
            output_path, f"{image_name.split('.')[0]}_figure.png"
        )
        label = torch.squeeze(label, 0)
        visualize_comparison(
            output_filename, image_name, images_folder, label, generated_masks
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generación de figuras de segmentación con el modelo entrenado"
    )
    parser.add_argument(
        "weights_folder", type=str, help="Ruta del directorio de pesos del modelo."
    )
    # parser.add_argument(
    #    "weights_file", type=str, help="Nombre del archivo de pesos del modelo."
    # )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Ruta del directorio de salida para las figuras.",
    )
    parser.add_argument(
        "images_folder", type=str, help="Ruta del directorio de imágenes."
    )
    parser.add_argument(
        "mask_folder", type=str, help="Ruta del directorio de máscaras."
    )
    parser.add_argument(
        "base_folder", type=str, help="Ruta base de las características a evaluar."
    )

    # def list_of_strings(arg):
    #    return arg.split(",")

    # parser.add_argument(
    #    "--feat_folders",
    #    type=list_of_strings,
    #    help="Lista de directorios de las características a evaluar.",
    # )

    args = parser.parse_args()
    # if args.feat_folders == None:
    #    print(
    #        "Debe especificar los directorios correspondientes a los canales de entrada: ",
    #        features_dirs,
    #    )
    #    exit(-1)
    print("Argumentos: ", args)

    # Leemos alguna configuración disponible en el directorio base de canales
    channels_evaluate_path = os.path.join("channels", "evaluate")
    channels_trained_path = os.path.join("channels", "trained")
    channels_training_path = os.path.join("channels", "training")
    # Obtenemos de forma aleatoria una combinación de canales
    slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    slurm_array_task_id = 1 if slurm_array_task_id == None else int(slurm_array_task_id)
    print("SLURM_ARRAY_TASK_ID: ", slurm_array_task_id)

    list_channels_configurations = os.listdir(channels_trained_path)
    len_channels_configurations = len(list_channels_configurations)
    channels_configuration = list_channels_configurations[slurm_array_task_id - 1]

    print(channels_configuration, channels_configuration.split("."))
    features_indexes = [int(idx) for idx in channels_configuration.split(".")]
    features_channels = [features_dirs[idx] for idx in features_indexes]
    print("Features to use: ", features_indexes, features_channels)

    main(
        arch_name,
        encoder_name,
        channels_configuration,
        args.weights_folder,
        args.output_dir,
        args.images_folder,
        args.mask_folder,
        args.base_folder,
        features_channels,
    )
