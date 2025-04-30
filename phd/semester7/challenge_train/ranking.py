import os
import argparse
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from skimage.io import imread
import pandas as pd
from dataset import SatelliteDataset, features_dirs


def iou(pred, target, thr_tol=0.5):
    # Calcular umbral binario en base al promedio de predicción
    thr_binary = 2 * np.mean(pred)

    # Aplicar el umbral para convertir 'pred' en una máscara binaria
    pred_binary = (pred > thr_binary).astype(float)

    # Convertir las imágenes a tensores de PyTorch, si es necesario
    if not isinstance(pred, torch.Tensor):
        pred_binary = torch.tensor(pred_binary)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)

    # Convertir a tipo flotante para facilitar la operación
    pred_binary = pred_binary.float()
    target = target.float()

    # Calcular la intersección y unión
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection

    # Retornar el IoU si la unión es mayor que cero, o cero en caso contrario
    return (intersection / union).item() if union > 0 else 0


def dice_coefficient(pred, target, thr_tol=0.5):
    # Calcular umbral binario en base al promedio de predicción
    thr_binary = 2 * np.mean(pred)

    # Aplicar el umbral para convertir 'pred' en una máscara binaria
    pred_binary = (pred > thr_binary).astype(float)

    # Convertir las imágenes a tensores de PyTorch, si es necesario
    if not isinstance(pred, torch.Tensor):
        pred_binary = torch.tensor(pred_binary)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)

    # Convertir a tipo flotante para facilitar la operación
    pred_binary = pred_binary.float()
    target = target.float()

    # Calcular la intersección y el denominador de la fórmula de Dice
    intersection = (pred_binary * target).sum()
    total_pixels = pred_binary.sum() + target.sum()

    # Retornar el coeficiente de Dice si hay píxeles en las máscaras, o cero en caso contrario
    return (2 * intersection / total_pixels).item() if total_pixels > 0 else 0


# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# Función principal
def main(
    weights_folder,
    weights_file,
    output_dir,
    images_folder,
    mask_folder,
    base_folder,
    feat_folders,
):
    # Cargar el modelo
    model = load_model(
        os.path.join(weights_folder, weights_file), in_channels=len(feat_folders)
    )

    # Crear el dataset y dataloader
    dataset = SatelliteDataset(
        img_dir=images_folder,
        mask_dir=mask_folder,
        base_dir=base_folder,
        feat_dirs=feat_folders,
        return_names=True,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Almacenar resultados de IoU y Dice junto con el nombre de la imagen
    results = []

    model.eval()
    print("Empieza evaluación")
    with torch.no_grad():
        for images, labels, img_name in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Obtener predicciones de la U-Net
            outputs = model(images)
            outputs = torch.sigmoid(outputs).squeeze().cpu().numpy()

            # Calcular IoU y Dice
            iou_score = iou(outputs[0], labels[0])
            dice_score = dice_coefficient(outputs[0], labels[0])

            # Guardar los resultados en el diccionario
            results.append(
                {
                    "image_name": img_name[0],  # Nombre de la imagen (batch_size = 1)
                    "iou_score": iou_score,
                    "dice_score": dice_score,
                }
            )

    # Crear un DataFrame con los resultados
    df = pd.DataFrame(results)
    print(df.columns)

    # Ordenar el DataFrame por IoU y Dice de menor a mayor
    df_iou_ranked = df.sort_values(by="iou_score", ascending=False).reset_index(
        drop=True
    )
    df_dice_ranked = df.sort_values(by="dice_score", ascending=False).reset_index(
        drop=True
    )

    df_iou_ranked["id"] = df_iou_ranked.index
    df_dice_ranked["id"] = df_dice_ranked.index

    # Select only the 'id' and 'Image' columns and rename them
    df_output_iou = df_iou_ranked[["id", "image_name"]].rename(
        columns={"image_name": "imageid"}
    )

    df_output_dice = df_dice_ranked[["id", "image_name"]].rename(
        columns={"image_name": "imageid"}
    )

    # Guardar el DataFrame como CSV
    os.makedirs(output_dir, exist_ok=True)
    output_file_iou = os.path.join(output_dir, f"results_{weights_file}_iou.csv")
    df_output_iou.to_csv(output_file_iou, index=False)

    output_file_dice = os.path.join(output_dir, f"results_{weights_file}_dice.csv")
    df_output_dice.to_csv(output_file_dice, index=False)

    # print(f"Resultados guardados en: {output_file}")


# Si el script se ejecuta directamente
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación del modelo con IoU y Dice."
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
        help="Ruta del directorio de salida para los resultados CSV.",
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

    weights_file = f"unet_resnet152_{channels_configuration}_40_weights.pth"
    main(
        args.weights_folder,
        weights_file,
        args.output_dir,
        args.images_folder,
        args.mask_folder,
        args.base_folder,
        features_channels,
    )
    weights_file = f"unet_resnet152_{channels_configuration}_50_weights.pth"
    main(
        args.weights_folder,
        weights_file,
        args.output_dir,
        args.images_folder,
        args.mask_folder,
        args.base_folder,
        features_channels,
    )
    weights_file = f"unet_resnet152_{channels_configuration}_60_weights.pth"
    main(
        args.weights_folder,
        weights_file,
        args.output_dir,
        args.images_folder,
        args.mask_folder,
        args.base_folder,
        features_channels,
    )
