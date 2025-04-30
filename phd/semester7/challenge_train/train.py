import os
import time
import shutil
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import SatelliteDataset, features_dirs

base_jl_path = "/home/est_posgrado_jose.deleon/Data-Centri-Land-Cover-Challenge/data-centric-land-cover-classification-challenge/dataset"
base_ml_path = "/home/est_posgrado_manuel.suarez/data"

labels_path = os.path.join(base_jl_path, "training_noisy_labels")
patches_path = os.path.join(base_jl_path, "training_patches")
feat_path = os.path.join(base_ml_path, "challenge-features", "dataset")


# Definimos el modelo U-Net con un backbone preentrenado (ResNet)
arch_name = "unet"
encoder_name = "senet154"
epochs = 20
print("Arch+decoder+epochs: ", arch_name, encoder_name, epochs)

# Leemos alguna configuración disponible en el directorio base de canales
channels_evaluate_path = os.path.join("channels", "evaluate")
channels_trained_path = os.path.join("channels", "trained")
channels_training_path = os.path.join("channels", "training")
# Obtenemos de forma aleatoria una combinación de canales
slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
slurm_array_task_id = 1 if slurm_array_task_id == None else int(slurm_array_task_id)
print("SLURM_ARRAY_TASK_ID: ", slurm_array_task_id)

list_channels_configurations = os.listdir(channels_evaluate_path)
len_channels_configurations = len(list_channels_configurations)
channels_configuration = list_channels_configurations[slurm_array_task_id - 1]
# Movemos el archivo al directorio de entrenamiento para evitar que otro proceso lo tome
shutil.copy(
    os.path.join(channels_evaluate_path, channels_configuration),
    os.path.join(channels_training_path, channels_configuration),
)
# Convertimos a enteros
print(channels_configuration, channels_configuration.split("."))
features_indexes = [int(idx) for idx in channels_configuration.split(".")]
features_channels = [features_dirs[idx] for idx in features_indexes]
print("Features to use: ", features_indexes, features_channels)

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Configuración del modelo
model = smp.Unet(
    encoder_name=encoder_name,  # Cambia si prefieres otro backbone
    encoder_weights=None,
    # We need to use in_channels to len of features_channels list
    in_channels=len(features_channels),
    classes=1,
).to(device)


# Crear DataLoader
# features_to_use = features_dirs[0:3]
# print("Features to use: ", features_to_use)
dataset = SatelliteDataset(
    img_dir=patches_path,
    mask_dir=labels_path,
    base_dir=feat_path,
    feat_dirs=features_channels,
    max_images=160,
)
train_data, valid_data = random_split(dataset, [0.7, 0.3])
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=4, shuffle=False)

# Configurar optimizador y pérdida
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Entrenamiento
# epochs = 50
os.makedirs("results", exist_ok=True)
print("Inicio de entrenamiento")
startTime = time.time()
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    valid_loss = 0.0
    for images, masks in valid_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        valid_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{epochs}, Train Loss: {(train_loss/len(train_loader)):.4f}, Valid Loss: {(valid_loss/len(train_loader))}"
    )
    if (epoch + 1) % 5 == 0:
        torch.save(
            model.state_dict(),
            os.path.join(
                "results",
                f"{arch_name}_{encoder_name}_{channels_configuration}_{epoch+1}_weights.pth",
            ),
        )


endTime = time.time()
print("Entrenamiento finalizado.")
print("[INFO] tiempo de entrenamiento total: {:.2f}s".format(endTime - startTime))
# Movemos el archivo de entrenamiento al directorio correspondiente para indicar que hemos finalizado
shutil.copy(
    os.path.join(channels_training_path, channels_configuration),
    os.path.join(channels_trained_path, channels_configuration),
)
