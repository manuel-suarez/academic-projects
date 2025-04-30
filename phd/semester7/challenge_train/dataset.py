import os
import torch
from skimage.io import imread
from torch.utils.data import Dataset

# features_dirs = os.listdir(feat_path)
features_dirs = [
    "rgb-channels-red_patches",  # 0
    "rgb-channels-green_patches",  # 1
    "rgb-channels-blue_patches",  # 2
    "rgf-channels-red_patches",  # 3
    "rgf-channels-green_patches",  # 4
    "rgf-channels-blue_patches",  # 5
    "rgb-threshold-red_patches",  # 6
    "rgb-threshold-green_patches",  # 7
    "rgb-threshold-blue_patches",  # 8
    "texture_contrast_patches",  # 9
    "texture_correlation_patches",  # 10
    "texture_homogeneity_patches",  # 11
    "texture_energy_patches",  # 12
    "variance_patches",  # 13
    "wavelets_haar_patches",  # 14
    "wavelets_bior1.1_patches",  # 15
]


# Dataset personalizado
class SatelliteDataset(Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        base_dir,
        feat_dirs,
        max_images=None,
        return_names=False,
    ):
        # Directorio de los parches originales de las imágenes, los nombres se extraeran de aquí
        self.img_dir = img_dir
        # Directorio de las etiquetas
        self.mask_dir = mask_dir
        # Directorio base de las características para conformar los canales de entrada
        self.base_dir = base_dir
        # Lista con los nombres de los directorios que representan los features a cargar para los canales de entrada
        self.feat_dirs = feat_dirs
        # Si max_images se asigna entonces solo cargamos el número correspondiente (o hasta el número de imágenes disponibles en el directorio)
        listdir = os.listdir(img_dir)
        num_images = (
            len(listdir) if max_images == None else min(max_images, len(listdir))
        )
        listimgs = listdir[:num_images]

        # Solo incluir imágenes que tienen una máscara correspondiente (extraemos solo el nombre ya que las características se han guardado en formato .tif)
        self.images = [
            img.split(".")[0]
            for img in listimgs
            if os.path.exists(os.path.join(mask_dir, img))
        ]
        # Regresar nombre de imagen en iterador para efectos del ranking (debe desactivarse en el entrenamiento)
        self.return_names = return_names

        # Verificar si la lista de imágenes está vacía
        if len(self.images) == 0:
            raise ValueError(
                "No se encontraron imágenes con máscaras correspondientes. Verifica las rutas y los archivos."
            )
        # Verificar que el tamaño del dataset sea de 5000
        if len(self.images) != num_images:
            raise ValueError(
                f"El tamaño del dataset debería ser de {num_images} imágenes"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Imágen en canales de acuerdo a la lista de directorios con los features a cargar
        features = torch.stack(
            [
                torch.from_numpy(
                    imread(
                        os.path.join(
                            self.base_dir, feat_dir, self.images[idx] + ".tif"
                        ),
                        as_gray=True,
                    )
                ).float()
                for feat_dir in self.feat_dirs
            ]
        )

        # Creamos imagen de n canales de acuerdo a las características proporcionadas
        mask_path = os.path.join(self.mask_dir, self.images[idx] + ".png")
        mask = (
            torch.from_numpy(imread(mask_path, as_gray=True) / 255.0)
            .unsqueeze(0)
            .float()
        )

        return (
            (features, mask, self.images[idx] + ".png")
            if self.return_names
            else (features, mask)
        )
