import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import os 
from skimage.io import imread

base_path = os.path.expanduser("~")
data_path = os.path.join(base_path, "challenge", "dataset")
labels_path = os.path.join(data_path, "training_noisy_labels")

# Rutas a imágenes de ejemplo
name_image = "2_16_51_0_0.png"
mask_path = os.path.join(labels_path, name_image)
mask = imread(mask_path)
# Etiquetar los componentes conectados
labels = measure.label(mask, connectivity=1)  # Para 4-conectividad
# Si prefieres 8-conectividad, usa connectivity=2

# Obtener el número de componentes conectados
num_components = np.max(labels)

print(f"Número de componentes conectados: {num_components}")

# Obtener propiedades de los componentes conectados
properties = measure.regionprops(labels)

# Opcional: Mostrar áreas de cada componente
areas = [prop.area for prop in properties]
print(f"Áreas de los componentes conectados: {areas}")

# Definir un umbral mínimo de área
min_area = 50  # Ajusta este valor según tu caso

# Lista para almacenar áreas de componentes pequeños
small_components = []

for prop in properties:
    if prop.area < min_area:
        small_components.append(prop)

num_small_components = len(small_components)
print(f"Número de componentes pequeños (posible ruido): {num_small_components}")