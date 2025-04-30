import os
import numpy as np
from PIL import Image
# from tqdm import tqdm
import logging
# import matplotlib.pyplot as plt

# Ruta local
# root_folder = r"C:\Users\masua\Downloads\Cimat\Nyu v2"
# Ruta del clúster
root_folder = "/home/est_posgrado_manuel.suarez/data"
base_dir = "NYU_GT"
clean_dir = "NYU_CL"
under_dir = "NYU_underwater"
target_dir = "type1_data/underwater_type_1"

# Type of submarine ambient
type_I = [0.85, 0.961, 0.982]
type_IA = [0.84, 0.955, 0.975]
type_IB = [0.83, 0.95, 0.968]
type_II = [0.80, 0.925, 0.94]
type_III = [0.75, 0.885, 0.89]
type_1 = [0.75, 0.885, 0.875]
type_3 = [0.71, 0.82, 0.8]
type_5 = [0.67, 0.73, 0.67]
type_7 = [0.62, 0.61, 0.5]
type_9 = [0.55, 0.46, 0.29]

TYPE = type_1

# Parámetros aleatorios de profundidad y horizonte
# deep = 5 - 2 * np.random.rand()
# horization = 15 - 14.5 * np.random.rand()

# Obtenemos lista de archivos del directorio destino
tfiles = os.listdir(os.path.join(root_folder, target_dir))
tfiles.sort(key = lambda x: int(x.split('_')[0]))
# print(tfiles[:10])

# Traverse all files in directory
for num_image in range(1, 1450):
    # Conform image and depth filenames
    fname = f"{num_image}_Image_.bmp"
    dname = f"{num_image}_Depth_.bmp"
    tname = tfiles[num_image - 1]

    # Open images
    image = np.array(Image.open(os.path.join(root_folder, base_dir, fname)))
    depth = np.array(Image.open(os.path.join(root_folder, base_dir, dname)))
    logging.info(f"Image: {image.shape}, {image.dtype}, {np.max(image)}, {np.min(image)}")
    logging.info(f"Depth: {depth.shape}, {depth.dtype}, {np.max(depth)}, {np.min(depth)}")
    # Remove blank edges
    image0 = image[10:-10, 10:-10, :] / 255
    depth0 = depth[10:-10, 10:-10] / 255
    # print("Image0: ", image0.shape, image0.dtype, np.max(image0), np.min(image0))
    # print("Depth0: ", depth0.shape, depth0.dtype, np.max(depth0), np.min(depth0))

    # Leemos los valores de los parámetros del directorio de ejemplo para verificar si
    # obtenemos los mismos resultados
    deep = float(tname.split('_')[2])
    horization = float(tname.split('_')[4])

    # Término beta
    A = 1.5 * np.array(TYPE)**deep
    # print("A", A.shape, A.dtype)

    # Término t
    t = np.zeros(image0.shape)
    # t = np.array(TYPE)**(depth0 * horization)
    t[:, :, 0] = TYPE[0]**(depth0 * horization)
    t[:, :, 1] = TYPE[1]**(depth0 * horization)
    t[:, :, 2] = TYPE[2]**(depth0 * horization)
    # print("T", t.shape, t.dtype)

    # print("T1", t1.shape, t1.dtype)
    # print("T2", t2.shape, t2.dtype)
    # print("T3", t3.shape, t3.dtype)

    # Underwater image
    # I = np.zeros(image0.shape)
    # I[:, :, 0] = image0[:, :, 0] * t[:, :, 0] + (1 - t[:, :, 0]) * A[0]
    # I[:, :, 1] = image0[:, :, 1] * t[:, :, 1] + (1 - t[:, :, 1]) * A[1]
    # I[:, :, 2] = image0[:, :, 2] * t[:, :, 2] + (1 - t[:, :, 2]) * A[2]
    I = A * image0 * t + (1 - t) * A
    # print(num_image)
    # print("I", I.shape, I.dtype, np.min(I), np.max(I))
    # plt.figure()
    #plt.imshow(I)

    # Guardamos imagen
    clean_img = Image.fromarray((image0 * 255).astype(np.uint8))
    cname = f"{num_image}_Image_deep_{deep}_horiz_{horization}_.bmp"
    clean_img.save(os.path.join(root_folder, under_dir, cname))
    underwater_img = Image.fromarray((np.clip(I, 0, 1) * 255).astype(np.uint8))
    uname = f"{num_image}_Underwater_deep_{deep}_horiz_{horization}_.bmp"
    underwater_img.save(os.path.join(root_folder, under_dir, uname))

print("Done!")