import os
import rasterio
from rasterio.transform import rowcol
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

# Configuración de rutas
base_path = os.path.expanduser("~")
data_path = os.path.join(base_path, "data", "cimat", "dataset-noaa")
ruta_imagenes = os.path.join(data_path, "sentinel1", "TIFF")
ruta_csv = os.path.join(data_path, "noaa_sentinel1_products.csv")
output_dir = os.path.join(data_path, "sentinel1", "TIFF_OP")
os.makedirs(output_dir, exist_ok=True)  # Crear la carpeta si no existe

# Leer las coordenadas del archivo CSV con GeoPandas
df = pd.read_csv(ruta_csv)
gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]))
gdf = gdf.dropna(subset=["products"])  # Filtrar filas sin productos
gdf["lon"] = gdf.geometry.x
gdf["lat"] = gdf.geometry.y

# Listar imágenes en la carpeta
imagenes = [
    os.path.basename(f) for f in os.listdir(ruta_imagenes) if f.endswith(".tif")
]
if not imagenes:
    raise FileNotFoundError("No se encontraron imágenes TIFF en la ruta especificada.")


# Extraer identificadores clave de productos
def extraer_identificador(nombre_producto):
    match = re.search(r"\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_\w+", nombre_producto)
    return match.group(0) if match else None


# Crear diccionario de coincidencias
print("Creando diccionario de coincidencias")
coincidencias = {}
for idx, row in gdf.iterrows():
    productos = row["products"].strip("[]").split(", ")
    print(productos)
    for producto in productos:
        identificador = extraer_identificador(producto)
        if identificador:
            for imagen in imagenes:
                if identificador in imagen:
                    coincidencias[row["products"]] = imagen
                    break

# Procesar todas las imágenes
print("Procesando imágenes")
for idx, row in tqdm(gdf.iterrows()):
    producto = row["products"]
    if producto not in coincidencias:
        continue  # Saltar si no hay coincidencia

    ruta_imagen = os.path.join(ruta_imagenes, coincidencias[producto])
    lat, lon = row["lat"], row["lon"]

    with rasterio.open(ruta_imagen) as dataset:
        fila, columna = rowcol(dataset.transform, lon, lat)
        datos = dataset.read()

        # Validar si el píxel está dentro de los límites
        if 0 <= fila < datos.shape[1] and 0 <= columna < datos.shape[2]:
            # Crear copia y resaltar el píxel
            datos_copia = datos.copy()
            if datos_copia.shape[0] >= 3:
                datos_copia[0, fila, columna] = 255
                datos_copia[1, fila, columna] = 0
                datos_copia[2, fila, columna] = 0
            else:
                datos_copia[0, fila, columna] = datos_copia[0].max()

            # Guardar la imagen resaltada en formato PNG
            output_ruta = os.path.join(
                output_dir,
                os.path.basename(ruta_imagen).replace(".tif", "_resaltado.png"),
            )
            plt.figure(figsize=(10, 10))
            if datos.shape[0] == 1:
                plt.imshow(datos_copia[0], cmap="gray")
            else:
                plt.imshow(datos_copia[:3].transpose(1, 2, 0))
            plt.scatter([columna], [fila], color="red", s=50)
            plt.axis("off")
            plt.savefig(output_ruta, dpi=300, bbox_inches="tight")
            plt.close()

print("Procesamiento completado. Todas las imágenes resaltadas han sido guardadas.")
