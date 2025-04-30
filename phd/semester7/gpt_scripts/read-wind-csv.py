import os
import rasterio
import numpy as np
import geopandas as gpd

from skimage.io import imsave
from rasterio.features import rasterize
from scipy.interpolate import griddata
from matplotlib import pyplot as plt

fname = "ASA_IMP_1PNESA20050425_075225_000000182036_00393_16479_0000"

base_path = os.path.expanduser("~")
data_path = os.path.join(base_path, "data", "cimat", "dataset-cimat", "temp4")
prod_path = os.path.join(data_path, fname)
tiff_path = os.path.join(data_path, fname, "image.tif")

wind_path = os.path.join(prod_path, "wind_07.data", "vector_data")
wind_input_file = os.path.join(wind_path, "WindField.csv")
wind_output_file = os.path.join(wind_path, "WindField_2.csv")

# Open TIFF data and get shape
print("Open TIFF file")
with rasterio.open(tiff_path) as tiff_file:
    sar_image = tiff_file.read(1)
    sar_transform = tiff_file.transform
    sar_shape = sar_image.shape

print("SAR image shape: ", sar_image.shape, sar_image.dtype)

# Open file to remove # characters and :Datatype marks
print("Open WindField CSV")
with open(wind_input_file, "r") as input_file, open(
    wind_output_file, "w"
) as output_file:
    for index, line in enumerate(input_file):
        if line.startswith("#"):
            continue
        if index == 3:
            line = (
                line.replace(":String", "").replace(":Double", "").replace(":Point", "")
            )
        output_file.write(line)

# Open wind CSV file
wind_gdf = gpd.read_file(wind_output_file)

# Extract wind field points and values
print("Get speed and coordinates values")
points = gpd.GeoSeries.from_wkt(wind_gdf["geometry"])
points_values = np.array([(geom.x, geom.y) for geom in points])
speed_values = wind_gdf["speed"].values.astype(np.float32)
print("speed_values datatype: ", speed_values.shape, speed_values.dtype)

# Create interpolation grid (for SAR image)
print("Create interpolation grid")
x_coords = np.arange(0, sar_shape[1])
y_coords = np.arange(0, sar_shape[0])
x_grid, y_grid = np.meshgrid(x_coords, y_coords)
lon, lat = rasterio.transform.xy(sar_transform, y_grid, x_grid, offset="center")

grid_points = np.column_stack([lon.flatten(), lat.flatten()])

# Interpolate wind field values onto the SAR grid
print("Interpolate wind data")
wind_raster = rasterize(
    zip(points, speed_values),
    out_shape=(sar_shape[0], sar_shape[1]),
    transform=sar_transform,
    fill=0,
)
w_min = np.min(wind_raster)
w_max = np.max(wind_raster)
print(
    "wind raster result: ",
    wind_raster.shape,
    wind_raster.dtype,
    w_min,
    w_max,
)

# Save wind field image
print("Save windfield output image")
imsave(os.path.join(data_path, fname, "wind.tif"), wind_raster)

# Make figure
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].imshow(sar_image, cmap="gray")
axs[1].imshow(wind_raster, cmap="viridis", vmin=w_min, vmax=w_max)
plt.savefig(os.path.join(data_path, fname, "figure.png"))
plt.close()

print("Done!")
