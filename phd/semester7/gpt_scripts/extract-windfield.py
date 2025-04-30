import os
import sys
import rasterio
import argparse
import numpy as np
import geopandas as gpd

from skimage.io import imsave
from rasterio.features import rasterize
from scipy.interpolate import griddata
from matplotlib import pyplot as plt

from osgeo import ogr, osr

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, required=True, help="product file name")
args = parser.parse_args()
print("Args: ", args)

fname = args.fname

base_path = os.path.expanduser("~")
data_path = os.path.join(base_path, "data", "cimat", "dataset-tarso", "temp4", fname)
# output_path = os.path.join(base_path, "data", "cimat", "dataset-tarso", "wind", fname)
# os.makedirs(output_path, exist_ok=True)
# prod_path = os.path.join(data_path, fname)

# Wind field vector data
wind_path = os.path.join(data_path, "wind_07.data", "vector_data")
wind_input_file = os.path.join(wind_path, "WindField.csv")
wind_output_file = os.path.join(wind_path, "WindField_2.csv")

# Input and output file paths
output_shapefile = os.path.join(data_path, "WindField_Point.shp")

# Open file to remove # characters and :Datatype marks
print("Open WindField CSV")
flag = False
with open(wind_input_file, "r") as input_file, open(
    wind_output_file, "w"
) as output_file:
    for index, line in enumerate(input_file):
        if line.startswith("#"):
            continue
        if not flag:
            line = (
                line.replace("snap_geometry", "")
                .replace(":String", "")
                .replace(":Double", "")
                .replace(":Point", "")
            )
            flag = True
        output_file.write(line)

# Open wind CSV file
wind_gdf = gpd.read_file(wind_output_file, ignore_geometry=True)

# Extract wind field points and values
print("Get speed and coordinates values")
geometry = gpd.GeoSeries.from_wkt(wind_gdf["geometry"])
speed_values = wind_gdf["speed"].values.astype(np.float32)
heading_values = wind_gdf["heading"].values.astype(np.float32)

# Prepare OGR for shapefile creation
driver = ogr.GetDriverByName("ESRI Shapefile")
if os.path.exists(output_shapefile):
    driver.DeleteDataSource(output_shapefile)
data_source = driver.CreateDataSource(output_shapefile)

# Create the shapefile layer with WGS84 spatial reference
spatial_ref = osr.SpatialReference()
spatial_ref.ImportFromEPSG(4326)
layer = data_source.CreateLayer("WindField_Point", spatial_ref, ogr.wkbPoint)

# Add fieldds to the shapefile
layer.CreateField(ogr.FieldDefn("speed", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("heading", ogr.OFTReal))

# Extract the features from the vector data node
for geom, speed, heading in zip(geometry, speed_values, heading_values):

    # Create OGR point feature
    ogr_feature = ogr.Feature(layer.GetLayerDefn())
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(geom.x, geom.y)

    ogr_feature.SetGeometry(point)

    # Set attribute values
    ogr_feature.SetField("speed", str(speed))
    ogr_feature.SetField("heading", str(heading))

    layer.CreateFeature(ogr_feature)

# Save and close
data_source = None
print(f"Shapefile successfully saved: {output_shapefile}")
