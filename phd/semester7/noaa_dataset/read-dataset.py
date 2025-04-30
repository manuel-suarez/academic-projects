import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon


base_path = os.path.expanduser("~")
data_path = os.path.join(base_path, "data", "cimat", "dataset-noaa")

# Open CSV file, lat and lon are the coordinates
incidents_df = pd.read_csv(os.path.join(data_path, "noaa_sentinel1_incidents.csv"))
# Convert the datetime
incidents_df["open_date"] = pd.to_datetime(incidents_df["open_date"], format="%Y-%m-%d")

# Create a GeoDataFrame from DataFrame
geometry = [Point(xy) for xy in zip(incidents_df["lon"], incidents_df["lat"])]
incidents_noaa_gdf = gpd.GeoDataFrame(incidents_df, geometry=geometry)

# Filter rows within bounding box
# Now we are searching in all NOAA dataset
# incidents_gm_gdf = incidents_gdf[incidents_gdf.within(bbox)]

# Define Sentinel-1, Sentinel-2 operating dates to filter incidents by date range
incidents_noaa_sen1_gdf = incidents_noaa_gdf.loc[
    (incidents_noaa_gdf["open_date"] > "2014-04-03")
]
incidents_noaa_sen2_gdf = incidents_noaa_gdf.loc[
    (incidents_noaa_gdf["open_date"] > "2015-06-23")
]

# Save filtered data
print(incidents_noaa_gdf)
print(incidents_noaa_sen1_gdf)
print(incidents_noaa_sen2_gdf)


incidents_noaa_gdf.to_csv(
    os.path.join(data_path, "noa_sentinel_incidents.csv"), index=False
)
incidents_noaa_sen1_gdf.to_csv(
    os.path.join(data_path, "noaa_sentinel1_incidents.csv"), index=False
)
incidents_noaa_sen2_gdf.to_csv(
    os.path.join(data_path, "noaa_sentinel2_incidents.csv"), index=False
)
