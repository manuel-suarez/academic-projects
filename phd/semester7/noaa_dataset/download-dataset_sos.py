import os
from datetime import date, timedelta
from tqdm import tqdm
import requests
import pandas as pd
import geopandas as gpd
import subprocess
from dotenv import load_dotenv
from shapely.geometry import shape


load_dotenv()
# Set copernicus access variable
copernicus_user = os.getenv("COPERNICUS_USER")
copernicus_password = os.getenv("COPERNICUS_PASSWORD")
# Get indexes to analyze from CSV files
index_start = int(os.getenv("INDEX_START"))
index_end = int(os.getenv("INDEX_END"))

# WKT representation of BBOX of AOI
ft = "POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))"
# Sentinel satellite that you are interested in
data_collection = "SENTINEL-2"

# Dates of search
today = date.today()
today_string = today.strftime("%Y-%m-%d")
yesterday = today - timedelta(days=1)
yesterday_string = yesterday.strftime("%Y-%m%d")


def get_keycloak(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Keycloak token creation failed. Response from the server was: {r.json()}"
        )
    return r.json()["access_token"]


def get_products(
    data_collection, geometry, date_string, outputpath, products_filter=None
):
    json_ = requests.get(
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{geometry}') and ContentDate/Start gt {date_string}T00:00:00.000Z and ContentDate/Start lt {date_string}T23:59:59.000Z&$count=True&$top=1000"
    ).json()
    # print(json_)
    p = pd.DataFrame.from_dict(json_["value"])
    if p.shape[0] > 0:
        p["geometry"] = p["GeoFootprint"].apply(shape)
        # Convert pandas dataframe to Geopandas dataframe by setting up geometry
        productDF = gpd.GeoDataFrame(p).set_geometry("geometry")
        # Remove L1C dataset if not needed
        productDF = productDF[~productDF["Name"].str.contains("L1C")]
        print(f" total L2A tiles found {len(productDF)}")
        productDF["identifier"] = productDF["Name"].str.split(".").str[0]
        allfeat = len(productDF)

        if allfeat == 0:
            print("No tiles found for {date_string}")
        else:
            for index, feat in enumerate(productDF.iterfeatures()):
                # if name is according to products filter then download
                identifier = feat["properties"]["identifier"]
                print(identifier)
                # We are omitting the COG products
                # if "COG" in identifier:
                #    print("It's a COG product, continue next")
                #    continue
                if products_filter is not None and any(
                    map(lambda x: x in identifier, products_filter)
                ):
                    print(
                        f"Identifier {identifier} in list {products_filter}, checking if exists"
                    )
                    # If file exists resume next (we need to search in each of the subdirectories)
                    if any(
                        map(
                            # Need to check on destination path or downloaded path
                            lambda subdir: os.path.exists(
                                os.path.join(
                                    outputpath,
                                    subdir,
                                    f"{feat['properties']['identifier']}.zip",
                                )
                            ),
                            products_filter,
                        )
                    ) or os.path.exists(
                        os.path.join(
                            outputpath,
                            "downloaded",
                            f"{feat['properties']['identifier']}.txt",
                        )
                    ):
                        print("Exists, continue next")
                        continue
                    try:
                        print(
                            f"Identifier {identifier} doesn't exists, proceeding to download"
                        )
                        # Create requests session
                        session = requests.Session()
                        # Get access token based on username and password
                        keycloak_token = get_keycloak(
                            copernicus_user, copernicus_password
                        )
                        session.headers.update(
                            {"Authorization": f"Bearer {keycloak_token}"}
                        )
                        url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({feat['properties']['Id']})/$value"
                        print(identifier, url)
                        response = session.get(url, allow_redirects=False)
                        while response.status_code in (301, 302, 303, 307):
                            url = response.headers["Location"]
                            response = session.get(url, allow_redirects=False)
                        print(feat["properties"]["Id"])
                        # Refresh token
                        keycloak_token = get_keycloak(
                            copernicus_user, copernicus_password
                        )
                        session.headers.update(
                            {"Authorization": f"Bearer {keycloak_token}"}
                        )
                        file = session.get(url, verify=False, allow_redirects=True)

                        # Subdirectory of destination is according to name
                        subdir = ""
                        for name in products_filter:
                            if name in identifier:
                                subdir = name
                                break
                        # Write destination file
                        with open(
                            os.path.join(
                                outputpath,
                                subdir,
                                f"{feat['properties']['identifier']}.zip",
                            ),  # location to save zip from copernicus
                            "wb",
                        ) as p:
                            print(feat["properties"]["Name"])
                            p.write(file.content)
                        # Write indicator file
                        with open(
                            os.path.join(
                                outputpath,
                                "downloaded",
                                f"{feat['properties']['identifier']}.txt",
                            ),
                            "w",
                        ) as p:
                            p.write("done!")
                        # Send file to siimon5 over ssh
                        subprocess.run(
                            [
                                "scp",
                                "-P 2235",
                                os.path.join(
                                    outputpath,
                                    subdir,
                                    f"{feat['properties']['identifier']}.zip",
                                ),
                                f"manuelsuarez@siimon5.cimat.mx:/home/mariocanul/image_storage/dataset-noaa/sentinel2/{subdir}",
                            ]
                        )
                        # Delete file from el-insurgente
                        os.remove(
                            os.path.join(
                                outputpath,
                                subdir,
                                f"{feat['properties']['identifier']}.zip",
                            )
                        )
                    except:
                        print("problem with server")
    else:
        print("No products")


def open_and_search_products(inputfile, outputpath, satellite, products_filter):
    incidents_ds = gpd.read_file(inputfile)
    total_rows = len(incidents_ds)
    print(incidents_ds, total_rows)
    print(type(incidents_ds["geometry"]))

    for index, row in tqdm(incidents_ds.iterrows()):
        if not (index >= index_start and index < index_end):
            if index == index_end:
                print(f"Index {index} end {index_end}, terminate")
                break
            print(f"Index {index} not in range ({index_start}-{index_end}), continue")
            continue
        print(f"Row {index} of {total_rows}, start {index_start} to end {index_end}")
        coordinates = row.geometry
        date = row.open_date
        prods = get_products(satellite, coordinates, date, outputpath, products_filter)
        # print(index, date, coordinates, prods)
    # save result


# We need to open CSV dataset and get date and coordinates to search a Sentinel 1/2 product
if __name__ == "__main__":
    # Define paths

    base_path = os.path.expanduser("~")
    data_path = os.path.join(base_path, "data", "cimat", "dataset-noaa")
    input_file = os.path.join(data_path, "noaa_sentinel2_incidents.csv")
    output_path = os.path.join(data_path, "sentinel2_products")

    # Open CSV file with geopandas to get access to dates and coordinates

    open_and_search_products(input_file, output_path, "SENTINEL-2", ["MSI"])
