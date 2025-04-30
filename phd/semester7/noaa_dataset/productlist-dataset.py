import os
from datetime import date, timedelta
from tqdm import tqdm
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape


# Set copernicus access variable
copernicus_user = os.getenv("copernicus_user")
copernicus_password = os.getenv("copernicus_password")
# WKT representation of BBOX of AOI
ft = "POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))"
# Sentinel satellite that you are interested in
data_collection = "SENTINEL-1"

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


def get_products(data_collection, geometry, date_string):
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
            return "No tiles found for {date_string}"
        else:
            response = [feat["properties"]["Name"] for feat in productDF.iterfeatures()]
            return response
    else:
        return ""
        # else:
        #    for index, feat in enumerate(productDF.iterfeatures()):
        #        try:
        #            # Create requests session
        #            session = requests.Session()
        #            # Get access token based on username and password
        #            keycloak_token = get_keycloak(copernicus_user, copernicus_password)

        #            session.headers.update(
        #                {"Authorization": f"Bearer {keycloak_token}"}
        #            )
        #            url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({feat['properties']['Id']})/$value"
        #            response = session.get(url, allow_redirects=False)
        #            while response.status_code in (301, 302, 303, 307):
        #                url = response.headers["Location"]
        #                response = session.get(url, allow_redirects=False)
        #            print(feat["properties"]["Id"])
        #            file = session.get(url, verify=False, allow_redirects=True)

        #            with open(
        #                f"location/to/save/{feat['properties']['identifier']}.zip",  # location to save zip from copernicus
        #                "wb",
        #            ) as p:
        #                print(feat["properties"]["Name"])
        #                p.write(file.content)
        #        except:
        #            print("problem with server")
        # else:
        #    print("no data found")


def open_and_search_products(inputfile, outputfile, satellite):
    incidents_ds = gpd.read_file(inputfile)
    print(incidents_ds)
    print(type(incidents_ds["geometry"]))

    data = {
        "date": [],
        "threat": [],
        "products": [],
        "geometry": [],
    }

    for index, row in tqdm(incidents_ds.iterrows()):
        coordinates = row.geometry
        threat = row.threat
        date = row.open_date
        prods = get_products(satellite, coordinates, date)
        # print(index, date, coordinates, prods)
        data["date"].append(date)
        data["threat"].append(threat)
        data["products"].append(prods)
        data["geometry"].append(coordinates)
    # save result
    df = pd.DataFrame(data)
    df.to_csv(outputfile, index=False)


# We need to open CSV dataset and get date and coordinates to search a Sentinel 1/2 product
if __name__ == "__main__":
    # Define paths

    base_path = os.path.expanduser("~")
    data_path = os.path.join(base_path, "data", "cimat", "dataset-noaa")
    input_file = os.path.join(data_path, "noaa_sentinel2_incidents.csv")
    output_file = os.path.join(data_path, "noaa_sentinel2_products.csv")

    # Open CSV file with geopandas to get access to dates and coordinates

    open_and_search_products(input_file, output_file, "SENTINEL-2")
