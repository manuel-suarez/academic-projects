import os

from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

Image.MAX_IMAGE_PIXELS = None


home_path = os.path.expanduser("~")
data_path = os.path.join(home_path, "data", "cimat", "dataset-tarso")
results_path = os.path.join(data_path, "results")

products = ["_".join(dname.split("_")[:-2]) for dname in os.listdir(results_path)]
print(products)

os.makedirs("figures", exist_ok=True)
for pname in tqdm(products):
    # Open result image
    image = Image.open(
        os.path.join(results_path, pname + "_not_wind", pname + "_result.png")
    )
    width, height = image.size
    # Reduce 10%
    width = width * 10 // 100
    height = height * 10 // 100
    # Resize
    newsize = (width, height)
    image = image.resize(newsize)
    # Save
    image.save(os.path.join("figures", pname + "_result.png"))
