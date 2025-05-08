import os
import numpy as np

from skimage.io import imread, imsave
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

base_path = os.path.expanduser("~")
data_path = os.path.join(base_path, "data", "cimat", "dataset-cimat")
input_path = os.path.join(data_path, "mask_png")
output_path = os.path.join(data_path, "mask_bin")
os.makedirs(output_path, exist_ok=True)


# Use SLURM array environment variables to determine training and cross validation set number
# If there is a command line argument we are using instead the environment variable (it takes precedence)
slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
slurm_node_list = os.getenv("SLURM_JOB_NODELIST")
print(f"SLURM_ARRAY_JOB_ID: {slurm_array_job_id}")
print(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")
print(f"SLURM_JOB_NODELIST: {slurm_node_list}")

mask_names = os.listdir(input_path)
mask_name = mask_names[int(slurm_array_task_id) - 1]
mask = imread(os.path.join(input_path, mask_name))
mask = mask // 255

imsave(os.path.join(output_path, mask_name), mask.astype(np.int8), check_contrast=False)

print("Done!")
