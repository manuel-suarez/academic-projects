import os
import subprocess

base_path = os.path.expanduser("~")
data_path = os.path.join(base_path, "data", "cimat", "dataset-sentinel")

# Use SLURM array environment variables to determine training and cross validation set number
# If there is a command line argument we are using instead the environment variable (it takes precedence)
slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
slurm_node_list = os.getenv("SLURM_JOB_NODELIST")
print(f"SLURM_ARRAY_JOB_ID: {slurm_array_job_id}")
print(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")
print(f"SLURM_JOB_NODELIST: {slurm_node_list}")

# Get file name and path and execute bash script to process file
fnames = os.listdir(os.path.join(data_path, "GRD"))
fname = fnames[int(slurm_array_task_id) - 1]
subprocess.run(
    [
        "bash",
        "make_texture_matrices.sh",
        data_path,
        "GRD",
        "GLCM",
        "temp2",
        fname,
    ]
)
