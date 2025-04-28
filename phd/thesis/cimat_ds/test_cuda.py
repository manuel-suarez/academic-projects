import torch

from slurm import slurm_vars

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device found: {device}")
print("SLURM VARS")
for var in slurm_vars:
    print(f"Var ({var}): {slurm_vars[var]}")
