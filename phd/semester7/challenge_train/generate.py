import itertools
import os
import numpy as np
from dataset import features_dirs

# Configure paths
channels_path = "channels"
evaluate_path = os.path.join(channels_path, "evaluate")
trained_path = os.path.join(channels_path, "trained")
training_path = os.path.join(channels_path, "training")
ranked_path = os.path.join(channels_path, "ranked")
# Generate directories
os.makedirs("channels", exist_ok=True)
os.makedirs(evaluate_path, exist_ok=True)
os.makedirs(trained_path, exist_ok=True)
os.makedirs(training_path, exist_ok=True)
os.makedirs(ranked_path, exist_ok=True)

# Generate channels combinations
for channels in [
    f for n in range(3) for f in itertools.combinations(features_dirs, n + 1)
]:
    indexes = [str(features_dirs.index(channel)) for channel in channels]
    print(channels, indexes)
    fname = ".".join(indexes)
    print(fname)
    open(os.path.join(evaluate_path, fname), "w")
