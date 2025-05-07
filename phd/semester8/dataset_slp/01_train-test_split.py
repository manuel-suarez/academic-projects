import os
import glob
import numpy as np
import rasterio
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# CONFIGURATION
LABELS_DIR = "path_to_label_tiff_files"
SAR_DIR = "path_to_sar_images"
OUTPUT_DIR = "path_to_dataset_root"  # Will create train/test structure here
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Step 1: Get label files
label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.tif")))


# Step 2: Oil spill % calculator
def calculate_percentage(label_path):
    with rasterio.open(label_path) as src:
        mask = src.read(1)
    oil_pixels = np.sum(mask == 1)
    total_pixels = mask.size
    return oil_pixels / total_pixels if total_pixels > 0 else 0


print("Calculating oil spill percentages...")
percentages = [calculate_percentage(fp) for fp in label_files]

# Step 3: Build dataframe
df = pd.DataFrame({"label_path": label_files, "oil_spill_percentage": percentages})
df["image_path"] = df["label_path"].apply(
    lambda x: os.path.join(SAR_DIR, os.path.basename(x))
)

# Step 4: Binning for stratification
df["bin"] = pd.qcut(df["oil_spill_percentage"], q=10, duplicates="drop")

# Step 5: Stratified split
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["bin"]
)


# Step 6: Report stats
def report_stats(name, subset):
    with rasterio.open(subset.iloc[0]["label_path"]) as src:
        total_px = src.read(1).size
    total_spill = (subset["oil_spill_percentage"] * total_px).sum()
    print(f"{name} oil spill: {100 * total_spill / (len(subset)*total_px):.2f}%")


print("\n📊 Dataset Spill Stats:")
report_stats("Original", df)
report_stats("Train", train_df)
report_stats("Test", test_df)

# Step 7: Save CSVs
train_df.drop(columns="bin").to_csv("train_split.csv", index=False)
test_df.drop(columns="bin").to_csv("test_split.csv", index=False)
print("\n✅ CSVs saved: 'train_split.csv', 'test_split.csv'")

# Step 8: Create folder structure
for split in ["train", "test"]:
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, sub), exist_ok=True)


# Step 9: Copy files
def copy_files(split_df, split_name):
    for _, row in split_df.iterrows():
        image_name = os.path.basename(row["image_path"])
        label_name = os.path.basename(row["label_path"])
        shutil.copy(
            row["image_path"],
            os.path.join(OUTPUT_DIR, split_name, "images", image_name),
        )
        shutil.copy(
            row["label_path"],
            os.path.join(OUTPUT_DIR, split_name, "labels", label_name),
        )


print("Copying files...")
copy_files(train_df, "train")
copy_files(test_df, "test")
print("✅ All files copied to split directories.")

# Step 10: Plot distribution
plt.figure(figsize=(10, 6))
sns.kdeplot(df["oil_spill_percentage"], label="Full", fill=True)
sns.kdeplot(train_df["oil_spill_percentage"], label="Train", fill=True)
sns.kdeplot(test_df["oil_spill_percentage"], label="Test", fill=True)
plt.xlabel("Oil Spill Percentage")
plt.ylabel("Density")
plt.title("Oil Spill Percentage Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("oil_spill_distribution.png")
plt.show()
