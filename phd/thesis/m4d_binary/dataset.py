import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


class KrestenitisDataset(Dataset):
    def __init__(self, base_dir, return_names=False, max_images=None) -> None:
        super().__init__()
        self.images_dir = os.path.join(base_dir, "images")
        self.labels_dir = os.path.join(base_dir, "labels_1D")

        self.return_names = return_names
        self.ids = os.listdir(self.images_dir)
        if max_images != None:
            self.ids = self.ids[:max_images]
        self.images_fps = [
            os.path.join(self.images_dir, image_id.split(".")[0] + ".jpg")
            for image_id in self.ids
        ]
        self.labels_fps = [
            os.path.join(self.labels_dir, image_id.split(".")[0] + ".png")
            for image_id in self.ids
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Read Image
        image = torch.from_numpy(
            np.expand_dims(imread(self.images_fps[idx], as_gray=True), 0)
        ).type(torch.float)

        label = torch.from_numpy(
            np.expand_dims(imread(self.labels_fps[idx], as_gray=True), 0)
        ).type(torch.float)
        # For binary segmentation we only let the 0 value corresponding to oil spill (all other data will be tagged as zero)
        label[label > 1] = 0

        if self.return_names:
            return image, label, self.ids[idx]
        return image, label


def prepare_dataloaders(
    base_dir,
    version="1",  # 160x160 dataset
    max_train_images=None,
    max_test_images=None,
    split_valid=False,
    split_proportion=[0.9, 0.1],
):
    data_dir = os.path.join(base_dir, "data", f"krestenitis_v{version}")

    train_dir = os.path.join(data_dir, "train")
    train_dataset = KrestenitisDataset(base_dir=train_dir, max_images=max_train_images)
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, pin_memory=True, shuffle=True, num_workers=8
    )

    test_dir = os.path.join(data_dir, "test")
    test_dataset = KrestenitisDataset(base_dir=test_dir, max_images=max_test_images)
    test_dataloader = DataLoader(
        test_dataset, batch_size=8, pin_memory=True, shuffle=False, num_workers=4
    )

    if split_valid:
        # Split train dataset with 10% for validation data
        train_dataset, valid_dataset = random_split(train_dataset, split_proportion)
        print(f"Training dataset length: {len(train_dataset)}")
        print(f"Validation dataset length: {len(valid_dataset)}")
        print(f"Testing dataset length: {len(test_dataset)}")
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=4, pin_memory=True, shuffle=False, num_workers=4
        )
        return train_dataloader, valid_dataloader, test_dataloader

    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Testing dataset length: {len(test_dataset)}")
    return train_dataloader, test_dataloader
