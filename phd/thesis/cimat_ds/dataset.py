import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader, random_split

channels = {
    "o": "origin",
    "ta": "texture/asm",
    "tc": "texture/contrast",
    "td": "texture/dissimilarity",
    "tn": "texture/energy",
    "te": "texture/entropy",
    "tr": "texture/glcmcorrelation",
    "tm": "texture/glcmmean",
    "tv": "texture/glcmvariance",
    "th": "texture/homogeneity",
    "tx": "texture/max",
    # "v": "var",
    # "w": "wind",
}

full_channels = {"o": "ORIGIN", "v": "VAR", "w": "WIND"}


class CimatDataset(Dataset):
    def __init__(
        self,
        base_dir,
        # Now we need to separate features channels with a symbol like -
        features_channels,
        features_separator="-",
        # Parameters for full dataset (~60000 image patches with data augmentation)
        oil_spill_num=17,  # Folder num
        dataset_num=1,  # Crossvalidation CSV file
        dataset_type="train",  # Dataset
        return_names=False,
        max_images=None,
    ):
        super().__init__()
        # Check that features_channels correspond to valid channels
        sep = features_separator
        if not all([x in full_channels.keys() for x in features_channels.split(sep)]):
            elem = [
                x for x in features_channels.split(sep) if x not in full_channels.keys()
            ]
            if len(elem) == 1:
                raise Exception(
                    f"Feature channel {','.join(elem)} is not a valid channel indentification"
                )
            else:
                raise Exception(
                    f"Features channels: {','.join(elem)} are not valid channel indentifications"
                )
        # Initialization
        self.data_dir = os.path.join(
            base_dir,
            "data",
            "projects",
            "consorcio-ia",
            "data",
            f"oil_spills_{oil_spill_num:02d}",
            "augmented_dataset",
        )
        if dataset_type == "train":
            dataset_dir = "trainingFiles"
        elif dataset_type == "test":
            dataset_dir = "testingFiles"
        else:
            dataset_dir = "crossFiles"
        self.datafiles_path = os.path.join(
            self.data_dir,
            "learningCSV",
            dataset_dir,
            f"{dataset_type}{dataset_num:02d}.csv",
        )
        if not os.path.exists(self.datafiles_path):
            raise Exception("Dataset {self.datafiles_path} directory doesn't exists")
        self.return_names = return_names
        self.features_dir = os.path.join(self.data_dir, "features")
        self.labels_dir = os.path.join(self.data_dir, "labels")
        self.features_channels = [
            full_channels[feat] for feat in features_channels.split(sep)
        ]
        # Open CSV with crossvalidation specification file
        self.datafiles_df = pd.read_csv(self.datafiles_path)
        self.filenames = self.datafiles_df["key"].tolist()
        if max_images is not None:
            self.filenames = self.filenames[:max_images]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        # Load label
        label = torch.from_numpy(
            np.expand_dims(
                imread(
                    os.path.join(self.labels_dir, filename + ".pgm"),
                    as_gray=True,
                ).astype(np.float32),
                0,
            )
        )
        # Label is in 0-255 range so we need to divide between 255
        label /= 255

        # Load features
        features = torch.stack(
            [
                torch.from_numpy(
                    imread(
                        os.path.join(
                            self.features_dir,
                            feature,
                            filename + ".tiff",
                        ),
                        as_gray=True,
                    )
                )
                for feature in self.features_channels
            ]
        )
        if self.return_names:
            return features, label, filename
        return features, label


def prepare_dataloaders(
    base_dir,
    feat_channels,
    feat_separator="-",
    oil_spill_num=17,
    dataset_num=1,
    train_batch_size=8,
    valid_batch_size=4,
    test_batch_size=4,
    return_names=False,
    max_images=None,
):
    train_dataset = CimatDataset(
        base_dir=base_dir,
        features_channels=feat_channels,
        features_separator=feat_separator,
        oil_spill_num=oil_spill_num,
        dataset_num=dataset_num,
        dataset_type="train",
        return_names=return_names,
        max_images=max_images,
    )
    valid_dataset = CimatDataset(
        base_dir=base_dir,
        features_channels=feat_channels,
        features_separator=feat_separator,
        oil_spill_num=oil_spill_num,
        dataset_num=dataset_num,
        dataset_type="cross",
        return_names=return_names,
        max_images=max_images,
    )
    test_dataset = CimatDataset(
        base_dir=base_dir,
        features_channels=feat_channels,
        features_separator=feat_separator,
        oil_spill_num=oil_spill_num,
        dataset_num=dataset_num,
        dataset_type="test",
        return_names=return_names,
        max_images=max_images,
    )
    print("Train dataset: ", len(train_dataset))
    print("Valid dataset: ", len(valid_dataset))
    print("Test dataset: ", len(test_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=8,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
    )
    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F

    home_dir = os.path.expanduser("~")
    for feat_channels in channels.keys():
        print(f"Feature channels: {feat_channels}: {channels[feat_channels]}")
        train_dataset = CimatDataset(
            base_dir=home_dir,
            features_channels=feat_channels,
        )
        features, label = train_dataset[0]
        print(f"\tTensor image shape: {features.shape}")
        print(f"\tTensor label shape: {label.shape}")
        np_features = features.numpy()
        np_label = label.numpy()
        print(f"\tImage: max: {np.max(np_features)}, min: {np.min(np_features)}")
        print(f"\tLabel: max: {np.max(np_label)}, min: {np.min(np_label)}")
