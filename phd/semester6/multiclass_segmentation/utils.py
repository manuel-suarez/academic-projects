from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import CityscapesDataset


def get_cityscapes_data(
    mode,
    split,
    root_dir="datasets/cityscapes",
    target_type="semantic",
    transforms=None,
    batch_size=1,
    eval=False,
    shuffle=True,
    pin_memory=True,
):
    data = CityscapesDataset(
        mode=mode,
        split=split,
        target_type=target_type,
        transform=transforms,
        root_dir=root_dir,
        eval=eval,
    )

    data_loaded = DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
    )

    return data_loaded


# Functions to save predictions as images
def save_as_images(tensor_pred, folder, image_name):
    tensor_pred = transforms.ToPILImage()(tensor_pred.byte())
    filename = f"{folder}/{image_name}.png"
    tensor_pred.save(filename)
