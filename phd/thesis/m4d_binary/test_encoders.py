import torch
import unittest
import itertools
import argparse

from models.encoders import build_encoder


in_channels = 3
base_channels = 64
base_resolution = 256


def list_of_strings(arg):
    return arg.split(",")


def build_encoder_input(
    encoder_name, in_channels, encoder_layers, encoder_mr_block, encoder_dal_layers
):
    input_tensor = torch.randn(8, in_channels, base_resolution, base_resolution)
    encoder = build_encoder(
        in_channels=in_channels,
        encoder_name=encoder_name,
        encoder_layer=encoder_layers,
        encoder_mr_block=encoder_mr_block,
        encoder_dal_layer=encoder_dal_layers,
        pretrained=True,
    )
    return encoder, input_tensor


class TestBase(unittest.TestCase):
    pass


def create_test_encoder_channels(
    encoder_name,
    encoder_layers,
    encoder_mr_block,
    encoder_dal_layers,
    encoder_channels,
):
    def test_encoder_channels(self):
        encoder, input_tensor = build_encoder_input(
            encoder_name,
            in_channels,
            encoder_layers,
            encoder_mr_block,
            encoder_dal_layers,
        )
        enc1, enc2, enc3, enc4, enc5 = encoder(input_tensor)
        self.assertEqual(encoder_channels[0], enc1.shape[1])
        self.assertEqual(encoder_channels[1], enc2.shape[1])
        self.assertEqual(encoder_channels[2], enc3.shape[1])
        self.assertEqual(encoder_channels[3], enc4.shape[1])
        self.assertEqual(encoder_channels[4], enc5.shape[1])

    return test_encoder_channels


def create_test_encoder_resolution(
    encoder_name,
    encoder_layers,
    encoder_mr_block,
    encoder_dal_layers,
    encoder_resolutions,
):
    def test_encoder_resolution(self):
        encoder, input_tensor = build_encoder_input(
            encoder_name,
            in_channels,
            encoder_layers,
            encoder_mr_block,
            encoder_dal_layers,
        )
        enc1, enc2, enc3, enc4, enc5 = encoder(input_tensor)
        self.assertEqual(
            (encoder_resolutions[0], encoder_resolutions[0]), enc1.shape[2:]
        )
        self.assertEqual(
            (encoder_resolutions[1], encoder_resolutions[1]), enc2.shape[2:]
        )
        self.assertEqual(
            (encoder_resolutions[2], encoder_resolutions[2]), enc3.shape[2:]
        )
        self.assertEqual(
            (encoder_resolutions[3], encoder_resolutions[3]), enc4.shape[2:]
        )
        self.assertEqual(
            (encoder_resolutions[4], encoder_resolutions[4]), enc5.shape[2:]
        )

    return test_encoder_resolution


def create_test_encoder_blocks(
    encoder_name, encoder_layers, encoder_mr_block, encoder_dal_layers, encoder_blocks
):
    def test_encoder_blocks(self):
        encoder, _ = build_encoder_input(
            encoder_name,
            in_channels,
            encoder_layers,
            encoder_mr_block,
            encoder_dal_layers,
        )
        self.assertEqual(encoder_blocks[0], len(encoder.initial))
        self.assertEqual(encoder_blocks[1], len(encoder.layer1))
        self.assertEqual(encoder_blocks[2], len(encoder.layer2))
        self.assertEqual(encoder_blocks[3], len(encoder.layer3))
        self.assertEqual(encoder_blocks[4], len(encoder.layer4))

    return test_encoder_blocks


# Original ResNet architecture uses 1 convolutional block for initial layer, however we are
# using Conv+BN+ReLu so we need to test for 3 blocks instead of 1 (previous test version)
encoders_configurations = {
    "18": {
        "encoder_name": "resnet18",
        "encoder_channels": [64, 64, 128, 256, 512],
        "encoder_resolutions": [128, 64, 32, 16, 8],
        "encoder_blocks": [3, 2, 2, 2, 2],
        "expansion": 1,
    },
    "34": {
        "encoder_name": "resnet34",
        "encoder_channels": [64, 64, 128, 256, 512],
        "encoder_resolutions": [128, 64, 32, 16, 8],
        "encoder_blocks": [3, 3, 4, 6, 3],
        "expansion": 1,
    },
    "50": {
        "encoder_name": "resnet50",
        "encoder_channels": [64, 256, 512, 1024, 2048],
        "encoder_resolutions": [128, 64, 32, 16, 8],
        "encoder_blocks": [3, 3, 4, 6, 3],
        "expansion": 4,
    },
    "101": {
        "encoder_name": "resnet101",
        "encoder_channels": [64, 256, 512, 1024, 2048],
        "encoder_resolutions": [128, 64, 32, 16, 8],
        "encoder_blocks": [3, 3, 4, 23, 3],
        "expansion": 4,
    },
    "152": {
        "encoder_name": "resnet152",
        "encoder_channels": [64, 256, 512, 1024, 2048],
        "encoder_resolutions": [128, 64, 32, 16, 8],
        "encoder_blocks": [3, 3, 8, 36, 3],
        "expansion": 4,
    },
}
encoder_capitalnames = {"resnet": "ResNet", "senet": "SeNet", "cbamnet": "CBAMNet"}
encoders_names = encoder_capitalnames.keys()
encoders_layers = [18, 34, 50, 101, 152]
encoders_mr_blocks = ["", *[f"v{i+1}" for i in range(8)]]
encoders_dal_layers = ["", *[f"v{i+1}" for i in range(4)]]

for (
    encoder_name,
    encoder_layers,
    encoder_mr_block,
    encoder_dal_layers,
) in itertools.product(
    encoders_names,
    encoders_layers,
    encoders_mr_blocks,
    encoders_dal_layers,
):
    # Can't use DAL layers without MR encoder so this combination is not valid
    if encoder_dal_layers != "" and encoder_mr_block == "":
        continue
    tests = {
        "test_channels": create_test_encoder_channels(
            encoder_name,
            encoder_layers,
            encoder_mr_block,
            encoder_dal_layers,
            encoders_configurations[str(encoder_layers)]["encoder_channels"],
        ),
        "test_resolutions": create_test_encoder_resolution(
            encoder_name,
            encoder_layers,
            encoder_mr_block,
            encoder_dal_layers,
            encoders_configurations[str(encoder_layers)]["encoder_resolutions"],
        ),
        "test_blocks": create_test_encoder_blocks(
            encoder_name,
            encoder_layers,
            encoder_mr_block,
            encoder_dal_layers,
            encoders_configurations[str(encoder_layers)]["encoder_blocks"],
        ),
    }
    encoder_class_name = encoder_capitalnames[encoder_name]
    classname = f"Test{encoder_class_name}{encoder_layers}"
    if encoder_mr_block != "":
        classname += f"_MR{encoder_mr_block}"
    if encoder_dal_layers != "":
        classname += f"_MD{encoder_dal_layers}"
    globals()[classname] = type(classname, (TestBase,), tests)
