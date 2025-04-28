import torch
import unittest
import itertools

from models.encoders import build_encoder


in_channels = 3
base_channels = 64
base_resolution = 256


def build_encoder_input(encoder_name):
    input_tensor = torch.randn(8, in_channels, base_resolution, base_resolution)
    encoder = build_encoder(
        encoder_name=encoder_name,
        in_channels=in_channels,
        last_block=True,
    )
    return encoder, input_tensor


class TestBase(unittest.TestCase):
    pass


def create_test_encoder_channels(encoder_name, encoder_channels):
    def test_encoder_channels(self):
        encoder, input_tensor = build_encoder_input(encoder_name)
        enc1, enc2, enc3, enc4, enc5 = encoder(input_tensor)
        self.assertEqual(encoder_channels[0], enc1.shape[1])
        self.assertEqual(encoder_channels[1], enc2.shape[1])
        self.assertEqual(encoder_channels[2], enc3.shape[1])
        self.assertEqual(encoder_channels[3], enc4.shape[1])
        self.assertEqual(encoder_channels[4], enc5.shape[1])

    return test_encoder_channels


def create_test_encoder_resolution(encoder_name, encoder_resolutions):
    def test_encoder_resolution(self):
        encoder, input_tensor = build_encoder_input(encoder_name)
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


def create_test_encoder_blocks(encoder_name, encoder_blocks):
    def test_encoder_blocks(self):
        encoder, _ = build_encoder_input(encoder_name)
        self.assertEqual(encoder_blocks[0], len(encoder.block1))
        self.assertEqual(encoder_blocks[1], len(encoder.block2))
        self.assertEqual(encoder_blocks[2], len(encoder.block3))
        self.assertEqual(encoder_blocks[3], len(encoder.block4))
        self.assertEqual(encoder_blocks[4], len(encoder.block5))

    return test_encoder_blocks


encoders_configurations = {
    "18": {
        "encoder_name": "resnet18",
        "encoder_channels": [64, 64, 128, 256, 512],
        "encoder_resolutions": [128, 64, 32, 16, 8],
        "encoder_blocks": [1, 2, 2, 2, 2],
        "expansion": 1,
    },
    "34": {
        "encoder_name": "resnet34",
        "encoder_channels": [64, 64, 128, 256, 512],
        "encoder_resolutions": [128, 64, 32, 16, 8],
        "encoder_blocks": [1, 3, 4, 6, 3],
        "expansion": 1,
    },
    "50": {
        "encoder_name": "resnet50",
        "encoder_channels": [64, 256, 512, 1024, 2048],
        "encoder_resolutions": [128, 64, 32, 16, 8],
        "encoder_blocks": [1, 3, 4, 6, 3],
        "expansion": 4,
    },
    "101": {
        "encoder_name": "resnet101",
        "encoder_channels": [64, 256, 512, 1024, 2048],
        "encoder_resolutions": [128, 64, 32, 16, 8],
        "encoder_blocks": [1, 3, 4, 23, 3],
        "expansion": 4,
    },
    "152": {
        "encoder_name": "resnet152",
        "encoder_channels": [64, 256, 512, 1024, 2048],
        "encoder_resolutions": [128, 64, 32, 16, 8],
        "encoder_blocks": [1, 3, 8, 36, 3],
        "expansion": 4,
    },
}
encoder_names = [
    "resnet",
    "senet",
    "cbamnet",
    "mrnet",
    "mrnetv2_",
    "mrnetv3_",
    "resnetmr",
    "senetmr",
    "cbamnetmr",
    "resnetmrv2_",
    "resnetmrv3_",
    "senetmrv2_",
    "cbamnetmrv2_",
]
encoder_capitalnames = {
    "resnet": "ResNet",
    "senet": "SeNet",
    "cbamnet": "CBAMNet",
    "mrnet": "MRNet",
    "mrnetv2_": "MRNetv2",
    "mrnetv3_": "MRNetv3",
    "resnetmr": "ResNetMR",
    "senetmr": "SeNetMR",
    "cbamnetmr": "CBAMNetMR",
    "resnetmrv2_": "ResNetMRv2",
    "resnetmrv3_",: "ResNetMRv3",
    "senetmrv2_": "SeNetMRv2",
    "cbamnetmrv2_": "CBAMNetMRv2",
}
encoder_sizes = [18, 34, 50, 101, 152]

for encoder, encoder_size in itertools.product(encoder_names, encoder_sizes):
    encoder_name = f"{encoder}{encoder_size}"
    tests = {
        "test_channels": create_test_encoder_channels(
            encoder_name, encoders_configurations[str(encoder_size)]["encoder_channels"]
        ),
        "test_resolutions": create_test_encoder_resolution(
            encoder_name,
            encoders_configurations[str(encoder_size)]["encoder_resolutions"],
        ),
        "test_blocks": create_test_encoder_blocks(
            encoder_name, encoders_configurations[str(encoder_size)]["encoder_blocks"]
        ),
    }
    encoder_class_name = encoder_capitalnames[encoder]
    classname = f"Test{encoder_class_name}{encoder_size}"
    globals()[classname] = type(classname, (TestBase,), tests)
