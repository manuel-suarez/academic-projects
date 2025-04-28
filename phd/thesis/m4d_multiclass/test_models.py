import torch
import unittest
from torchview import draw_graph

from models.unet import UNet
from models.unet2p import UNet2P
from models.unet3p import UNet3P
from models.linknet import LinkNet
from models.pspnet import PSPNet
from models.fpnet import FPNet
from models.deeplabv3p import DeepLabV3P
from models.manet import MANet
from models.vit import ViT
from models.swin import SwinTransformer

models = {
    "unet": UNet,
    "unet2p": UNet2P,
    "unet3p": UNet3P,
    "linknet": LinkNet,
    "pspnet": PSPNet,
    "fpn": FPNet,
    "deeplabv3p": DeepLabV3P,
    "manet": MANet,
    "vit": ViT,
    "swin": SwinTransformer,
}


class TestBase(unittest.TestCase):
    pass


def create_test_shape_for_encoder(model_name, encoder_name):
    def test_shape(self):
        print(f"Testing shape {model_name}-{encoder_name}")
        input_tensor = torch.randn(8, 3, 256, 256)
        model = models[model_name](in_channels=3, out_channels=1)
        output = model(input_tensor)
        self.assertEqual((8, 1, 256, 256), output.shape)

    return test_shape


def create_test_multiclass_for_encoder(model_name, encoder_name):
    def test_shape(self):
        print(f"Testing multiclass {model_name}-{encoder_name}")
        input_tensor = torch.randn(8, 3, 256, 256)
        model = models[model_name](in_channels=3, out_channels=5)
        output = model(input_tensor)
        self.assertEqual((8, 5, 256, 256), output.shape)

    return test_shape


def create_test_for_encoder(model_name, encoder_name):
    def test_encoder(self):
        print(f"Testing graph {model_name}-{encoder_name}")
        try:
            model = models[model_name](in_channels=1, out_channels=1)
            draw_graph(
                model,
                input_size=(1, 1, 256, 256),
                depth=5,
                show_shapes=True,
                expand_nested=True,
                save_graph=True,
                filename=f"{model_name}-{encoder_name}_encoder",
                directory="figures",
            )
            return
        except Exception as e:
            self.fail(
                f"No se pudo crear el modelo: {model_name}, encoder: {encoder_name}, excepción: ({e})"
            )

    return test_encoder


encoders = ["basic"]

for model_name in models:
    tests = {
        f"test_{encoder}_encoder": create_test_for_encoder(model_name, encoder)
        for encoder in encoders
    }
    test_shapes = {
        f"test_{encoder}_shape": create_test_shape_for_encoder(model_name, encoder)
        for encoder in encoders
    }
    test_multiclass = {
        f"test_{encoder}_multiclass": create_test_multiclass_for_encoder(
            model_name, encoder
        )
        for encoder in encoders
    }
    tests.update(test_shapes)
    tests.update(test_multiclass)
    model_class = models[model_name].__name__
    classname = f"Test{model_class}"
    globals()[classname] = type(classname, (TestBase,), tests)

if __name__ == "__main__":
    unittest.main()
