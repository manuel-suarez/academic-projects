import os
import torch
import unittest
import itertools

from models import build_model, model_names, model_class
from torchview import draw_graph


class TestBase(unittest.TestCase):
    pass


def create_test_shape_for_encoder(model_name, encoder_name):
    def test_shape(self):
        print(f"Testing shape {model_name}-{encoder_name}")
        input_tensor = torch.randn(8, 3, 256, 256)
        model = build_model(
            model_name=model_name,
            encoder_name=encoder_name,
            in_channels=3,
            out_channels=1,
        )
        output = model(input_tensor)
        self.assertEqual((8, 1, 256, 256), output.shape)

    return test_shape


def create_test_for_encoder(model_name, encoder_name):
    def test_encoder(self):
        print(f"Testing graph {model_name}-{encoder_name}")
        try:
            model = build_model(
                model_name=model_name,
                encoder_name=encoder_name,
                in_channels=1,
                out_channels=1,
            )
            draw_graph(
                model,
                input_size=(1, 1, 256, 256),
                depth=5,
                show_shapes=True,
                expand_nested=True,
                save_graph=True,
                filename=f"encoder",
                directory=os.path.join("figures", model_name, encoder_name),
            )
            return
        except Exception as e:
            self.fail(
                f"No se pudo crear el modelo: {model_name}, encoder: {encoder_name}, excepción: ({e})"
            )

    return test_encoder


from test_encoders import (
    encoders_names,
    encoders_mr_blocks,
    encoders_dal_layers,
)

encoders_layers = [34]  # , 34, 50, 101, 152]
encoders = [
    f"{encoder_name}{encoder_layers}"
    + (f"_MR{encoder_mr_block_version}" if encoder_mr_block_version != "" else "")
    + (f"_MD{encoder_dal_layers_version}" if encoder_dal_layers_version != "" else "")
    for encoder_name, encoder_layers, encoder_mr_block_version, encoder_dal_layers_version in itertools.product(
        encoders_names, encoders_layers, encoders_mr_blocks, encoders_dal_layers
    )
    if not (encoder_mr_block_version == "" and encoder_dal_layers_version != "")
]

for model_name in model_names():
    tests = {
        f"test_encoder_{encoder}": create_test_for_encoder(
            model_name, encoder.split("_")[0]
        )
        for encoder in encoders
    }
    test_shapes = {
        f"test_shape_{encoder}": create_test_shape_for_encoder(
            model_name, encoder.split("_")[0]
        )
        for encoder in encoders
    }
    tests.update(test_shapes)
    model_class_name = model_class(model_name).__name__
    classname = f"Test{model_class_name}"
    globals()[classname] = type(classname, (TestBase,), tests)


if __name__ == "__main__":
    print(encoders)
