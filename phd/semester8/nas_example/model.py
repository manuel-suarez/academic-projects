import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = 3

        for layer_cfg in config:
            if layer_cfg["type"] == "conv":
                self.layers.append(
                    nn.Conv2d(
                        in_channels,
                        layer_cfg["out_channels"],
                        kernel_size=layer_cfg["kernel_size"],
                        stride=1,
                        padding=1,
                    )
                )
                in_channels = layer_cfg["out_channels"]
            elif layer_cfg["type"] == "relu":
                self.layers.append(nn.ReLU())
            elif layer_cfg["type"] == "pool":
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.softmax(x, dim=1)
