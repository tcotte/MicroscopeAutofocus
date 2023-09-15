from typing import Any, List

import torch
import torchvision
from torch import nn
from torchvision.models import MobileNetV3
from torchvision.models.mobilenetv3 import InvertedResidualConfig, _mobilenet_v3_conf, MobileNet_V3_Small_Weights

from utils import get_device, get_model_size

"""
Swish explanation: https://medium.com/@neuralnets/swish-activation-function-by-google-53e1ea86f820
"""


class RegressionMobilenet(MobileNetV3):
    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig], last_channel: int, dropout: float = 0.2,
                 **kwargs: Any):
        super().__init__(inverted_residual_setting, last_channel, **kwargs)

        self.dropout = dropout
        self.classifier = nn.Sequential(*self.get_regression_layers())

    def get_regression_layers(self) -> List:
        layers = []
        layers += [nn.Linear(in_features=576, out_features=1024)]
        layers += [nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        layers += [nn.Dropout(p=self.dropout)]
        layers += [nn.Linear(1024, 512, bias=True), nn.Hardswish(inplace=True)]
        layers += [nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        layers += [nn.Dropout(p=self.dropout)]
        layers += [nn.Linear(512, 16, bias=True), nn.Hardswish(inplace=True)]
        layers += [nn.Linear(16, 1)]
        return layers


if __name__ == "__main__":
    device = get_device()
    # inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small")
    # pretrained_weights = ("pretrained", MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    pretrained_weights = None

    model = RegressionMobilenet(*_mobilenet_v3_conf("mobilenet_v3_small"), dropout=0.4, weights=pretrained_weights)
    model.load_state_dict(torch.load("models/Run_complex_model_filtered_ds.pt", map_location=torch.device(device)))
    print("ok")

    print('model size: {:.3f}MB'.format(get_model_size(model)))


    model = torchvision.models.mobilenet_v3_small(weights='DEFAULT', dropout=0.2)


    layers = []
    layers += [nn.Linear(in_features=576, out_features=1024)]
    layers += [nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
    layers += [nn.Dropout(p=0.2)]
    layers += [nn.Linear(1024, 512, bias=True), nn.ReLU(inplace=True)]
    layers += [nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
    layers += [nn.Dropout(p=0.2)]
    layers += [nn.Linear(512, 16, bias=True), nn.ReLU(inplace=True)]
    layers += [nn.Linear(16, 1)]
    model.classifier = nn.Sequential(*layers)

    print('model size: {:.3f}MB'.format(get_model_size(model)))