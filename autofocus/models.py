import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0,'autofocus/mobile-vit-pytorch/mobile_vit')
sys.path.insert(0,'../autofocus/mobile-vit-pytorch/mobile_vit')

from mobilevit_v3_v1 import MobileViTv3_v1  #


class ModifiedMobileViT(MobileViTv3_v1):
    def __init__(self, drop_out: float = 0.2, **kwargs):
        super(ModifiedMobileViT, self).__init__(**kwargs)

        # Add your additional layers here
        self.fc1 = nn.Linear(576, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 16, bias=True)
        self.fc4 = nn.Linear(16, 1, bias=True)

        self.bn1 = nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Optional: You can add activation functions or dropout as needed
        self.dropout = nn.Dropout(drop_out)

        self.hardswish = nn.Hardswish(inplace=True)

    def forward(self, x):
        # Use the forward method from the original MobileViTv3_v1
        x = super(ModifiedMobileViT, self).forward(x)

        # Pass through the new layers after the original forward pass
        x = self.bn1(self.fc1(x))  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout (optional)

        x = self.hardswish(self.fc2(x))
        x = self.bn2(x)
        # x = s(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        x = self.dropout(x)

        x = self.hardswish((self.fc3(x)))

        x = self.fc4(x)

        return x


class MobileNetV3_Regressor(nn.Module):
    """
    Read paper Real-Time Facial Affective Computing on Mobile Devices to understand why choose full connected layers
    compared to average pooling.

    """
    def __init__(self, pretrained=True, dropout: float = 0.2):
        super().__init__()
        if pretrained:
            base = torchvision.models.mobilenet_v3_small(weights='DEFAULT', dropout=dropout)
        else:
            base = torchvision.models.mobilenet_v3_small(dropout=dropout)

        self.features = base.features  # everything up to the last feature map
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(576 * 7 * 7, 1)  # single scalar output

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dims except batch
        x = self.fc(x)
        return x


class LightweightNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_b1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_b1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_b2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv_b2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv_b2_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv_b3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv_b3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.conv_b3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv_b4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = F.relu(self.conv_b1_1(x))
        x = self.bn1(x)
        x = F.relu(self.conv_b1_2(x))
        x = self.bn1(x)
        x = self.maxpool(x)

        x = F.relu(self.conv_b2_1(x))
        x = self.bn2(x)
        x = F.relu(self.conv_b2_2(x))
        x = self.bn2(x)
        x = F.relu(self.conv_b2_3(x))
        x = self.bn2(x)
        x = self.maxpool(x)

        x = F.relu(self.conv_b3_1(x))
        x = self.bn3(x)
        x = F.relu(self.conv_b3_2(x))
        x = self.bn3(x)
        x = F.relu(self.conv_b3_3(x))
        x = self.bn3(x)
        x = self.maxpool(x)

        x = F.relu(self.conv_b4(x))

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)



if __name__ == '__main__':
    # model = ModifiedMobileViT(image_size=(224, 224), num_classes=576, mode='xx_small', drop_out=0.1)
    # x = torch.randn(2, 3, 224, 224)
    # print(model(x))

    model = LightweightNetwork()
    x = torch.randn(2, 3, 512, 512)
    print(model(x))
