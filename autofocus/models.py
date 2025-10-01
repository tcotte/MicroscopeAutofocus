import os

import torch
import torch.nn as nn
import sys

sys.path.insert(0,'autofocus/mobile-vit-pytorch/mobile_vit')
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


if __name__ == '__main__':
    model = ModifiedMobileViT(image_size=(224, 224), num_classes=576, mode='xx_small', drop_out=0.1)
    x = torch.randn(2, 3, 224, 224)
    print(model(x))
