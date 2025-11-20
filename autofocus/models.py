import math
import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import sys

from torch._C._te import Tensor


# sys.path.insert(0, 'autofocus/mobile-vit-pytorch/mobile_vit')
# sys.path.insert(0, '../autofocus/mobile-vit-pytorch/mobile_vit')
#
# from mobilevit_v3_v1 import MobileViTv3_v1  #


# class ModifiedMobileViT(MobileViTv3_v1):
#     def __init__(self, drop_out: float = 0.2, **kwargs):
#         super(ModifiedMobileViT, self).__init__(**kwargs)
#
#         # Add your additional layers here
#         self.fc1 = nn.Linear(576, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 16, bias=True)
#         self.fc4 = nn.Linear(16, 1, bias=True)
#
#         self.bn1 = nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.bn2 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#
#         # Optional: You can add activation functions or dropout as needed
#         self.dropout = nn.Dropout(drop_out)
#
#         self.hardswish = nn.Hardswish(inplace=True)
#
#     def forward(self, x):
#         # Use the forward method from the original MobileViTv3_v1
#         x = super(ModifiedMobileViT, self).forward(x)
#
#         # Pass through the new layers after the original forward pass
#         x = self.bn1(self.fc1(x))  # Apply ReLU activation
#         x = self.dropout(x)  # Apply dropout (optional)
#
#         x = self.hardswish(self.fc2(x))
#         x = self.bn2(x)
#         # x = s(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#
#         x = self.dropout(x)
#
#         x = self.hardswish((self.fc3(x)))
#
#         x = self.fc4(x)
#
#         return x


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
    """
    Network implemented in the paper:
    https://www.researchgate.net/publication/339096868_Real-Time_Facial_Affective_Computing_on_Mobile_Devices/link/6806ff17ded43315573521bc/download
    """

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


class DefocusFCFNN(nn.Module):
    """
    Feed-forward network for defocus prediction (the “trainable backend” part of the network).
    Based on the TensorFlow implementation in Waller-Lab/DeepAutofocus util/defocusnetwork.py. :contentReference[oaicite:0]{index=0}
    """

    def __init__(self,
                 input_dim: int,
                 num_hidden_units: list[int] = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                 dropout_rate: float = 0.0,
                 input_dropout_rate: float = 0.6,
                 regularization_strength: float = 0.0):
        """
        :param input_dim: size of the flattened input features (after deterministic part + normalization)
        :param num_hidden_units: list of hidden‐layer sizes
        :param dropout_rate: dropout rate applied after each hidden layer (train mode)
        :param input_dropout_rate: dropout rate applied on the normalized input (train mode)
        :param regularization_strength: L2 regularization coefficient (weight decay in optimizer)
        """
        super().__init__()
        self.input_dropout_rate = input_dropout_rate
        self.dropout_rate = dropout_rate
        self.regularization_strength = regularization_strength

        # Build hidden layers
        layers = []
        in_dim = input_dim
        for i, hidden_dim in enumerate(num_hidden_units):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            # (Dropout will be applied in forward)
            in_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)
        # Final output layer
        self.output_layer = nn.Linear(in_dim, 1)

    def forward(self, x):
        """
        Forward pass.
        :param x: tensor of shape (batch_size, input_dim)
        :return: tensor of shape (batch_size,) with predicted defocus value
        """
        # Input dropout
        if self.training and (self.input_dropout_rate > 0.0):
            x = F.dropout(x, p=self.input_dropout_rate, training=True)

        # Hidden layers + dropout
        h = x
        # If using nn.Sequential we can't easily interleave dropout layers automatically,
        # so we’ll apply dropout manually after each hidden layer block.
        for module in self.hidden_layers:
            h = module(h)
            if isinstance(module, nn.ReLU):
                # after the ReLU, apply dropout
                if self.training and (self.dropout_rate > 0.0):
                    h = F.dropout(h, p=self.dropout_rate, training=True)

        # Output layer
        out = self.output_layer(h)
        # Squeeze to shape (batch_size,)
        out = out.view(-1)
        return out


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


# class FlashAttention(nn.Module):
#     def __init__(self):
#         super(FlashAttention, self).__init__()
#         pass
#
#     def forward(self, x: Tensor):
#         q = self.query_conv(x).flatten(2).transpose(1, 2)
#         k = self.key_conv(x).flatten(2).transpose(1, 2)
#         v = self.value_conv(x).flatten(2).transpose(1, 2)
#
#         return torch.nn.functional.scaled_dot_product_attention(q, k, v)

class FlashAttention(nn.Module):
    def __init__(self, in_dim: int, num_heads: int = 1):
        super().__init__()
        head_dim = in_dim // num_heads

        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        seq_len = H * W

        # conv outputs (B, C, H, W)
        q = self.query_conv(x).reshape(B, self.num_heads, C // self.num_heads, seq_len).transpose(2, 3)
        k = self.key_conv(x).reshape(B, self.num_heads, C // self.num_heads, seq_len).transpose(2, 3)
        v = self.value_conv(x).reshape(B, self.num_heads, C // self.num_heads, seq_len).transpose(2, 3)

        # now: (B, heads, seq_len, head_dim)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # reshape back to (B, C, H, W)
        out = out.transpose(2, 3).reshape(B, C, H, W)
        return out


# class WindowAttention(nn.Module):
#     def __init__(self, dim, window_size=7, num_heads=4):
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.proj = nn.Linear(dim, dim)
#
#     def forward(self, x):
#         # x: (B, C, H, W)
#         B, C, H, W = x.shape
#         ws = self.window_size
#
#         # convert to channels-last
#         x = x.permute(0, 2, 3, 1)  # → (B,H,W,C)
#
#         # pad if needed
#         pad_h = (ws - H % ws) % ws
#         pad_w = (ws - W % ws) % ws
#         x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
#         _, Hp, Wp, _ = x.shape
#
#         # partition into windows
#         x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
#         x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
#
#         # qkv
#         qkv = self.qkv(x)
#         q, k, v = qkv.chunk(3, dim=-1)
#
#         q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
#         k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
#         v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)
#
#         # FlashAttention
#         out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
#
#         # merge heads
#         out = out.transpose(1, 2).reshape(out.size(0), ws * ws, C)
#
#         # reverse windows
#         out = out.view(B, Hp // ws, Wp // ws, ws, ws, C)
#         out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
#
#         # unpad
#         out = out[:, :H, :W, :]
#
#         # back to channels-first
#         out = out.permute(0, 3, 1, 2)  # → (B,C,H,W)
#
#         return out
#
#
# class DCNBlock(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: tuple[int, int] = (5, 5),
#                  stride: int = 1,
#                  flash: bool = True,
#                  window: bool = True):
#         super(DCNBlock, self).__init__()
#
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
#
#         if window:
#             self.attention = WindowAttention(dim=out_channels)
#
#         else:
#             if flash:
#                 self.attention = FlashAttention(in_dim=out_channels)
#
#             else:
#                 self.attention, _ = Self_Attn(in_channels, 'relu')
#
#         self.pool = nn.MaxPool2d(2)
#
#     def forward(self, x: Tensor):
#         x = F.relu(self.conv(x))
#         x = self.attention(x)
#         return self.pool(x)
#
#
# class DCNNetwork(nn.Module):
#     def __init__(self, flash: bool = True, window: bool = True):
#         super().__init__()
#
#         self.layers = nn.Sequential(
#             DCNBlock(3, 32, flash=flash, window=window),  # 3 → 32 channels
#             DCNBlock(32, 64, flash=flash, window=window),  # 32 → 64 channels
#             DCNBlock(64, 96, flash=flash, window=window),
#             DCNBlock(96, 128, flash=flash, window=window)
#         )
#
#         self.conv5 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(5, 5), stride=(1, 1))
#
#         self.fc1 = nn.Linear(in_features=1152, out_features=100)
#         self.fc2 = nn.Linear(in_features=100, out_features=2)
#
#     def forward(self, x: Tensor):
#         x = self.layers(x)
#         x = F.relu(self.conv5(x))
#
#         x = x.view(x.size()[0], -1)
#
#         x = self.fc1(x)
#         x = self.fc2(x)
#
#         return F.softmax(x)


# class DefocusingClassificationNetwork(nn.Module):
#     def __init__(self, flash: bool=True):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(1, 1))
#
#         if not flash:
#             self.attention1 = Self_Attn(in_dim=32, activation='relu')
#         else:
#             self.attention1 = FlashAttention()
#
#         self.pool1 = nn.MaxPool2d(2)
#
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1))
#
#         if not flash:
#             self.attention2 = Self_Attn(in_dim=64, activation='relu')
#         else:
#             self.attention2 = FlashAttention()
#
#         self.pool2 = nn.MaxPool2d(2)
#
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(5, 5), stride=(1, 1))
#         self.attention3 = Self_Attn(in_dim=96, activation='relu')
#         self.pool3 = nn.MaxPool2d(2)
#
#         self.conv4 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(5, 5), stride=(1, 1))
#         self.attention4 = Self_Attn(in_dim=128, activation='relu')
#         self.pool4 = nn.MaxPool2d(2)
#
#         self.conv5 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(5, 5), stride=(1, 1))
#
#         self.fc1 = nn.Linear(in_features=1152, out_features=100)
#         self.fc2 = nn.Linear(in_features=100, out_features=2)
#
#     def forward(self, x: Tensor):
#         """
#         Hight_out = (Hight_in + 2*padding - dilation*(kernel_size-1)-1)/stride +1
#         :param x:
#         :return:
#         """
#         print(x.size())
#         x = F.relu(self.conv1(x))
#         print(x.size())
#         x, _ = self.attention1(x)
#         x = self.pool1(x)
#         print(x.size())
#
#         x = F.relu(self.conv2(x))
#         x, _ = self.attention2(x)
#         x = self.pool2(x)
#
#         x = F.relu(self.conv3(x))
#         x, _ = self.attention3(x)
#         x = self.pool3(x)
#
#         x = F.relu(self.conv4(x))
#         x, _ = self.attention4(x)
#         x = self.pool4(x)
#
#         print(x.size())
#         x = F.relu(self.conv5(x))
#         print(x.size())
#         x = x.view(x.size()[0], -1)
#
#         x = self.fc1(x)
#         x = self.fc2(x)
#
#         return F.softmax(x)

# ---------------------------
# WINDOWED ATTENTION MODULE
# ---------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        ws = self.window_size

        # convert to channels-last
        x = x.permute(0, 2, 3, 1)  # → (B,H,W,C)

        # pad if needed
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = x.shape

        # partition into windows
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # FlashAttention
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # merge heads
        out = out.transpose(1, 2).reshape(out.size(0), ws * ws, C)

        # reverse windows
        out = out.view(B, Hp // ws, Wp // ws, ws, ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)

        # unpad
        out = out[:, :H, :W, :]

        # back to channels-first
        out = out.permute(0, 3, 1, 2)  # → (B,C,H,W)

        return out


# ---------------------------
# DCN BLOCK WITH FIXED ATTENTION
# ---------------------------
class DCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(5, 5), stride=1, window=True, max_pool_kernel: int=2, p=1, s=None):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        if window:
            self.attention = WindowAttention(dim=out_channels)
        else:
            raise NotImplementedError("Only windowed attention is stable for 224x224")

        self.pool = nn.MaxPool2d(max_pool_kernel, padding=p, stride=s)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.attention(x)
        x = self.pool(x)
        return x


# ---------------------------
# FINAL DCN NETWORK
# ---------------------------
class DCNNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            DCNBlock(3, 32),
            DCNBlock(32, 64),
            DCNBlock(64, 96),
            DCNBlock(96, 128, max_pool_kernel=3, p=1, s=None)
        )

        # after 4 poolings: 224 → 112 → 56 → 28 → 14
        self.conv5 = nn.Conv2d(128, 32, 5)

        # adaptive pooling avoids incorrect flatten sizes
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.layers(x)
        x = F.relu(self.conv5(x))
        # x = self.gap(x).flatten(1)  # → (B, 32)
        x = x.reshape(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ARB(nn.Module):
    def __init__(self, in_channels: int, version: int = 1):
        super().__init__()
        self._version = version

        if version == 1:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), stride=1,
                                   padding=1)
            out_channels = in_channels

        else:
            self.identity = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=(1, 1),
                                      stride=2)
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=(3, 3),
                                   stride=2, padding=1)
            out_channels = in_channels * 2

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.att1 = WindowAttention(dim=out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.att1(out)

        if self._version == 1:
            x = out + x
        else:
            x = out + self.identity(x)
        return F.relu(self.bn2(x))


class ARBBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.arb_v1 = ARB(in_channels=in_channels, version=1)
        self.arb_v2 = ARB(in_channels=in_channels, version=2)

    def forward(self, x):
        x = self.arb_v1(x)
        x = self.arb_v2(x)
        return x


class RefocusingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(4, 4))

        self.arb1 = ARBBlock(in_channels=32)
        self.arb2 = ARBBlock(in_channels=64)
        self.arb3 = ARBBlock(in_channels=128)
        self.arb4 = ARBBlock(in_channels=256)

        # self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_features=8192, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)

        x = self.arb1(x)
        x = self.arb2(x)
        x = self.arb3(x)
        x = self.arb4(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # model = ModifiedMobileViT(image_size=(224, 224), num_classes=576, mode='xx_small', drop_out=0.1)
    # x = torch.randn(2, 3, 224, 224)
    # print(model(x))

    # model = DefocusFCFNN(156672)
    # x = torch.randn(8, 156672)
    # print(model(x))

    # net = DCNNetwork()
    # x = torch.randn(1, 3, 224, 224)
    # x = x.to('cuda')
    # net.to('cuda')
    # print(net(x))

    net = DCNNetwork()
    x = torch.randn(1, 3, 224, 224)
    x = x.to('cuda')
    net.to('cuda')
    print(net(x))

    # net = ARB(64, version=2)
    # x = torch.randn(1, 64, 64, 64)
    # x = x.to('cuda')
    # net.to('cuda')
    # print(net(x).size())
