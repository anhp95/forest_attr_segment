import torch
import torch.nn as nn


class SingleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SingleConv3D, self).__init__()
        self.single_conv3d = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv3d(x)


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            SingleConv3D(in_channels, out_channels),
            SingleConv3D(out_channels, out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)


class TripleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConv3D, self).__init__()
        self.triple_conv = nn.Sequential(
            SingleConv3D(in_channels, out_channels),
            SingleConv3D(out_channels, out_channels),
            SingleConv3D(out_channels, out_channels),
        )

    def forward(self, x):
        return self.triple_conv(x)
