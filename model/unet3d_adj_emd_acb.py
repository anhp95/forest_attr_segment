#%%

import torch
import torch.nn as nn
from .acb import ACB
from .utils import SingleConv3D, TripleConv3D
from torchsummary import summary

RATES_ED1 = [3, 6, 12]
RATES_B = [2, 3, 4]


class _Middle_ACB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_Middle_ACB, self).__init__()

        self.bottle_neck_aspp = nn.Sequential(
            SingleConv3D(in_channels, out_channels, kernel_size=3),
            ACB(out_channels, out_channels, rates=RATES_B),
        )

    def forward(self, x):
        return self.bottle_neck_aspp(x)


class _Encoder_ACB(nn.Module):
    def __init__(self, in_channels, out_channels, ft_level=1):
        super(_Encoder_ACB, self).__init__()

        self.conv1 = SingleConv3D(in_channels, out_channels, 3)
        self.conv2 = SingleConv3D(out_channels, out_channels, 3)
        self.conv3 = SingleConv3D(out_channels, out_channels, 3)
        self.ft_level = ft_level
        if self.ft_level == 1:
            self.aspp = ACB(out_channels, out_channels, rates=RATES_ED1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.ft_level == 1:
            x = self.aspp(x)
        return x


class UNET3D_ADJ_EMD_ACB(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128]):
        super(UNET3D_ADJ_EMD_ACB, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        ft_level = 1

        for ft in features:
            self.encoder.append(_Encoder_ACB(in_channels, ft, ft_level))
            in_channels = ft
            ft_level = ft_level + 1
        for ft in reversed(features):
            self.decoder.append(
                nn.ConvTranspose3d(
                    ft * 2,
                    ft,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.decoder.append(TripleConv3D(ft * 2, ft))

        self.bottle_neck = _Middle_ACB(features[-1], features[-1] * 2)
        self.aspp = ACB(features[0], out_channels, rates=RATES_ED1)
        # self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottle_neck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        x = self.aspp(x)
        # x = self.final_conv(x)

        return x


# %%
# def test():
#     x = torch.randn((30, 9, 4, 32, 32))
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = UNET3D_ADJ_EMD_ACB(9, 4, [64, 128])
#     pred = model(x)
#     print(pred.shape)
#     summary(model.to(device), (9, 4, 32, 32))


# test()

# %%
