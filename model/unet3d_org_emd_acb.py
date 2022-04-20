#%%
import torch
import torch.nn as nn
from torchsummary import summary
from .utils import SingleConv3D
from .acb import ACB

RATES_ED1 = [3, 6, 9]
RATES_B = [2, 3, 4]


class _DoubleConv3D_ENC(nn.Module):
    def __init__(self, in_channels, out_channels, ft_level=1):
        super(_DoubleConv3D_ENC, self).__init__()
        self.double_conv = nn.Sequential(
            SingleConv3D(in_channels, int(out_channels / 2)),
            SingleConv3D(int(out_channels / 2), out_channels),
        )
        self.ft_level = ft_level
        if self.ft_level == 1:
            self.acb = ACB(out_channels, out_channels, rates=RATES_ED1)

    def forward(self, x):
        x = self.double_conv(x)
        if self.ft_level == 1:
            x = self.acb(x)
        return x


class _DoubleConv3D_DEC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DoubleConv3D_DEC, self).__init__()
        self.double_conv = nn.Sequential(
            SingleConv3D(in_channels, out_channels),
            SingleConv3D(out_channels, out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNET3D_ORG_EMD_ACB(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNET3D_ORG_EMD_ACB, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        ft_level = 1

        # Down part of Net
        for feature in features:
            self.downs.append(_DoubleConv3D_ENC(in_channels, feature, ft_level))
            in_channels = feature
            ft_level = ft_level + 1

        # Up part of Net
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(_DoubleConv3D_DEC(feature * 2, feature))

        self.mid_conv = _DoubleConv3D_ENC(features[-1], features[-1] * 2, ft_level=1)
        # self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self.final_conv = ACB(features[0], out_channels, rates=RATES_ED1)

    def forward(self, x):

        skip_connections = []

        # for down, down_pool_conv in zip(self.downs, self.down_pool_convs):
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.mid_conv(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)

        return x


# %%
# def test():
#     x = torch.randn((30, 9, 4, 32, 32))
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = UNET3D_ORG_EMD_ACB(in_channels=9, out_channels=4, features=[64, 128])
#     pred = model(x)
#     print(pred.shape)
#     summary(model.to(device), (9, 4, 32, 32))


# test()

# %%
