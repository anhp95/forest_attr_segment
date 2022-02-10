#%%
import torch
import torch.nn as nn
from torchsummary import summary
from .utils import DoubleConv3D, TripleConv3D


class UNET3D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNET3D, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down part of Net
        for feature in features:
            self.downs.append(TripleConv3D(in_channels, feature))
            in_channels = feature

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
            self.ups.append(TripleConv3D(feature * 2, feature))

        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):

        skip_connections = []

        # for down, down_pool_conv in zip(self.downs, self.down_pool_convs):
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # if x.shape != skip_connection.shape:
            #     print (x.shape, skip_connection.shape)
            #     x = TF.resize(x, size=skip_connection.shape[3:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            # print(concat_skip.shape)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)

        return x


# %%
# def test():
#     x = torch.randn((30, 9, 4, 32, 32))
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = UNET3D(in_channels=9, out_channels=4, features=[64, 128])
#     pred = model(x)
#     print(pred.shape)
#     summary(model.to(device), (9, 4, 32, 32))


# test()

# %%
