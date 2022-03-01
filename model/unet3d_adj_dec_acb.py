#%%

import torch
import torch.nn as nn
from torchsummary import summary
from .utils import TripleConv3D, DoubleConv3D
from .acb import ACB

RATES = [3, 6, 9]


class UNET3D_ADJ_DEC_ACB(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128]):
        super(UNET3D_ADJ_DEC_ACB, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        for ft in features:
            self.encoder.append(TripleConv3D(in_channels, ft))
            in_channels = ft
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

        self.mid_conv = DoubleConv3D(features[-1], features[-1] * 2)
        self.aspp = ACB(features[0], out_channels, rates=RATES)
        # self.final_conv = BasicConv3D(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.mid_conv(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        x = self.aspp(x)

        return x


#%%
# def test():
#     x = torch.randn((30, 13, 4, 32, 32))
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = UNET3D_ADJ_DEC_ACB(in_channels=13, out_channels=4, features=[64, 128])
#     pred = model(x)
#     print(pred.shape)
#     summary(model.to(device), (13, 4, 32, 32))


# test()

# %%
