#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class _ACBModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(_ACBModule, self).__init__()
        self.ac_conv3d = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ac_conv3d(x)


class ACB(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ACB, self).__init__()

        self.aspp1 = _ACBModule(in_channels, out_channels, 1, 1, 0)
        self.aspp2 = _ACBModule(in_channels, out_channels, 3, rates[0], rates[0])
        self.aspp3 = _ACBModule(in_channels, out_channels, 3, rates[1], rates[1])
        self.aspp4 = _ACBModule(in_channels, out_channels, 3, rates[2], rates[2])

        self.conv1 = nn.Conv3d(
            (len(rates) + 1) * out_channels, out_channels, kernel_size=1
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        return x


#%%
# def test():
#     x = torch.randn((30, 13, 4, 32, 32))
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = ENC_ASPP(in_channels=13, out_channels=4, rates=[1, 3, 6])
#     pred = model(x)
#     print(pred.shape)
#     summary(model.to(device), (13, 4, 32, 32))


# test()

# %%
