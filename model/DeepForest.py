#%%
import torch
import torch.nn as nn
from torchsummary import summary
from .unet2d import UNET2D
from .unet3d import UNET3D
from .unet3d_dec_acb import UNET3D_DEC_ACB
from .unet3d_enc_mid_dec_acb import UNET3D_EMD_ACB


class DeepForestSpecies(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        backbone,
        features=[64, 128, 256, 512],
    ):
        super(DeepForestSpecies, self).__init__()

        if backbone == "2d":
            self.model = UNET2D(
                in_channels=in_channels, out_channels=out_channels, features=features
            )
        elif backbone == "3d":
            self.model = UNET3D(
                in_channels=in_channels, out_channels=out_channels, features=features
            )
        elif backbone == "3d_dec_acb":
            self.model = UNET3D_DEC_ACB(
                in_channels=in_channels, out_channels=out_channels, features=features
            )
        elif backbone == "3d_enc_mid_dec_acb":
            self.model = UNET3D_EMD_ACB(
                in_channels=in_channels, out_channels=out_channels, features=features
            )

    def forward(self, x):

        x = self.model(x)

        return x


#%%

# def test():
#     x = torch.randn((30, 29, 32, 32))
#     model = DeepForestAge(in_channels=29, out_channels=64)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     preds = model(x)
#     print(preds.shape)

#     # summary(model.to(device), (29, 32, 32))


# if __name__ == "__main__":
#     test()

# %%
