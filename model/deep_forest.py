#%%
import torch.nn as nn
from .unet2d import UNET2D
from .unet3d_adj import UNET3D_ADJ
from .unet3d_org import UNET3D_ORG
from .unet3d_adj_dec_acb import UNET3D_ADJ_DEC_ACB
from .unet3d_adj_emd_acb import UNET3D_ADJ_EMD_ACB
from .unet3d_org_enc_mid_dec_acb import UNET3D_ORG_EMD_ACB


class DeepForestSpecies(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        backbone,
        features=[64, 128, 256, 512],
    ):
        super(DeepForestSpecies, self).__init__()

        if "2d" in backbone:
            self.model = UNET2D(
                in_channels=in_channels, out_channels=out_channels, features=features
            )
        elif backbone == "3d_org":
            self.model = UNET3D_ORG(
                in_channels=in_channels, out_channels=out_channels, features=features
            )
        elif backbone == "3d_adj":
            self.model = UNET3D_ADJ(
                in_channels=in_channels, out_channels=out_channels, features=features
            )
        elif backbone == "3d_adj_dec_acb":
            self.model = UNET3D_ADJ_DEC_ACB(
                in_channels=in_channels, out_channels=out_channels, features=features
            )
        elif backbone == "3d_adj_emd_acb":
            self.model = UNET3D_ADJ_EMD_ACB(
                in_channels=in_channels, out_channels=out_channels, features=features
            )
        elif backbone == "3d_org_emd_acb":
            self.model = UNET3D_ORG_EMD_ACB(
                in_channels=in_channels, out_channels=out_channels, features=features
            )

    def forward(self, x):

        x = self.model(x)

        return x
