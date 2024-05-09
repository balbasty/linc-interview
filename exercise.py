from loaders import get_train_eval_test
from typing import Literal
from torch import nn
from torch.nn import functional as F
import torch


class Backbone(nn.Module):
    """A 2D UNet

    ```
    C -[conv xN]-> F ----------------------(cat)----------------------> 2*F -[conv xN]-> Cout
                   |                                                     ^
                   v                                                     |
                  F*m -[conv xN]-> F*m  ---(cat)---> 2*F*m -[conv xN]-> F*m
                                    |                  ^
                                    v                  |
                                  F*m*m -[conv xN]-> F*m*m
    ```
    """  # noqa: E501

    def __init__(
            self,
            inp_channels: int = 2,
            out_channels: int = 2,
            nb_features: int = 16,
            mul_features: int = 2,
            nb_levels: int = 3,
            nb_conv_per_level: int = 2,
            # Implementing the following switches is optional.
            # If not implementing the switch, choose the mode you prefer.
            activation: Literal['ReLU', 'ELU'] = 'ReLU',
            pool: Literal['interpolate', 'conv'] = 'interpolate',
    ):
        """
        Parameters
        ----------
        inp_channels : int
            Number of input channels
        out_channels : int
            Number of output chanels
        nb_features : int
            Number of features at the finest level
        mul_features : int
            Multiply the number of features by this number
            each time we go down one level.
        nb_conv_per_level : int
            Number of convolutional layers at each level.
        pool : {'interpolate', 'conv'}
            Method used to go down/up one level.
            If `interpolate`, use `torch.nn.functional.interpolate`.
            If `conv`, use strided convolutions on the way down, and
            transposed convolutions on the way up.
        activation : {'ReLU', 'ELU'}
            Type of activation
        """
        raise NotImplementedError

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (B, in_channels, X, Y)
            Input tensor

        Returns
        -------
        out : (B, out_channels, X, Y)
            Output tensor
        """
        raise NotImplementedError


class VoxelMorph(nn.Module):
    """
    Construct a voxelmorph network with the given backbone
    """

    def __init__(self, **backbone_parameters):
        """
        Parameters
        ----------
        backbone_parameters : dict
            Parameters of the `Backbone` class
        """
        super().__init__()
        self.backbone = Backbone(2, 2, **backbone_parameters)

    def forward(self, fixmov):
        """
        Predict a displacement field from a fixed and moving images

        Parameters
        ----------
        fixmov : (B, 2, X, Y) tensor
            Input fixed and moving images, stacked along
            the channel dimension

        Returns
        -------
        disp : (B, 2, X, Y) tensor
            Predicted displacement field
        """
        return self.backbone(fixmov)

    def deform(self, mov, disp):
        """
        Deform the image `mov` using the displacement field `disp`

        Parameters
        ----------
        moving : (B, 1, X, Y) tensor
            Moving image
        disp : (B, 2, X, Y) tensor
            Displacement field

        Returns
        -------
        moved : (B, 1, X, Y) tensor
            Moved image
        """
        opt = dict(dtype=mov.dtype, device=mov.device)
        disp = disp.clone()
        nx, ny = mov.shape[-2:]

        # Rescale displacement to conform to torch conventions with
        # align_corners=True

        # 0) disp contains relative displacements in voxels
        mx, my = torch.meshgrid(torch.arange(nx, **opt),
                                torch.arange(ny, **opt), indexing='ij')
        disp[:, 0] += mx
        disp[:, 1] += my
        # 1) disp contains absolute coordinates in voxels
        disp[:, 0] *= 2 / (nx - 1)
        disp[:, 1] *= 2 / (ny - 1)
        # 2) disp contains absolute coordinates in (0, 2)
        disp -= 1
        # 3) disp contains absolute coordinates in (-1, 1)

        # Permute/flip to conform to torch conventions
        disp = disp.permute([0, 2, 3, 1])
        disp = disp.flip([-1])

        # Transform moving image
        return F.grid_sample(
            mov, disp,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )

    def membrane(self, disp):
        """
        Compute the membrane energy of the displacement field
        (the average of its squared spatial gradients)
        """
        return (
            (disp[:, :, 1:, :] - disp[:, :, :-1, :]).square().mean() +
            (disp[:, :, :, 1:] - disp[:, :, :, :-1]).square().mean())

    def loss(self, fix, mov, disp, lam=0.1):
        """
        Compute the regularized loss (mse + membrane * lam)

        Parameters
        ----------
        fix : (B, 1, X, Y) tensor
            Fixed image
        mov : (B, 1, X, Y) tensor
            Moving image
        disp : (B, 2, X, Y) tensor
            Displacement field
        lam : float
            Regularization
        """
        moved = self.deform(mov, disp)
        loss = nn.MSELoss()(moved, fix)
        loss += self.membrane(disp) * lam
        return loss


trainset, evalset, testset = get_train_eval_test()


def train(*args, **kwargs):
    """
    A training function
    """
    raise NotImplementedError('Implement this function yourself')


def test(*args, **kwargs):
    """
    A testing function
    """
    raise NotImplementedError('Implement this function yourself')
