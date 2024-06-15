from loaders import get_train_eval_test
from typing import Literal
from torch import nn
from torch.nn import functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

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
            activation: Literal['ReLU', 'ELU'] = 'ReLU',
            pool: Literal['interpolate', 'max', 'conv'] = 'interpolate',
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
            If 'max' use max pooling.
        activation : {'ReLU', 'ELU'}
            Type of activation
        """

        super(Backbone, self).__init__()

        self.nb_levels = nb_levels
        self.activation = activation
        self.pool = pool

        if activation == 'ReLU':
            self.activation_fn = nn.ReLU(inplace=True)
        elif activation == 'ELU':
            self.activation_fn = nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Encoder
        self.encoders = nn.ModuleList()
        in_channels = inp_channels
        out_channels_list = []

        for level in range(nb_levels):
            level_out_channels = nb_features * (mul_features ** level)
            out_channels_list.append(level_out_channels)
            self.encoders.append(self._make_level(in_channels, level_out_channels, nb_conv_per_level))
            in_channels = level_out_channels

        # Bottleneck
        self.bottleneck = self._make_level(in_channels, in_channels * mul_features, nb_conv_per_level)
        in_channels = in_channels * mul_features

        # Decoder
        self.decoders = nn.ModuleList()
        for level in range(nb_levels - 1, -1, -1):
            level_out_channels = out_channels_list[level]
            self.decoders.append(self._make_level(in_channels, level_out_channels, nb_conv_per_level))
            in_channels = level_out_channels

        # Final Convolution
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def _make_level(self, in_channels, out_channels, nb_conv):
        layers = []
        for _ in range(nb_conv):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(self.activation_fn)
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
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
        enc_features = []
        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            enc_features.append(x)
            x = self._downsample(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for decoder, enc in zip(self.decoders, reversed(enc_features)):
            x = self._upsample(x)
            x = torch.cat([x, enc], dim=1)
            x = decoder(x)
        x = self.final_conv(x)
        return x
    
    def _downsample(self, x):
        if self.pool == 'interpolate':
            return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        elif self.pool == 'conv':
            return nn.Conv2d(x.size(1), x.size(1), kernel_size=2, stride=2).to(x.device)(x)
        elif self.pool == 'max':
            return F.max_pool2d(x, kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unsupported pool method: {self.pool}")

    def _upsample(self, x):
        return nn.ConvTranspose2d(x.size(1), x.size(1) // 2, kernel_size=2, stride=2).to(x.device)(x)

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


def train(model, train_loader, val_loader, writer, num_epochs=100, learning_rate=1e-3, lam=0.1, device='cuda'):
    """
    Train the VoxelMorph model.

    Parameters:
    model (nn.Module): The VoxelMorph model to train.
    train_loader (DataLoader): DataLoader for the training data.
    val_loader (DataLoader): DataLoader for the validation data.
    writer (SummaryWriter): Tensorboard writer. 
    num_epochs (int): Number of epochs to train.
    learning_rate (float): Learning rate for the optimizer.
    lam (float): Regularization parameter for the loss function.
    device (str): Device to use for training ('cuda' or 'cpu').

    Returns:
    None
    """

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            batch = batch.to(device)
            moving, fixed = batch[:, 1:2], batch[:, 0:1]  # Extract moving and fixed images
            optimizer.zero_grad()
            fixmov = torch.cat([fixed, moving], dim=1)
            disp = model(fixmov)
            loss = model.loss(fixed, moving, disp, lam)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = batch.to(device)
                moving, fixed = batch[:, 1:2], batch[:, 0:1]  # Extract moving and fixed images
                fixmov = torch.cat([fixed, moving], dim=1)
                disp = model(fixmov)
                loss = model.loss(fixed, moving, disp, lam)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Log the losses to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/eval', val_loss, epoch)
        writer.add_images('Fixed', fixed[0:1], epoch)
        writer.add_images('Moving', moving[0:1], epoch)
        writer.add_images('Deformed', model.deform(moving, disp)[0:1], epoch)
        writer.add_images('Displacement_X', disp[0:1, 0:1, :, :], epoch)
        writer.add_images('Displacement_Y', disp[0:1, 1:2, :, :], epoch)


def test(model, test_loader, lam=0.1, device='cuda'):
    """
    Test the VoxelMorph model.

    Parameters:
    model (nn.Module): The VoxelMorph model to test.
    test_loader (DataLoader): DataLoader for the test data.
    lam (float): Regularization parameter for the loss function.
    device (str): Device to use for testing ('cuda' or 'cpu').

    Returns:
    None
    """

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            moving, fixed = batch[:, 1:2], batch[:, 0:1]  # Extract moving and fixed images
            fixmov = torch.cat([fixed, moving], dim=1)
            disp = model(fixmov)
            loss = model.loss(fixed, moving, disp, lam)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()

    # Training hyperparameteres
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--lam', default=0.1, type=float)

    # Model Hyperparameters
    parser.add_argument('--nb_features', default=16, type=int)
    parser.add_argument('--mul_features', default=2, type=int)
    parser.add_argument('--nb_levels', default=3, type=int)
    parser.add_argument('--nb_conv_per_level', default=2, type=int)
    parser.add_argument('--activation', default='ReLU', type=str)
    parser.add_argument('--pool', default='conv', type=str)
    
    args = parser.parse_args()
    
    # Initialize the model
    backbone_parameters = {
        'nb_features': args.nb_features,
        'mul_features': args.mul_features,
        'nb_levels': args.nb_levels,
        'nb_conv_per_level': args.nb_conv_per_level,
        'activation': args.activation,
        'pool': args.pool,
    }

    trainset, evalset, testset = get_train_eval_test()

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    model = VoxelMorph(**backbone_parameters)

    # Train the model
    train(model, trainset, evalset, writer, num_epochs=args.num_epochs, learning_rate=args.learning_rate, lam=args.lam)

    # # Test the model
    test(model, testset, lam=args.lam)

    writer.close()

if __name__ == '__main__':
    main()