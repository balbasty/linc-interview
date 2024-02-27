from loaders import get_train_eval_test
from typing import Literal
from torch import nn
from torch.nn import functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import os
import datetime


class ConvBlock(nn.Module):
    """
    A convolutional block that applies a specified number of convolutional layers
    with the given activation function.

    Attributes:
        conv (nn.Sequential): Sequential container of convolutional layers.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        nb_conv (int): Number of convolutional layers to apply.
        activation (str): Type of activation function to apply ('ReLU' or 'ELU').
    """
    def __init__(self, in_channels: int, out_channels: int, nb_conv: int, activation: str):
        super().__init__()
        layers = []
        for _ in range(nb_conv):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
            if activation == 'ReLU':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'ELU':
                layers.append(nn.ELU(inplace=True))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying convolutional layers.
        """
        return self.conv(x)


class DownsampleBlock(nn.Module):
    """
    A block for downsampling an input tensor.

    This block supports different downsampling strategies: max pooling, interpolation, and strided convolution.

    Attributes:

        channels (int): The number of channels in the input tensor to be preserved.

        pool_type (str): The type of downsampling to apply. Supported values are 'max' for max pooling,
                         'interpolate' for bilinear interpolation, and 'conv' for strided convolution.

    Args:
        pool_type (str): Specifies the downsampling technique to use. Defaults to 'max'.
    """
    def __init__(self, channels: int, pool_type: str = 'max'):
        super().__init__()
        self.channels = channels
        self.pool_type = pool_type
        self.conv_down = None
        if self.pool_type == 'conv':
            self.conv_down = nn.Conv2d(self.channels, self.channels, kernel_size=(2, 2), stride=(2, 2), padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the specified downsampling operation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to downsample.

        Returns:
            torch.Tensor: The downsampled tensor.
        """
        if self.pool_type == 'interpolate':
            return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        elif self.pool_type == 'conv':
            return self.conv_down(x)
        else:  # Default to max pooling if pool_type is not recognized
            return F.max_pool2d(x, kernel_size=2, stride=2)


class UpsampleBlock(nn.Module):
    """
    A block for upsampling an input tensor.

    This block can perform upsampling using either a transposed convolution or bilinear interpolation.

    Attributes:
        in_channels (int): The number of channels in the input tensor.
        out_channels (int): The number of channels in the output tensor.
        pool_type (str): The type of upsampling to apply. Supported values are 'conv' for transposed convolution
                         and any other value defaults to bilinear interpolation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor after upsampling.
        pool_type (str): Specifies the upsampling technique to use. Defaults to 'max'.
    """
    def __init__(self, in_channels: int, out_channels: int, pool_type: str = 'max'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_type = pool_type
        self.adjust_channels = None
        self.conv_up = None
        if self.pool_type == 'conv':
            self.conv_up = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=(2, 2), stride=(2, 2))
        else:
            self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the specified upsampling operation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to upsample.

        Returns:
            torch.Tensor: The upsampled tensor.
        """
        if self.pool_type == 'conv':
            return self.conv_up(x)
        else:  # Default to bilinear interpolation if pool_type is not 'conv'
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            return self.adjust_channels(x)


class Backbone(nn.Module):
    """
    Implements a modular 2D U-Net architecture. The UNet consists of a contracting path to capture context and a
    symmetric expanding path that enables precise localization.

    The architecture is defined by a series of convolutional blocks at different levels, with downsampling operations
    to decrease the spatial dimensions and upsampling operations to restore the dimensions. Feature maps from the
    contracting path are concatenated with those of the expanding path to preserve high-resolution features.

    Attributes:
        down_blocks (nn.ModuleList): A list of modules for the contracting path.
        up_blocks (nn.ModuleList): A list of modules for the expanding path.
        inp_conv (ConvBlock): Initial convolution block to process input.
        out_conv (nn.Conv2d): Convolution layer to generate final output.

    Args:
        inp_channels (int): Number of input channels. Default is 2.
        out_channels (int): Number of output channels. Default is 2.
        nb_features (int): Number of features in the first convolutional block. Default is 16.
        mul_features (int): Factor to increase features by at each level of the network. Default is 2.
        nb_levels (int): Number of levels in the U-Net architecture. Default is 3.
        nb_conv_per_level (int): Number of convolutional layers per level. Default is 2.
        activation (Literal['ReLU', 'ELU']): Activation function to use. Default is 'ReLU'.
        pool (Literal['interpolate', 'conv', 'pool']): Pooling operation for downsampling. Default is 'interpolate'.

    The architecture schematic:
    C -[conv xN]-> F ----------------------(cat)----------------------> 2*F -[conv xN]-> Cout
                   |                                                     ^
                   v                                                     |
                  F*m -[conv xN]-> F*m  ---(cat)---> 2*F*m -[conv xN]-> F*m
                                    |                  ^
                                    v                  |
                                  F*m*m -[conv xN]-> F*m*m
    """

    def __init__(
            self,
            inp_channels: int = 2,
            out_channels: int = 2,
            nb_features: int = 16,
            mul_features: int = 2,
            nb_levels: int = 3,
            nb_conv_per_level: int = 2,
            activation: Literal['ReLU', 'ELU'] = 'ReLU',
            pool: Literal['interpolate', 'conv', 'pool'] = 'interpolate'
    ):
        super().__init__()

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        channels = nb_features
        self.inp_conv = ConvBlock(inp_channels, channels, nb_conv_per_level, activation)

        # Constructing the contracting path
        for _ in range(nb_levels - 1):  # Exclude the last level as it doesn't apply downsample
            self.down_blocks.append(DownsampleBlock(channels, pool))
            self.down_blocks.append(ConvBlock(channels, channels * mul_features, nb_conv_per_level, activation))
            channels *= mul_features

        # Constructing the expanding path
        for _ in range(nb_levels - 1):
            self.up_blocks.append(UpsampleBlock(channels, channels // mul_features, pool))
            self.up_blocks.append(ConvBlock(channels, channels // mul_features, nb_conv_per_level, activation))
            channels //= mul_features

        self.out_conv = nn.Conv2d(channels, out_channels, kernel_size=(1, 1))

    def forward(self, inp):
        """
        Forward pass of the U-Net model.

        Args:
            inp (torch.Tensor): Input tensor of shape (B, inp_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of the model of shape (B, out_channels, H, W).
        """
        connections = []
        x = self.inp_conv(inp)

        # Contracting path
        for i in range(0, len(self.down_blocks), 2):
            connections.append(x)
            x = self.down_blocks[i](x)
            x = self.down_blocks[i + 1](x)

        # Expanding path
        for i in range(0, len(self.up_blocks), 2):
            upsample = self.up_blocks[i]
            conv_block = self.up_blocks[i + 1]
            x = upsample(x)
            x = torch.cat([x, connections.pop()], dim=1)
            x = conv_block(x)

        out = self.out_conv(x)
        return out


class VoxelMorph(nn.Module):
    """
    Implements the VoxelMorph model for image registration tasks, utilizing a backbone network.

    This class defines the core functionality of VoxelMorph, including the forward pass through the backbone,
    deformation of moving images based on displacement fields, and calculation of the registration loss.

    Attributes:
        backbone (Backbone): An instance of the Backbone class used for feature extraction and image transformation.
    Args:
        **backbone_parameters: Variable keyword arguments passed to initialize the Backbone network.
    """

    def __init__(self, **backbone_parameters):
        super().__init__()
        self.backbone = Backbone(**backbone_parameters)

    def forward(self, fixmov: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the transformed moving image.

        Args:
            fixmov (torch.Tensor): The fixed and moving images concatenated along the channel dimension.

        Returns:
            torch.Tensor: The output from the backbone network.
        """
        return self.backbone(fixmov)

    def deform(self, mov: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """
        Deforms the moving image based on the displacement field.

        Args:
            mov (torch.Tensor): The moving image.
            disp (torch.Tensor): The displacement field.

        Returns:
            torch.Tensor: The deformed moving image.
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
        return F.grid_sample(mov, disp, mode='bilinear', padding_mode='zeros', align_corners=True)

    def membrane(self, disp: torch.Tensor) -> torch.Tensor:
        """
        Calculates the membrane energy of the displacement field as a regularization term.

        Args:
            disp (torch.Tensor): The displacement field.

        Returns:
            torch.Tensor: The membrane energy of the displacement field.
        """
        return ((disp[:, :, 1:, :] - disp[:, :, :-1, :]).square().mean() +
                (disp[:, :, :, 1:] - disp[:, :, :, :-1]).square().mean())

    def loss(self, fix: torch.Tensor, mov: torch.Tensor, disp: torch.Tensor, lam=0.1) -> torch.Tensor:
        """
        Computes the loss for image registration, combining image similarity and displacement field smoothness.

        Args:
            fix (torch.Tensor): The fixed image.
            mov (torch.Tensor): The moving image.
            disp (torch.Tensor): The displacement field.
            lam (float, optional): Regularization weight for the membrane energy term. Defaults to 0.1.

        Returns:
            torch.Tensor: The total loss for the image registration.
        """
        moved = self.deform(mov, disp)
        loss = nn.MSELoss()(moved, fix)
        loss += self.membrane(disp) * lam
        return loss


def tensorboard_visualization(model, loader, device, epoch, writer, num_examples=4):
    """
    Visualizes model predictions and displacement fields using TensorBoard.

    This function takes a batch of fixed and moving images, performs image registration using the model, and
    visualizes the fixed images, moving images, magnitude of displacement fields, and moved images side by side
    for a selected number of examples.

    Args:
        model (nn.Module): The trained model for image registration.
        loader (DataLoader): DataLoader providing batches of fixed and moving images.
        device (str): The device to perform computations on.
        epoch (int): The current epoch number, used for labeling in TensorBoard.
        writer (SummaryWriter): The TensorBoard writer object.
        num_examples (int, optional): Number of examples to visualize. Defaults to 4.
    """
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        fixmov = batch.to(device)
        if num_examples > len(fixmov):
            num_examples = len(fixmov)
        fixmov = fixmov[:num_examples]
        disp = model(fixmov)
        fix, mov = fixmov[:, 0, :, :].unsqueeze(1), fixmov[:, 1, :, :].unsqueeze(1)
        moved = model.deform(mov, disp)
        magnitude = torch.sqrt(disp[:, 0, :, :] ** 2 + disp[:, 1, :, :] ** 2)
        examples = torch.cat([fix, mov, magnitude.unsqueeze(1), moved], 0)
        grid = vutils.make_grid(examples, nrow=num_examples, normalize=True, scale_each=True)
        writer.add_image('Visualizations/Epoch_{}'.format(epoch), grid, epoch)
    model.train()


def train(model, train_loader, eval_loader, optimizer, epochs=10, visualize_every=None, device='cuda', save_weights=True):
    """
    Trains the model using the provided training and evaluation data loaders.

    Iterates over the dataset for a given number of epochs, performing backpropagation and optimizing the model's
    weights. Optionally, it visualizes intermediate results and evaluates the model on a validation set.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        eval_loader (DataLoader): DataLoader for the evaluation dataset.
        optimizer (Optimizer): The optimization algorithm.
        epochs (int, optional): Total number of training epochs. Defaults to 10.
        visualize_every (int, optional): Interval of epochs after which to visualize predictions.
        device (str, optional): Device to train the model on ('cuda' or 'cpu'). Defaults to 'cuda'.
        save_weights (bool, optional): Whether to save the model weights after training. Defaults to True.
    """
    model.train()
    model.to(device)
    size = len(train_loader.dataset)
    num_batches = len(train_loader)
    update_interval = max(1, num_batches // 10)
    start_time = datetime.datetime.now()
    start_str_time = start_time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('runs', start_str_time)
    weights_dir = os.path.join('runs', start_str_time + '.pth')
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}")
        for batch_idx, fixmov in enumerate(train_loader):
            fixmov = fixmov.to(device)
            optimizer.zero_grad()
            disp = model(fixmov)
            fix, mov = fixmov[:, 0, :, :].unsqueeze(1), fixmov[:, 1, :, :].unsqueeze(1)
            loss = model.loss(fix, mov, disp)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % update_interval == 0 or batch_idx == num_batches - 1:
                loss, current = loss.item(), batch_idx * len(fixmov) + len(fixmov)
                print(f"Loss = {loss: .4f}  [{current: >5d}/{size: >5d}]")
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
        writer.add_scalar("Loss/train", avg_loss, epoch)
        if visualize_every and (epoch + 1) % visualize_every == 0:
            print(f'Visualizing predictions for epoch {epoch + 1}...')
            tensorboard_visualization(model, train_loader, device, epoch, writer, num_examples=4)
        if eval_loader:
            model.eval()
            eval_loss = 0
            with torch.no_grad():
                for fixmov in eval_loader:
                    fixmov = fixmov.to(device)
                    disp = model(fixmov)
                    fix, mov = fixmov[:, 0, :, :].unsqueeze(1), fixmov[:, 1, :, :].unsqueeze(1)
                    eval_loss += model.loss(fix, mov, disp).item()
            eval_loss /= len(eval_loader)
            print(f'Eval Loss: {eval_loss: .4f}')
            writer.add_scalar("Loss/val", eval_loss, epoch)
            model.train()
    writer.close()
    if save_weights:
        torch.save(model.state_dict(), weights_dir)


def test(model, test_loader, device='cuda'):
    """
    Evaluates the model on a test dataset.

    Computes the average loss over the test dataset without performing any backpropagation. This function is intended
    to be used after model training is complete to assess performance on unseen data.

    Args:
        model (nn.Module): The trained model to be evaluated.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str, optional): The device to perform computations on. Defaults to 'cuda'.
    """
    model.eval()
    model.to(device)
    test_loss = 0
    with torch.no_grad():
        for fixmov in test_loader:
            fixmov = fixmov.to(device)
            disp = model(fixmov)
            fix, mov = fixmov[:, 0, :, :].unsqueeze(1), fixmov[:, 1, :, :].unsqueeze(1)
            loss = model.loss(fix, mov, disp)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss: .4f}')


if __name__ == '__main__':
    # Model configuration parameters
    # Ready for fine-tuning, e.g., with grid search
    backbone_parameters = dict(inp_channels=2, out_channels=2, nb_features=16, mul_features=2, nb_levels=3,
                               nb_conv_per_level=2, pool='interpolate', activation='ReLU')

    # Initialize the VoxelMorph model with the specified parameters
    model = VoxelMorph(**backbone_parameters)

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Load training, evaluation, and test datasets
    train_loader, eval_loader, test_loader = get_train_eval_test()

    # Determine the computation device based on CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train the model
    train(model, train_loader, eval_loader, optimizer, epochs=10, visualize_every=1, device=device, save_weights=True)

    # Test the model
    test(model, test_loader, device=device)

    # tensorboard --logdir=runs