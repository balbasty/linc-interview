import os
import torch
import pytorch_lightning as pl
from torch import nn
from blocks import Encoder, Decoder
from torch.nn import functional as F
from lightning_model_ import TrainerModule
from pytorch_lightning.callbacks import ModelCheckpoint

class Backbone(nn.Module):
    """
        2D U-Net flexibly parameterized

        Args:
            inp_channels (int): number of input channels
            out_channels (int): number of output channels
            nb_features (list) = number of features per layer
            nb_conv_per_level (int) = number of total convolutional blocks per layer 
            pooling (bool): True if use MaxPool2D
            scale_factor (int): upsampling scale factor
            act_function:(str): activate function (ReLU, ELU or LeakyReLU)
            last_act_function = activate function used on last layer ("softmax", "sigmoid" or "no_act_function")
            unpooling: upsampling function ("transp_conv", "interpolation" or "no_pool")
            mode: interpolation algorithm used if unpooling == "interpolation" 
                    ('nearest', 'linear', 'bilinear', 'bicubic' or 'trilinear')
            
        Return:
            output tensor
    """
    def __init__(
            self,
            inp_channels = 2,
            out_channels = 2,
            nb_features = [32,64,128],
            nb_conv_per_level= 2,
            scale_factor = 2,
            activation = 'ReLU',
            last_act_function = 'sigmoid',
            unpooling = "transp_conv", 
            mode = "nearest"
    ):

        super().__init__()
        if activation == 'ReLU':
            act_function = nn.ReLU()
        elif activation == 'LeakyReLU':
            act_function = nn.LeakyReLU()
        elif activation == 'ELU':
            act_function = nn.ELU()
        else:
            raise ValueError('activation function must be ReLU, ELU or LeakyReLU')
        
        last_act_function_list = ["softmax", "sigmoid", "ReLU", "no_act_function"]
        unpooling_list = ["transp_conv", "interpolation"]
        mode_list = ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']
        assert isinstance(inp_channels, int), 'inp_channels should be a interger'
        assert isinstance(out_channels, int), 'out_channels should be a interger'
        assert isinstance(nb_features, list), 'inp_channels should be a list'
        assert isinstance(nb_conv_per_level, int), 'nb_conv_per_level be a interger'
        assert isinstance(scale_factor, int), 'scale_factor should be a interger'
        assert isinstance(scale_factor, int), 'scale_factor should be a interger'
        assert last_act_function in last_act_function_list, 'last_act_function should be "softmax", "sigmoid","ReLU" or "no_act_function"'
        assert unpooling in unpooling_list, 'unpooling should be "transp_conv", "interpolation"'
        assert mode in mode_list, 'mode should be "nearest", "linear", "bilinear", "bicubic" or "trilinear"'
        
        self.encoders = Encoder(inp_channels, nb_features, nb_conv_per_level, scale_factor, act_function)
        self.decoders = Decoder(out_channels, nb_features, nb_conv_per_level, scale_factor, unpooling, mode, act_function,last_act_function)
    
        
    def forward(self, inp):

        (encoders_feat, inp_decoder) = self.encoders(inp)    
        out = self.decoders(encoders_feat, inp_decoder)
        return out

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
        self.backbone = Backbone(**backbone_parameters)
                
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
        mov = mov[:,None,:,:]

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

def set_trainer(checkpoint_path, accelerator, devices, max_epochs = 300, accumulate_grad_batches = 1,min_delta = 0.001, patience = 50):
    """
    Lightning model function
    
    Args:

    max_epochs (int): max epochs used for training
    accumulate_grad_batches (int): accumulate gradients before running optimizer
    checkpoint_path (str): path to save the checkpoint
    patience (int): number of epochs the training will run before stop if there
                    is no improvement on the monitored metric

    Return:
    Lightning model used for training and testing
    """

    #Define checkpoint path
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    #callback to save checkpoint
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                        monitor="val_loss",
                                        mode="min",
                                        save_last = True,
                                        save_top_k=5)
    #Stop criteria (early stop)                                        
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", 
                                min_delta=min_delta, 
                                patience=patience, 
                                verbose=True, mode="min")

    trainer = pl.Trainer(
                        max_epochs=max_epochs,
                        accumulate_grad_batches=accumulate_grad_batches,accelerator = accelerator,
                        devices = devices,
                        callbacks=[checkpoint_callback, early_stop_callback],
                        )

    return trainer

def train(model, trainset, evalset, testset, lr,lam,accelerator, devices, checkpoint_path,  max_epochs=300, accumulate_grad_batches = 1,min_delta = 0.001,patience = 50):
    """
    A training function

    Args:
    trainer: model trainer
    trainer_module: lightning model 
    """
    #define lightning model
    trainer_module = TrainerModule(model = model,
                    trainset=trainset,
                    evalset=evalset, 
                    testset=testset,
                    lr = lr,
                    lam = lam)
    #define trainer
    trainer = set_trainer(checkpoint_path, accelerator, devices,max_epochs, 
                accumulate_grad_batches,min_delta, patience)
    #fit trainer
    trainer.fit(trainer_module)

def test(model, testset, checkpoint_path,lr, lam, accelerator, devices, max_epochs=300, accumulate_grad_batches = 1,min_delta = 0.001, patience = 50):
    """
    A training function

    Args:
    trainer: model trainer
    trainer_module: lightning model 

    Return
    metric: test metric
    deform: deformed image
    """
    ckpt = torch.load(checkpoint_path)['state_dict']
    #adjust checkpoint_path keys
    new_state_dict = {}
    for keys, values in ckpt.items():
        new_state_dict[keys[6:]] = values
    #load state_dict
    model.load_state_dict(new_state_dict)
    trainer = set_trainer(checkpoint_path, accelerator, devices,max_epochs, 
                accumulate_grad_batches,min_delta, patience)
    trainer_module = TrainerModule(model,
                    trainset=None,
                    evalset=None, 
                    testset=testset,
                    lr =None,
                    lam = lam)
    outs = trainer.predict(trainer_module, testset)
    out_loss = [out[1].unsqueeze(0) for out in outs]
    out_deformed = [out[0] for out in outs]
    loss = float(torch.cat(out_loss,0).mean())
    deformed = torch.cat(out_deformed, 0)
    return  loss, deformed
