import torch.nn as nn   

def conv_block(inp_channels_conv, out_channels_conv, act_function,padding, kernel_size=3, batch_norm=True):
    """
        Create a 2-D convolutional block.

        Args:
            inp_channels_conv (int): number of input channels
            out_channels_conv (int): number of output channels
            act_function(str): active function (ReLU, ELU or LeakyReLU)
            padding (int): Additional size added to one side of each dimension
            kernel_size(int): kernel size
            
        Return:
            convolutional block
    """        
    if batch_norm:
        batch_norm = nn.BatchNorm2d(out_channels_conv)
    else:
        batch_norm = nn.Identity()

    layers = [
        nn.Conv2d(in_channels=inp_channels_conv,
                    out_channels=out_channels_conv,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                    ),
        batch_norm,
        act_function,
    ]

    return nn.Sequential(*layers)

def upsample_block(scale_factor, mode, act_function):
        """
        Create a upsample block using interpolation.

        Args:
            scale_factor (int): scale factor
            mode (str): upsampling algorithm ('nearest', 'linear', 'bilinear', 'bicubic' or 'trilinear')
            act_function(str): active function (ReLU, ELU or LeakyReLU)
            
        Return:
            upsample block
        """        
        layers = [
            nn.Upsample(size=None,
                        scale_factor=scale_factor,
                        mode=mode,
                        ),
            act_function,
        ]
        return nn.Sequential(*layers)

def tranpconv_block(inp_channels_conv, scale_factor,act_function):
    """
        Create a upsample block using transposed convolution.

        Args:
            inp_channels_conv (int): number of input channels
            scale_factor (int): scale factor
            act_function(str): active function (ReLU, ELU or LeakyReLU)
            
        Return:
            upsample block
    """     
    layers = [
        nn.ConvTranspose2d(in_channels=inp_channels_conv,
                    out_channels=inp_channels_conv,
                    kernel_size=scale_factor,
                    stride=scale_factor
                    ),           
                    
        act_function,
        ]
    return nn.Sequential(*layers)
