import torch
import torch.nn as nn
from layers import conv_block, upsample_block, tranpconv_block

class EncoderBlock(nn.Module):
    """
        Creates a encoder block

        Args:
            inp_channels (int): number of input channels
            out_channels (int): number of output channels
            nb_conv_per_level (int): number of convolutions per layer
            act_function:(str): activate function (ReLU, ELU or LeakyReLU)
            
        Return:
            encoder convolutional block
    """

    def __init__(
            self,
            inp_channels,
            out_channels,
            nb_conv_per_level= 2,
            act_function = 'ReLU'
    ):

        super().__init__()
        
        self.nb_conv_per_level = nb_conv_per_level   
        #add one convolution to the moduleList that will contain all convs     
        self.encoder_block_list = nn.ModuleList([conv_block(inp_channels_conv=inp_channels, 
                            out_channels_conv = out_channels, act_function=act_function,
                            padding=1, 
                            kernel_size=3, )])
        
        #loop that will add convolutions if nb_conv_per_level!=1
        for _ in range(0,nb_conv_per_level-1):
            aditional_conv_block = conv_block(inp_channels_conv=out_channels, 
                            out_channels_conv = out_channels, act_function=act_function,
                            padding=1, 
                            kernel_size=3, )
            self.encoder_block_list.append(aditional_conv_block)

    def forward(self,x):
        
        for encoder_conv_block in self.encoder_block_list:
            x = encoder_conv_block(x)
        return x
       
class DecoderBlock(nn.Module):
    """
        Creates a decoder block

        Args:
            inp_channels (int): number of input channels
            out_channels (int): number of output channels
            layer (int): layer number [0,nb_conv_per_level+1]
            nb_conv_per_level (int): number of convolutions per layer
            scale_factor (int): upsampling scale factor
            act_function:(str): activate function (ReLU, ELU or LeakyReLU)
            unpooling: upsampling function ("transp_conv", "interpolation" or "no_pool")
            mode: interpolation algorithm used if unpooling == "interpolation" 
                    ('nearest', 'linear', 'bilinear', 'bicubic' or 'trilinear')
            
        Return:
            decoder convolutional block
    """
   
    def __init__(
            self,
            in_channels = 32,
            out_channels = 32,
            layer=0,
            nb_conv_per_level= 2,
            scale_factor = 2,
            act_function='ReLU',
            unpooling = "transp_conv", 
            mode = "nearest",
    ):
        super().__init__()
        
        self.unpooling = unpooling
        self.layer=layer
        if layer != 0:
            in_channels *= 2

        self.decoder_block_list = nn.ModuleList([
            conv_block(inp_channels_conv=in_channels, 
                    out_channels_conv = out_channels, 
                    act_function = act_function,
                    padding=1, 
                    kernel_size=3, )
        ])

        #loop that will add convolutions if nb_conv_per_level!=1
        for _ in range(0,nb_conv_per_level-1):        
            aditional_conv_block = conv_block(inp_channels_conv=out_channels, 
                                out_channels_conv = out_channels, 
                                act_function=act_function,
                                padding=1, 
                                kernel_size=3)
            self.decoder_block_list.append(aditional_conv_block)

        #define unpooling
        if self.unpooling=="transp_conv":
            self.unpool = tranpconv_block(out_channels, scale_factor, act_function)
        elif self.unpooling=="interpolation":
            self.unpool = upsample_block(scale_factor,mode, act_function)
        elif self.unpooling == "no_pool":
            self.unpool = nn.Identity()
        else:
            raise ValueError('unpooling must be "transp_conv", "interpolation" or "no_pool"')
        

    def forward(self, x, encode_feat):
                    
        for idx, decoder_conv_block in enumerate(self.decoder_block_list):
            if self.layer != 0 and idx==0:
                #concatenate to add the skip connections
                x = torch.concat([encode_feat, x], dim=1)
            x = decoder_conv_block(x)
        x = self.unpool(x)
        return x

class Decoder(nn.Module):
    """
        Creates the decoder path

        Args:
            out_channels (int): number of output channels
            nb_features (list) = number of features per layer, 
            nb_conv_per_level (int) = number of total convolutional blocks per layer, 
            scale_factor (int): upsampling scale factor
            act_function:(str): activate function (ReLU, ELU or LeakyReLU)
            unpooling: upsampling function ("transp_conv", "interpolation" or "no_pool")
            mode: interpolation algorithm used if unpooling == "interpolation" 
                    ('nearest', 'linear', 'bilinear', 'bicubic' or 'trilinear')
            last_act_function = activate function used on last layer ("softmax", "sigmoid" or "no_act_function")
            
        Return:
            output tensor
    """
    def __init__(
                self,
                out_channels=2, 
                nb_features=[32,64,128], 
                nb_conv_per_level=2, 
                scale_factor=2, 
                unpooling="transp_conv", 
                mode = "nearest",
                act_function="ReLU",
                last_act_function = "sigmoid"
                ):   
        super().__init__()

        #creates a moduleList that will contain decoder blocks
        self.decoder_blocks = nn.ModuleList([])
        nb_features = nb_features[::-1]
        for layer in range(len(nb_features)):
            #bottleneck (layer==0)
            if layer == 0:
                in_channels = nb_features[layer]
            else:
                in_channels = nb_features[layer-1]
            decoder_block = DecoderBlock(in_channels, nb_features[layer],layer, nb_conv_per_level,
            scale_factor, act_function, unpooling, mode)
            self.decoder_blocks.append(decoder_block)  

        #last layer (no unpooling)
        if last_act_function == "sigmoid":
            last_act_function = nn.Sigmoid()
        elif last_act_function == "softmax":
            last_act_function = nn.Softmax()
        elif last_act_function == "ReLU":
            last_act_function = nn.ReLU()
        elif last_act_function == "no_act_function":
            last_act_function = nn.Identity()
        else:
            raise ValueError('activation function must be sigmoid, softmax, ReLU or no_act_function')

        #last block with no pooling
        last_block = DecoderBlock(nb_features[-1], nb_features[-1], layer+1, 
                                        nb_conv_per_level, scale_factor, act_function = act_function,
                                        unpooling = "no_pool")

        self.decoder_blocks.append(last_block)                                        
        #last conv using kernel size 1x1                                             
        self.last_conv = conv_block(inp_channels_conv=nb_features[-1], 
                out_channels_conv = out_channels, 
                act_function = last_act_function,
                padding=0, 
                batch_norm=False,
                kernel_size=1)

    def forward(self,x, encoder_feat):
        encoder_feat = encoder_feat[::-1]       
        for idx, decoder_block in enumerate(self.decoder_blocks):
            #in the bottleneck there is no skip connection 
            #(hence, no concatenation with encoder features)
            if idx ==0:
                x = decoder_block(x, None)
            if idx !=0:
                x = decoder_block(x, encoder_feat[idx-1])

        x = self.last_conv(x)

        return x

class Encoder(nn.Module):
    """
        Creates the decoder path

        Args:
            inp_channels (int): number of input channels
            nb_features (list) = number of features per layer, 
            nb_conv_per_level (int) = number of total convolutional blocks per layer, 
            scale_factor (int): upsampling scale factor
            pooling (bool): True if use MaxPool2D
            act_function:(str): activate function (ReLU, ELU or LeakyReLU)
            
        Return:
            x (tensor): encoder output tensor 
            encode_feat (list): encoder features
    """
    def __init__(
            self,
            inp_channels=2,
            nb_features=[32,64,128],
            nb_conv_per_level=5,
            scale_factor = 2, 
            act_function = "ReLU"
    ):  
        super().__init__()
        channels = [inp_channels]+ nb_features
        #creates a moduleList that will contain encoder blocks
        self.encoder_list = nn.ModuleList([])

        for layers in range(len(nb_features)):
            inp_channels = channels[layers]
            out_channels = channels[layers+1]
            encoder_block = EncoderBlock(inp_channels,out_channels, nb_conv_per_level, act_function)
            self.encoder_list.append(encoder_block)

        self.pool = nn.MaxPool2d(scale_factor)        

    def forward(self,x):
        encode_feat = []
        for encoder_block in self.encoder_list:
            x = encoder_block(x)
            encode_feat.append(x)
            x = self.pool(x) 
        return x, encode_feat