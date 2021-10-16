import tensorflow as tf
import numpy as np
from typing import List
from vit_unet.tf.functions import *


# Models
## HViT
class HViT_UNet(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int=128,
                 patch_size:List[int]=[8,16,32],
                 projection_dim:int=None,
                 num_channels:int=3,
                 num_heads:int=8,
                 transformer_layers:List[int]=[4,4],
                 size_bottleneck:int=4,
                 hidden_unit_factor:float=2.,
                 drop_attn:float=.2,
                 drop_proj:float=.2,
                 drop_linear:float=.4,
                 resampling_type:str='standard',
                 original_attn:bool=True,
                 ):
        super(HViT_UNet, self).__init__()
        #Validations
        assert resampling_type in ['max', 'avg', 'standard'], f"Resampling type must be either 'max', 'avg' or 'standard'."
        assert all([img_size//patch==img_size/patch for patch in patch_size]), f"Patch sizes must divide image size."
        assert all([patch_size[i]<patch_size[i+1] for i in range(len(patch_size)-1)]), f"Patch sizes must be a strictly increasing sequence."
        assert (resampling_type in ["max","avg"] and projection_dim is not None) or (resampling_type=="standard" and projection_dim is None),f"\
            If resampling_type is in ['max', 'avg'], projection_dim must be specified. If resampling_type is 'standard', projection_dim is auto\
            matically computed."
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_size_rev = self.patch_size[-1::-1]
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.size_bottleneck = size_bottleneck
        self.drop_attn = drop_attn
        self.drop_proj = drop_proj
        self.drop_linear = drop_linear
        self.resampling_type = resampling_type
        self.original_attn = original_attn
        self.num_patches = [(self.img_size//patch)**2 for patch in self.patch_size]
        if projection_dim is not None:
            self.projection_dim = [projection_dim for _ in self.patch_size]
        else:
            self.projection_dim = [self.num_channels*patch**2 for patch in self.patch_size]
        self.hidden_units = [int(hidden_unit_factor*proj) for proj in self.projection_dim]
        # Layers
        ##Positional Encoding
        self.PE = PatchEncoder(self.img_size, self.patch_size[0], self.num_channels, self.projection_dim[0])
        ##Encoder
        self.Encoder = []
        self.Encoder_RS = []        
        if self.original_attn:
            for i in range(len(self.patch_size)-1):
                self.Encoder.append(
                                    AttentionTransformerEncoder(self.img_size,
                                            self.patch_size[i],
                                            self.num_channels,
                                            self.num_heads,
                                            self.transformer_layers[i],
                                            self.projection_dim[i], 
                                            self.hidden_units[i],
                                            self.drop_attn,
                                            self.drop_proj,
                                            )
                                    )
                self.Encoder_RS.append(
                                        Resampling(self.img_size,
                                                   self.patch_size[i:i+2],
                                                   self.num_channels,
                                                   self.projection_dim[i+1],
                                                   self.resampling_type,
                                                   )
                                        )
        else:
            for i in range(len(self.patch_size)-1):
                self.Encoder.append(
                                    ReAttentionTransformerEncoder(self.img_size,
                                                                  self.patch_size[i],
                                                                  self.num_channels,
                                                                  self.num_heads,
                                                                  self.transformer_layers[i],
                                                                  self.projection_dim[i],
                                                                  self.hidden_units[i],
                                                                  self.drop_attn,
                                                                  self.drop_proj,
                                                                  )
                                    )
                self.Encoder_RS.append(
                                        Resampling(self.img_size,
                                                   self.patch_size[i:i+2],
                                                   self.num_channels,
                                                   self.projection_dim[i+1],
                                                   self.resampling_type,
                                                   )
                                        )
        ##BottleNeck
        if self.original_attn:
            self.BottleNeck = tf.keras.Sequential([
                                                    AttentionTransformerEncoder(self.img_size,
                                                                                self.patch_size[i],
                                                                                self.num_channels,
                                                                                self.num_heads,
                                                                                self.size_bottleneck,
                                                                                self.projection_dim[-1], 
                                                                                self.hidden_units[-1],
                                                                                self.drop_attn,
                                                                                self.drop_proj,
                                                                                )
                                                    ])
        else:
            self.BottleNeck = tf.keras.Sequential([
                                                    ReAttentionTransformerEncoder(self.img_size,
                                                                  self.patch_size[i],
                                                                  self.num_channels,
                                                                  self.num_heads,
                                                                  self.size_bottleneck,
                                                                  self.projection_dim[-1],
                                                                  self.hidden_units[-1],
                                                                  self.drop_attn,
                                                                  self.drop_proj,
                                                                  )
                                                    ])
        ##Decoder
        self.Decoder = []
        self.Decoder_RS = []         
        if self.original_attn:
            for i in range(len(self.patch_size)-1):
                self.Decoder_RS.append(
                                        Resampling(self.img_size,
                                                   self.patch_size_rev[i:i+2],
                                                   self.num_channels,
                                                   self.projection_dim[len(patch_size)-(i+2)],
                                                   self.resampling_type,
                                                   )
                               )
                self.Decoder.append(
                                    AttentionTransformerEncoder(self.img_size,
                                                                self.patch_size_rev[i+1],
                                                                self.num_channels,
                                                                self.num_heads,
                                                                self.transformer_layers[len(patch_size)-(i+2)],
                                                                self.projection_dim[len(patch_size)-(i+2)], 
                                                                self.hidden_units[len(patch_size)-(i+2)],
                                                                self.drop_attn,
                                                                self.drop_proj,
                                                                )
                                    )
                
        else:
            for i in range(len(self.patch_size)-1):
                self.Decoder_RS.append(
                                        Resampling(self.img_size,
                                                   self.patch_size_rev[i:i+2],
                                                   self.num_channels,
                                                   self.projection_dim[len(patch_size)-(i+2)],
                                                   self.resampling_type,
                                                   )
                               )
                self.Decoder.append(
                                    ReAttentionTransformerEncoder(self.img_size,
                                                                  self.patch_size_rev[i+1],
                                                                  self.num_channels,
                                                                  self.num_heads,
                                                                  self.transformer_layers[len(patch_size)-(i+2)],
                                                                  self.projection_dim[len(patch_size)-(i+2)], 
                                                                  self.hidden_units[len(patch_size)-(i+2)],
                                                                  self.drop_attn,
                                                                  self.drop_proj,
                                                                  )
                                    )
        ## Skip connections
        self.SkipConnections = []
        for i in range(len(self.patch_size)-1):
            self.SkipConnections.append(
                                        SkipConnection(self.img_size,
                                                       self.patch_size_rev[i+1],
                                                       self.num_channels,
                                                       self.projection_dim[len(patch_size)-(i+2)],
                                                       self.num_heads,
                                                       self.drop_attn,
                                                       )
                                        )

    def call(self, X:tf.Tensor):
        # Patch
        encoded = self.PE(X)
        # Encoder
        encoded_list = []
        for i in range(len(self.patch_size)-1):
            encoded = self.Encoder[i](encoded)
            encoded_list.append(encoded)
            encoded = self.Encoder_RS[i](encoded)
        # BottleNeck
        encoded = self.BottleNeck(encoded)
        # Decoder
        encoded_list = encoded_list[-1::-1]
        for i in range(len(self.patch_size)-1):
            encoded = self.Decoder_RS[i](encoded)
            encoded = self.Decoder[i](encoded)
            encoded = self.SkipConnections[i](encoded_list[i], encoded)
        # Return original image
        Y = X + tf.squeeze(unpatch(unflatten(encoded, self.num_channels), self.num_channels), axis = 1)
        return Y
