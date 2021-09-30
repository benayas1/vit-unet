import numpy as np
import torch
import torchvision
import itertools

# Auxiliary functions to create & undo patches
def patch(X:torch.Tensor,
          patch_size:int,
          ):
    if len(X.size())==5:
        X = torch.squeeze(X, dim=1)
    h, w = X.shape[-2], X.shape[-1]
    assert h%patch_size==0, f"Patch size must divide images height"
    assert w%patch_size==0, f"Patch size must divide images width"
    patches = X.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patch_list = []
    for row, col in itertools.product(range(h//patch_size), range(w//patch_size)):
        patch_list.append(patches[:,:,row,col,:,:])
    patches = torch.stack(patch_list, dim = 1)
    return patches

def unflatten(flattened, num_channels):
        # Alberto: Added to reconstruct from bs, n, projection_dim -> bs, n, c, h, w
        bs, n, p = flattened.size()
        unflattened = torch.reshape(flattened, (bs, n, num_channels, int(np.sqrt(p//num_channels)), int(np.sqrt(p//num_channels))))
        return unflattened

def unpatch(x, num_channels):
    if len(x.size()) < 5:
        batch_size, num_patches, ch, h, w = unflatten(x, num_channels).size()
    else:
        batch_size, num_patches, ch, h, w = x.size()
    assert ch==num_channels, f"Num. channels must agree"
    elem_per_axis = int(np.sqrt(num_patches))
    patches_middle = torch.stack([torch.cat([patch for patch in x.reshape(batch_size,elem_per_axis,elem_per_axis,ch,h,w)[i]], dim = -2) for i in range(batch_size)], dim = 0)
    restored_images = torch.stack([torch.cat([patch for patch in patches_middle[i]], dim = -1) for i in range(batch_size)], dim = 0).reshape(batch_size,1,ch,h*elem_per_axis,w*elem_per_axis)
    return restored_images


# Auxiliary methods to downsampling & upsampling
def downsampling(encoded_patches, num_channels):
    _, _, embeddings = encoded_patches.size()
    ch, h, w = num_channels, int(np.sqrt(embeddings/num_channels)), int(np.sqrt(embeddings/num_channels))
    original_image = unpatch(unflatten(encoded_patches, num_channels), num_channels)
    new_patches = patch(original_image, patch_size = h//2)
    new_patches_flattened = torch.nn.Flatten(start_dim = -3, end_dim = -1).forward(new_patches)
    return new_patches_flattened

def upsampling(encoded_patches, num_channels):
    _, _, embeddings = encoded_patches.size()
    ch, h, w = num_channels, int(np.sqrt(embeddings/num_channels)), int(np.sqrt(embeddings/num_channels))
    original_image = unpatch(unflatten(encoded_patches, num_channels), num_channels)
    new_patches = patch(original_image, patch_size = h*2)
    new_patches_flattened = torch.nn.Flatten(start_dim = -3, end_dim = -1).forward(new_patches)
    return new_patches_flattened


# Class PatchEncoder, to include initial and positional encoding
class PatchEncoder(torch.nn.Module):
    def __init__(self,
                 depth:int,
                 num_patches:int,
                 patch_size:int,
                 num_channels:int,
                 preprocessing:str,
                 dtype:torch.dtype,
                 ):
        super(PatchEncoder, self).__init__()
        # Parameters
        self.depth = depth
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.patch_size_final = self.patch_size//(2**self.depth)
        self.num_patches = num_patches
        self.num_patches_final = self.num_patches*(4**self.depth)
        assert preprocessing in ['conv', 'fourier', 'none'], f"Preprocessing can only be 'conv', 'fourier' or 'none'."
        self.preprocessing = preprocessing
        self.dtype = dtype
        self.positions = torch.arange(start = 0,
                         end = self.num_patches_final,
                         step = 1,
                         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                         )

        # Layers
        if self.preprocessing == "conv":
            self.conv2d = torch.nn.Conv2d(self.num_channels, self.num_channels, 3, padding = 'same')
        self.position_embedding = torch.nn.Embedding(num_embeddings=self.num_patches_final,
                                                     embedding_dim = self.num_channels*self.patch_size_final**2,
                                                     )

    def forward(self, X):
        if self.preprocessing == 'conv':
            X = self.conv2d(X)
        elif self.preprocessing == 'fourier':
            X = torch.fft.fft2(X).real
        patches = patch(X, self.patch_size_final)
        flat_patches = torch.flatten(patches, -3, -1)
        encoded = flat_patches + self.position_embedding(self.positions)
        encoded = unflatten(encoded, self.num_channels)
        encoded = unpatch(encoded, self.num_channels)
        encoded = torch.flatten(patch(encoded, patch_size = self.patch_size), -3, -1)
        return encoded


# AutoEncoder implementation
class FeedForward(torch.nn.Module):
    def __init__(self,
                 projection_dim:int,
                 hidden_dim:int,
                 dropout:float,
                 dtype:torch.dtype,
                 ):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(projection_dim, hidden_dim, dtype = dtype),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, projection_dim, dtype = dtype),
            torch.nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class ReAttention(torch.nn.Module):
    def __init__(self,
                 dim,
                 num_channels=3,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 expansion_ratio = 3,
                 apply_transform=True,
                 transform_scale=False,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels
        head_dim = dim // num_heads
        self.apply_transform = apply_transform
        self.scale = qk_scale or head_dim ** -0.5
        if apply_transform:
            self.reatten_matrix = torch.nn.Conv2d(self.num_heads,self.num_heads, 1, 1)
            self.var_norm = torch.nn.BatchNorm2d(self.num_heads)
            self.qconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)
            self.kconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)
            self.vconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.qconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)
            self.kconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)
            self.vconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)
        
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
    def forward(self, x, atten=None):
        B, N, C = x.shape
        q = torch.flatten(torch.stack([self.qconv2d(y) for y in unflatten(x, self.num_channels)], dim = 0), -3,-1).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        k = torch.flatten(torch.stack([self.kconv2d(y) for y in unflatten(x, self.num_channels)], dim = 0), -3,-1).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        v = torch.flatten(torch.stack([self.vconv2d(y) for y in unflatten(x, self.num_channels)], dim = 0), -3,-1).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        attn = (torch.matmul(q, k.transpose(-2, -1))) * self.scale
        attn = torch.nn.functional.softmax(attn, dim = -1)
        attn = self.attn_drop(attn)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        attn_next = attn
        x = (torch.matmul(attn, v)).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_next


class ReAttentionTransformerEncoder(torch.nn.Module):
    def __init__(self,
                 num_patches:int,
                 num_channels:int,
                 projection_dim:int,
                 hidden_dim:int,
                 num_heads:int,
                 attn_drop:int,
                 proj_drop:int,
                 linear_drop:float,
                 dtype:torch.dtype,
                 ):
        super().__init__()
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.linear_drop = linear_drop
        self.dtype = dtype
        self.ReAttn = ReAttention(self.projection_dim,
                                  num_channels = self.num_channels,
                                  num_heads = self.num_heads,
                                  attn_drop = self.attn_drop,
                                  proj_drop = self.proj_drop,
                                  )
        self.LN = torch.nn.LayerNorm(normalized_shape = (self.num_patches, self.projection_dim),
                                     dtype = self.dtype,
                                     )
        self.FeedForward = FeedForward(projection_dim = self.projection_dim,
                                       hidden_dim = self.hidden_dim,
                                       dropout = self.linear_drop,
                                       dtype = self.dtype,
                                       )
    def forward(self, encoded_patches):
        encoded_patch_attn, _ = self.ReAttn(encoded_patches)
        encoded_patches += encoded_patch_attn
        encoded_patches = self.LN(encoded_patches)
        encoded_patches += self.FeedForward(encoded_patches)
        encoded_patches = self.LN(encoded_patches)
        return encoded_patches


# Skip connections
class SkipConnection(torch.nn.Module):
    """
    It is observed that similarity along same batch of data is extremely large. 
    Thus can reduce the bs dimension when calculating the attention map.
    """
    def __init__(self,
                 dim,
                 num_channels=3,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 transform_scale=False,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels
        head_dim = dim // num_heads
        
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5
        self.reatten_matrix = torch.nn.Conv2d(self.num_heads,self.num_heads, 1, 1)
        self.var_norm = torch.nn.BatchNorm2d(self.num_heads)
        self.qconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)
        self.kconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)
        self.vconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)

        self.reatten_scale = self.scale if transform_scale else 1.0
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        

    def forward(self, q, k, v):
        assert q.shape==k.shape
        assert k.shape==v.shape
        B, N, C = q.shape
        q = torch.flatten(torch.stack([self.qconv2d(y) for y in unflatten(q, self.num_channels)], dim = 0), -3,-1).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        k = torch.flatten(torch.stack([self.kconv2d(y) for y in unflatten(k, self.num_channels)], dim = 0), -3,-1).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        v = torch.flatten(torch.stack([self.vconv2d(y) for y in unflatten(v, self.num_channels)], dim = 0), -3,-1).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        attn = (torch.matmul(q,k.transpose(-2, -1))) * self.scale
        attn = torch.nn.functional.softmax(attn, dim = -1)
        attn = self.attn_drop(attn)
        attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Model architecture
class ViT_UNet(torch.nn.Module):
    def __init__(self,
                 depth:int,
                 depth_te:int,
                 size_bottleneck:int,
                 preprocessing:str,
                 im_size:int,
                 patch_size:int,
                 num_channels:int,
                 hidden_dim:int,
                 num_heads:int,
                 attn_drop:float,
                 proj_drop:float,
                 linear_drop:float,
                 dtype:torch.dtype,
                 ):
        super().__init__()
        # Testing
        assert patch_size%(2**(depth))==0, f"Depth must be adjusted, final patch size is incompatible."
        assert patch_size//(2**(depth))>=4, f"Depth must be adjusted, final patch size is too small (lower than 4)."
        assert im_size%patch_size==0, f"Patch size is not compatible with image size."
        # Parameters
        self.depth = depth
        self.depth_te = depth_te
        self.size_bottleneck = size_bottleneck
        self.preprocessing = preprocessing
        self.im_size = im_size
        self.num_patches = (self.im_size//self.patch_size)**2
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.projection_dim = self.num_channels*(self.patch_size)**2
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.linear_drop = linear_drop
        self.dtype = dtype
        # Info
        print('Architecture information:')
        for i in range(depth+1):
            print('Level {}:'.format(i))
            print('\tPatch size:',self.patch_size//(2**i))
            print('\tNum. patches:',self.num_patches*(4**i))
            print('\tProjection size:',(self.num_channels*self.patch_size**2)//(4**i))
            print('\tHidden dim. size:',self.hidden_dim//(2**i))   
        # Layers
        self.PE = PatchEncoder(self.depth,self.num_patches,self.patch_size,self.num_channels,self.preprocessing,self.dtype)
        self.Encoders = torch.nn.ModuleList()
        for level in range(self.depth):
            exp_factor = 4**(level)
            exp_factor_hidden = 2**(level)
            for _ in range(depth_te):
                self.Encoders.append(
                    ReAttentionTransformerEncoder(self.num_patches*exp_factor,
                                                  self.num_channels,
                                                  self.projection_dim//exp_factor,
                                                  self.hidden_dim//exp_factor_hidden,
                                                  self.num_heads,
                                                  self.attn_drop,
                                                  self.proj_drop,
                                                  self.linear_drop,
                                                  self.dtype,
                                                  )
                )
        self.BottleNeck = torch.nn.ModuleList()
        for _ in range(self.size_bottleneck):
            exp_factor = 4**(self.depth)
            exp_factor_hidden = 2**(self.depth)
            self.BottleNeck.append(
                ReAttentionTransformerEncoder(self.num_patches*exp_factor,
                                              self.num_channels,
                                              self.projection_dim//exp_factor,
                                              self.hidden_dim//exp_factor_hidden,
                                              self.num_heads,
                                              self.attn_drop,
                                              self.proj_drop,
                                              self.linear_drop,
                                              self.dtype,
                                              )
            )
        self.Decoders = torch.nn.ModuleList()
        self.SkipConnections = torch.nn.ModuleList()
        for level in range(self.depth):
            exp_factor = 4**(self.depth-level)
            exp_factor_skip = 4**(self.depth-level-1)
            exp_factor_hidden = 2**(self.depth-level)
            for _ in range(depth_te):
                self.Decoders.append(
                    ReAttentionTransformerEncoder(self.num_patches*exp_factor,
                                                  self.num_channels,
                                                  self.projection_dim//exp_factor,
                                                  self.hidden_dim//exp_factor_hidden,
                                                  self.num_heads,
                                                  self.attn_drop,
                                                  self.proj_drop,
                                                  self.linear_drop,
                                                  self.dtype,
                                                  )
                )
            self.SkipConnections.append(
                SkipConnection(dim = self.projection_dim//exp_factor_skip,
                               num_channels = self.num_channels,
                               num_heads = self.num_heads,
                               attn_drop = self.attn_drop,
                               proj_drop = self.proj_drop,
                               )
                )
        
        # Output
        if self.preprocessing == 'conv':
            self.conv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same')
    
    def forward(self,
                X:torch.Tensor,
                ):
        # Previous validations
        X = torchvision.transforms.Resize(self.im_size)(X)
        batch_size, _, _, _ = X.size()

        # "Preprocessing"
        X_patch = self.PE(X)

        # Encoders
        encoder_skip = []
        #print('Start encoding. Original shape:',X_patch.size())
        for i, enc in enumerate(self.Encoders):
            X_patch = enc(X_patch)
            if (i+1)%self.depth_te==0:
                encoder_skip.append(X_patch)
                X_patch = downsampling(X_patch, self.num_channels)
                #print("\t Shape after level " + str((i+1)//self.depth_te) + " of encoding:",X_patch.size())
        # Bottleneck
        #print('Start bottleneck')
        for i, bottle in enumerate(self.BottleNeck):
            X_patch = bottle(X_patch)
            #print("\tShape after step " + str(i+1) + " of bottleneck:",X_patch.size())
        # Decoders
        #print('Start decoding')
        for i, dec in enumerate(self.Decoders):
            #print('\tStep',i+1)
            X_patch = dec(X_patch)
            if (i+1)%self.depth_te==0:
                X_patch = upsampling(X_patch, self.num_channels)
                #print("\tShape after level " + str((i+1)//self.depth_te) + " of decoding:",X_patch.size())
                #print('\tSkip connection')
                assert encoder_skip[self.depth-((i+1)//self.depth_te)].shape==X_patch.shape, f"enc and dec not same shape"
                X_patch = self.SkipConnections[(i+1)//self.depth_te-1](encoder_skip[self.depth-((i+1)//self.depth_te)], X_patch, X_patch)
        
        # Output
        X_restored = unpatch(unflatten(X_patch, self.num_channels), self.num_channels).reshape(batch_size, self.num_channels, self.im_size, self.im_size)
        #print('Final processing is: ' + self.preprocessing)
        if self.preprocessing == 'conv':
            X_restored = self.conv2d(X_restored)
        elif self.preprocessing == 'fourier':
            X_restored = torch.fft.ifft2(X, norm='ortho').real

        return X_restored


def get_vit_unet(model_string: str):
    if model_string.lower() == 'lite':
        return ViT_UNet(depth = 2,
                        depth_te = 1,
                        size_bottleneck = 2,
                        preprocessing = 'conv',
                        im_size = 224,
                        patch_size = 16,
                        num_channels = 3,
                        hidden_dim = 64,
                        num_heads = 4,
                        attn_drop = 0.2,
                        proj_drop = 0.2,
                        linear_drop = 0,
                        dtype = torch.float32,
                        )

    if model_string.lower() == 'base':
        return ViT_UNet(depth = 2,
                        depth_te = 2,
                        size_bottleneck = 2,
                        preprocessing = 'conv',
                        im_size = 224,
                        patch_size = 32,
                        num_channels = 3,
                        hidden_dim = 128,
                        num_heads = 8,
                        attn_drop = .2,
                        proj_drop = .2,
                        linear_drop = 0,
                        dtype = torch.float32,
                        )

    if model_string.lower() == 'large':
        return ViT_UNet(depth = 2,
                        depth_te = 4,
                        size_bottleneck = 4,
                        preprocessing = 'conv',
                        im_size = 224,
                        patch_size = 32,
                        num_channels = 3,
                        hidden_dim = 128,
                        num_heads = 8,
                        attn_drop = .2,
                        proj_drop = .2,
                        linear_drop = 0,
                        dtype = torch.float32,
                        )
    raise ValueError(f'Model string {model_string} not valid')