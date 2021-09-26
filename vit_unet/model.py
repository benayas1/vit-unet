import numpy as np
import torch
import itertools


# Auxiliary functions to create & undo patches
def patch(
    X:torch.Tensor,
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

def unflatten(
    flattened:torch.Tensor,
    ):
    bs, n, p = flattened.size()
    unflattened = torch.reshape(flattened, (bs, n, 3, int(np.sqrt(p/3)), int(np.sqrt(p/3))))
    return unflattened

def unpatch(
    patches:torch.Tensor,
    ):
    if len(patches.size()) < 5:
        batch_size, num_patches, ch, h, w = unflatten(patches).size()
    else:
        batch_size, num_patches, ch, h, w = patches.size()
    elem_per_axis = int(np.sqrt(num_patches))
    patches_middle = torch.cat([patch for patch in patches.reshape(batch_size,elem_per_axis,elem_per_axis,ch,h,w)[0]], dim = -2)
    restored_image = torch.cat([patch for patch in patches_middle], dim = -1).reshape(batch_size,1,ch,h*elem_per_axis,w*elem_per_axis)
    return restored_image


# Auxiliary methods to downsampling & upsampling
def downsampling(encoded_patches):
    _, _, embeddings = encoded_patches.size()
    ch, h, w = 3, int(np.sqrt(embeddings/3)), int(np.sqrt(embeddings/3))
    original_image = unpatch(unflatten(encoded_patches))
    new_patches = patch(original_image, patch_size = h//2)
    new_patches_flattened = torch.nn.Flatten(start_dim = -3, end_dim = -1).forward(new_patches)
    return new_patches_flattened

def upsampling(encoded_patches):
    _, _, embeddings = encoded_patches.size()
    _, h, _ = 3, int(np.sqrt(embeddings/3)), int(np.sqrt(embeddings/3))
    original_image = unpatch(unflatten(encoded_patches))
    new_patches = patch(original_image, patch_size = h*2)
    new_patches_flattened = torch.flatten(new_patches, start_dim = -3, end_dim = -1)
    return new_patches_flattened


# Class PatchEncoder, to include initial and positional encoding
class PatchEncoder(torch.nn.Module):
    def __init__(self,
                 depth:int,
                 num_patches:int,
                 patch_size:int,
                 preprocessing:str,
                 dtype:torch.dtype,
                 ):
        super(PatchEncoder, self).__init__()
        # Parameters
        self.depth = depth
        self.patch_size = patch_size
        self.patch_size_final = self.patch_size//(2**self.depth)
        self.num_patches = num_patches
        self.num_patches_final = self.num_patches*(4**self.depth)
        assert preprocessing in ['conv', 'fourier', 'none'], f"Preprocessing can only be 'conv', 'fourier' or 'none'."
        self.preprocessing = preprocessing
        self.dtype = dtype
        self.positions = torch.arange(start = 0,
                         end = self.num_patches_final,
                         step = 1,
                         )

        # Layers
        if self.preprocessing == "conv":
            self.conv2d = torch.nn.Conv2d(3, 3, 3, padding = 'same')
        self.position_embedding = torch.nn.Embedding(num_embeddings=self.num_patches_final,
                                                     embedding_dim = 3*self.patch_size_final**2,
                                                     )

    def forward(self, X):
        if self.preprocessing == 'conv':
            X = self.conv2d(X)
        elif self.preprocessing == 'fourier':
            X = torch.fft.fft2(X).real
        patches = patch(X, self.patch_size_final)
        flat_patches = torch.flatten(patches, -3, -1)
        encoded = flat_patches + self.position_embedding(self.positions)
        encoded = torch.flatten(patch(unpatch(unflatten(encoded)), patch_size = self.patch_size), -3, -1)
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
        head_dim = dim // num_heads
        self.apply_transform = apply_transform
        
        self.scale = qk_scale or head_dim ** -0.5
        if apply_transform:
            self.reatten_matrix = torch.nn.Conv2d(self.num_heads,self.num_heads, 1, 1)
            self.var_norm = torch.nn.BatchNorm2d(self.num_heads)
            self.qconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)
            self.kconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)
            self.vconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.qconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)
            self.kconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)
            self.vconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)
        
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
    def forward(self, x, atten=None):
        B, N, C = x.shape
        q = torch.flatten(torch.stack([self.qconv2d(y) for y in unflatten(x)], dim = 0), -3,-1)
        k = torch.flatten(torch.stack([self.kconv2d(y) for y in unflatten(x)], dim = 0), -3,-1)
        v = torch.flatten(torch.stack([self.vconv2d(y) for y in unflatten(x)], dim = 0), -3,-1)
        qkv = torch.cat([q,k,v], dim = -1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
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
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.linear_drop = linear_drop
        self.dtype = dtype
        self.ReAttn = ReAttention(self.projection_dim,
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
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 transform_scale=False,
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.reatten_matrix = torch.nn.Conv2d(self.num_heads,self.num_heads, 1, 1)
        self.var_norm = torch.nn.BatchNorm2d(self.num_heads)
        self.qconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)
        self.kconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)
        self.vconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)

        self.reatten_scale = self.scale if transform_scale else 1.0
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        

    def forward(self, q, k, v):
        assert q.shape==k.shape
        assert k.shape==v.shape
        B, N, C = q.shape
        q = torch.flatten(torch.stack([self.qconv2d(y) for y in unflatten(q)], dim = 0), -3,-1)
        k = torch.flatten(torch.stack([self.kconv2d(y) for y in unflatten(k)], dim = 0), -3,-1)
        v = torch.flatten(torch.stack([self.vconv2d(y) for y in unflatten(v)], dim = 0), -3,-1)
        qkv = torch.cat([q,k,v], dim = -1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
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
                 num_patches:int,
                 patch_size:int,
                 projection_dim:int,
                 hidden_dim:int,
                 num_heads:int,
                 attn_drop:int,
                 proj_drop:int,
                 linear_drop:float,
                 dtype:torch.dtype,
                 ):
        super().__init__()
        # Testing
        assert patch_size%(2**(depth))==0, f"Depth must be adjusted, final patch size is incompatible."
        assert patch_size//(2**(depth))>=4, f"Depth must be adjusted, final patch size is too small (lower than 4)."
        print('Architecture information:')
        for i in range(depth+1):
            print('Level {}:'.format(i))
            print('\tPatch size:',patch_size//(2**i))
            print('\tNum. patches:',num_patches*(4**i))
            print('\tProjection size:',projection_dim//(4**i))
            print('\tHidden dim. size:',hidden_dim//(2**i))
        # Parameters
        self.depth = depth
        self.depth_te = depth_te
        self.size_bottleneck = size_bottleneck
        self.preprocessing = preprocessing
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.linear_drop = linear_drop
        self.dtype = dtype
        # Layers
        self.PE = PatchEncoder(self.depth,self.num_patches,self.patch_size,self.preprocessing,self.dtype)
        self.Encoders = torch.nn.ModuleList()
        for level in range(self.depth):
            exp_factor = 4**(level)
            exp_factor_hidden = 2**(level)
            for _ in range(depth_te):
                self.Encoders.append(
                    ReAttentionTransformerEncoder(self.num_patches*exp_factor,
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
                               num_heads = self.num_heads,
                               attn_drop = self.attn_drop,
                               proj_drop = self.proj_drop,
                               )
                )
        
        # Output
        if self.preprocessing == 'conv':
            self.conv2d = torch.nn.Conv2d(3,3,3,padding = 'same')
    
    def forward(self,
                X:torch.Tensor,
                ):
        # Previous validations
        batch_size, ch, h, w = X.size()

        # "Preprocessing"
        X_patch = self.PE(X)

        # Encoders
        encoder_skip = []
        #print('Start encoding. Original shape:',X_patch.size())
        for i, enc in enumerate(self.Encoders):
            X_patch = enc(X_patch)
            if (i+1)%self.depth_te==0:
                encoder_skip.append(X_patch)
                X_patch = downsampling(X_patch)
                #print("\t Shape after level " + str((i+1)//self.depth_te) + " of encoding:",X_patch.size())
        # Bottleneck
        #print('Start bottleneck')
        for i, bottle in enumerate(self.BottleNeck):
            X_patch = bottle(X_patch)
            #print("\t Shape after step " + str(i+1) + " of bottleneck:",X_patch.size())
        # Decoders
        #print('Start decoding')
        for i, dec in enumerate(self.Decoders):
            #print('\tStep',i+1)
            X_patch = dec(X_patch)
            if (i+1)%self.depth_te==0:
                X_patch = upsampling(X_patch)
                #print("\t Shape after level " + str((i+1)//self.depth_te) + " of decoding:",X_patch.size())
                #print('\tSkip connection')
                assert encoder_skip[self.depth-((i+1)//self.depth_te)].shape==X_patch.shape, f"enc and dec not same shape"
                X_patch = self.SkipConnections[(i-1)//self.depth_te](encoder_skip[self.depth-((i+1)//self.depth_te)], X_patch, X_patch)
        
        # Output
        X_restored = unpatch(unflatten(X_patch)).reshape(batch_size, ch, h, w)
        #print('Final processing is: ' + self.preprocessing)
        if self.preprocessing == 'conv':
            X_restored = self.conv2d(X_restored)
        elif self.preprocessing == 'fourier':
            X_restored = torch.fft.ifft2(X, norm='ortho').real

        return X_restored