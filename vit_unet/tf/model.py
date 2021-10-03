import tensorflow as tf
import numpy as np

# Auxiliary methods
def patch(X:tf.Tensor,
          patch_size:int,
          ):
    if len(X.shape)==5:
        X = tf.squeeze(X, axis=1)
    batch_size, h, w, channels = tf.shape(X)
    assert h%patch_size==0, f"Patch size must divide images height"
    assert w%patch_size==0, f"Patch size must divide images width"
    patches_tf = tf.image.extract_patches(X,
                                          sizes = [1, patch_size, patch_size, 1],
                                          strides = [1, patch_size, patch_size, 1],
                                          rates = [1,1,1,1],
                                          padding = 'VALID')
    patches_tf = tf.reshape(patches_tf,[batch_size,-1,patch_size,patch_size,3])
    return patches_tf

def unflatten(flattened, num_channels):
        # Alberto: Added to reconstruct from bs, n, projection_dim -> bs, n, c, h, w
        bs, n, p = flattened.shape
        unflattened = tf.reshape(flattened, (bs, n, int(np.sqrt(p//num_channels)), int(np.sqrt(p//num_channels)), num_channels))
        return unflattened

def unpatch(x, num_channels):
    if len(x.shape) < 5:
        batch_size, num_patches, h, w, ch = unflatten(x, num_channels).shape
    else:
        batch_size, num_patches, h, w, ch = x.shape
    assert ch==num_channels, f"Num. channels must agree"
    elem_per_axis = int(np.sqrt(num_patches))
    patches_middle = tf.stack([tf.concat([patch for patch in tf.reshape(x, shape=[batch_size,elem_per_axis,elem_per_axis,h,w,ch])[i]], axis = -3) for i in range(batch_size)], axis = 0)
    restored_images = tf.reshape(tf.stack([tf.concat([patch for patch in patches_middle[i]], axis = -2) for i in range(batch_size)], axis = 0), shape=[batch_size,1,h*elem_per_axis,w*elem_per_axis,ch])
    return restored_images

def downsampling(encoded_patches, num_channels):
    _, _, embeddings = encoded_patches.size()
    h, w, _ = int(np.sqrt(embeddings/num_channels)), int(np.sqrt(embeddings/num_channels)), num_channels
    original_image = unpatch(unflatten(encoded_patches, num_channels), num_channels)
    new_patches = patch(original_image, patch_size = h//2)
    new_patches_flattened = tf.reshape(new_patches, shape=[new_patches.shape[0], new_patches.shape[1], -1])
    return new_patches_flattened

def upsampling(encoded_patches, num_channels):
    _, _, embeddings = encoded_patches.size()
    h, w, _ = int(np.sqrt(embeddings/num_channels)), int(np.sqrt(embeddings/num_channels)), num_channels
    original_image = unpatch(unflatten(encoded_patches, num_channels), num_channels)
    new_patches = patch(original_image, patch_size = h*2)
    new_patches_flattened = tf.reshape(new_patches, shape=[new_patches.shape[0], new_patches.shape[1], -1])
    return new_patches_flattened


# Patch Encoder
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 depth:int,
                 num_patches:int,
                 patch_size:int,
                 num_channels:int,
                 preprocessing:str,
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
        self.positions = tf.range(start = 0,
                                  limit = self.num_patches_final,
                                  delta = 1,
                                  )

        # Layers
        if self.preprocessing == "conv":
            self.conv2d = tf.keras.layers.Conv2D(filters=self.num_channels, kernel_size=3, padding='same')
        self.position_embedding = tf.keras.layers.Embedding(input_dim=self.num_patches_final,
                                                            output_dim=self.num_channels*self.patch_size_final**2,
                                                            )

    def call(self, X):
        if self.preprocessing == 'conv':
            X = self.conv2d(X)
        elif self.preprocessing == 'fourier':
            X = tf.math.real(tf.signal.fft2d(X))
        patches = patch(X, self.patch_size_final)
        batch_size, _, _, _, _ = patches.shape
        flat_patches = tf.reshape(patches, shape = [batch_size, self.num_patches_final,-1])
        encoded = flat_patches + self.position_embedding(self.positions)
        encoded = unflatten(encoded, self.num_channels)
        encoded = unpatch(encoded, self.num_channels)
        encoded = tf.reshape(patch(encoded, patch_size = self.patch_size), shape = [batch_size, self.num_patches,-1])
        return encoded


# Encoder

## FeedForward
class FeedForward(tf.keras.layers.Layer):
    def __init__(self,
                 projection_dim:int,
                 hidden_dim:int,
                 dropout:float,
                 ):
        super().__init__()
        self.D1 = tf.keras.layers.Dense(hidden_dim)
        self.Drop1 = tf.keras.layers.Dropout(dropout)
        self.D2 = tf.keras.layers.Dense(projection_dim)
        self.Drop2 = tf.keras.layers.Dropout(dropout)

    def forward(self, x):
        x = self.D1(x)
        x = tf.keras.activations.gelu(x)
        x = self.Drop1(x)
        x = self.D2(x)
        x = self.Drop2(x)
        return x

## ReAttention
class ReAttention(tf.keras.layers.Layer):
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
            self.reatten_matrix = tf.keras.layers.Conv2D(self.num_heads, 1)
            self.var_norm = tf.keras.layers.BatchNormalization()
            self.qconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', use_bias=qkv_bias)
            self.kconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', use_bias=qkv_bias)
            self.vconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', use_bias=qkv_bias)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.qconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', use_bias=qkv_bias)
            self.kconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', use_bias=qkv_bias)
            self.vconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', use_bias=qkv_bias)
        
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)


    def create_queries(self, x, letter):
        if letter=='q':
            x = tf.stack([self.qconv2d(y) for y in unflatten(x, self.num_channels)], axis = 0)
        if letter == 'k':
            x = tf.stack([self.qconv2d(y) for y in unflatten(x, self.num_channels)], axis = 0)
        if letter == 'v':
            x = tf.stack([self.qconv2d(y) for y in unflatten(x, self.num_channels)], axis = 0)
        x = tf.reshape(x, shape=[x.shape[0], x.shape[1], -1])
        x = tf.reshape(x, shape = [x.shape[0], x.shape[1], self.num_heads, x.shape[2]//self.num_heads, 1])
        x = tf.transpose(x, perm = [4,0,2,1,3])
        return x[0]


    def forward(self, x, atten=None):
        B, N, C = x.shape
        q = create_queries(x, 'q')
        k = create_queries(x, 'k')
        v = create_queries(x, 'v')
        attn = (tf.linalg.matmul(q,tf.transpose(k, perm = [0,1,3,2]))) * self.scale
        attn = tf.keras.activations.softmax(attn, axis = -1)
        attn = self.attn_drop(attn)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        attn_next = attn
        x = tf.transpose(tf.linalg.matmul(attn, v), perm = [0,2,1,3]).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_next

## Transformer Encoder
class ReAttentionTransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_patches:int,
                 num_channels:int,
                 projection_dim:int,
                 hidden_dim:int,
                 num_heads:int,
                 attn_drop:int,
                 proj_drop:int,
                 linear_drop:float,
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
        self.ReAttn = ReAttention(self.projection_dim,
                                  num_channels = self.num_channels,
                                  num_heads = self.num_heads,
                                  attn_drop = self.attn_drop,
                                  proj_drop = self.proj_drop,
                                  )
        self.LN1 = tf.keras.layers.LayerNormalization()
        self.LN2 = tf.keras.layers.LayerNormalization()
        self.FeedForward = FeedForward(projection_dim = self.projection_dim,
                                       hidden_dim = self.hidden_dim,
                                       dropout = self.linear_drop,
                                       )
    def forward(self, encoded_patches):
        encoded_patch_attn, _ = self.ReAttn(encoded_patches)
        encoded_patches = encoded_patch_attn + encoded_patches
        encoded_patches = self.LN1(encoded_patches)
        encoded_patches = self.FeedForward(encoded_patches) + encoded_patches
        encoded_patches = self.LN2(encoded_patches)
        return encoded_patches


# Skip connection
class SkipConnection(tf.keras.layers.Layer):
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
        self.scale = head_dim ** -0.5
        self.reatten_matrix = tf.keras.layers.Conv2D(self.num_heads, 1)
        self.var_norm = tf.keras.layers.BatchNormalization()
        self.qconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', use_bias=qkv_bias)
        self.kconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', use_bias=qkv_bias)
        self.vconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', use_bias=qkv_bias)
        self.reatten_scale = self.scale if transform_scale else 1.0        
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)
        

    def create_queries(self, x, letter):
        if letter=='q':
            x = tf.stack([self.qconv2d(y) for y in unflatten(x, self.num_channels)], axis = 0)
        if letter == 'k':
            x = tf.stack([self.qconv2d(y) for y in unflatten(x, self.num_channels)], axis = 0)
        if letter == 'v':
            x = tf.stack([self.qconv2d(y) for y in unflatten(x, self.num_channels)], axis = 0)
        x = tf.reshape(x, shape=[x.shape[0], x.shape[1], -1])
        x = tf.reshape(x, shape = [x.shape[0], x.shape[1], self.num_heads, x.shape[2]//self.num_heads, 1])
        x = tf.transpose(x, perm = [4,0,2,1,3])
        return x[0]


    def forward(self, q,k,v):
        B, N, C = q.shape
        q = create_queries(q, 'q')
        k = create_queries(k, 'k')
        v = create_queries(v, 'v')
        attn = (tf.linalg.matmul(q,tf.transpose(k, perm = [0,1,3,2]))) * self.scale
        attn = tf.keras.activations.softmax(attn, axis = -1)
        attn = self.attn_drop(attn)
        attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        x = tf.transpose(tf.linalg.matmul(attn, v), perm = [0,2,1,3]).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Model
class ViT_UNet(tf.keras.layers.Layer):
    def __init__(self,
                 depth:int,
                 depth_te:int,
                 size_bottleneck:int,
                 preprocessing:str,
                 num_patches:int,
                 patch_size:int,
                 num_channels:int,
                 hidden_dim:int,
                 num_heads:int,
                 attn_drop:int,
                 proj_drop:int,
                 linear_drop:float,
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
            print('\tProjection size:',(num_channels*patch_size**2)//(4**i))
            print('\tHidden dim. size:',hidden_dim//(2**i))
        # Parameters
        self.depth = depth
        self.depth_te = depth_te
        self.size_bottleneck = size_bottleneck
        self.preprocessing = preprocessing
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.projection_dim = self.num_channels*(self.patch_size)**2
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.linear_drop = linear_drop
        # Layers
        self.PE = PatchEncoder(self.depth,self.num_patches,self.patch_size,self.num_channels,self.preprocessing,self.dtype)
        self.Encoders = []
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
        self.BottleNeck = []
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
        self.Decoders = []
        self.SkipConnections = []
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
            self.conv2d = tf.keras.layers.Conv2D(self.num_channels, kernel_size = 3, padding = 'same')
    
    def forward(self,
                X:tf.Tensor,
                ):
        # Previous validations
        batch_size, h, w, ch = X.size()

        # "Preprocessing"
        X_patch = self.PE(X)

        # Encoders
        encoder_skip = []
        print('Start encoding. Original shape:',X_patch.size())
        for i, enc in enumerate(self.Encoders):
            X_patch = enc(X_patch)
            if (i+1)%self.depth_te==0:
                encoder_skip.append(X_patch)
                X_patch = downsampling(X_patch, self.num_channels)
                print("\t Shape after level " + str((i+1)//self.depth_te) + " of encoding:",X_patch.size())
        # Bottleneck
        print('Start bottleneck')
        for i, bottle in enumerate(self.BottleNeck):
            X_patch = bottle(X_patch)
            print("\tShape after step " + str(i+1) + " of bottleneck:",X_patch.size())
        # Decoders
        print('Start decoding')
        for i, dec in enumerate(self.Decoders):
            print('\tStep',i+1)
            X_patch = dec(X_patch)
            if (i+1)%self.depth_te==0:
                X_patch = upsampling(X_patch, self.num_channels)
                print("\tShape after level " + str((i+1)//self.depth_te) + " of decoding:",X_patch.size())
                print('\tSkip connection')
                assert encoder_skip[self.depth-((i+1)//self.depth_te)].shape==X_patch.shape, f"enc and dec not same shape"
                X_patch = self.SkipConnections[(i+1)//self.depth_te-1](encoder_skip[self.depth-((i+1)//self.depth_te)], X_patch, X_patch)
        
        # Output
        X_restored = unpatch(unflatten(X_patch, self.num_channels), self.num_channels).reshape(batch_size, ch, h, w)
        print('Final processing is: ' + self.preprocessing)
        if self.preprocessing == 'conv':
            X_restored = self.conv2d(X_restored)
        elif self.preprocessing == 'fourier':
            X_restored = tf.math.real(tf.signal.fft2d(X_restored))

        return X_restored