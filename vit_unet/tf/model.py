import tensorflow as tf
import numpy as np
from typing import List

# Auxiliary methods
def patches(X:tf.Tensor,
          patch_size:int,
          ):

    def patches_2d(X:tf.Tensor):
        h, w = X.shape.as_list()
        X_middle = tf.stack(tf.split(X,h//patch_size, axis = 0), axis = 0)
        X_final = tf.map_fn(fn=lambda y: tf.stack(tf.split(y,w//patch_size, axis = 1), axis = 0), elems = X_middle)
        X_final = tf.reshape(X_final, shape=[-1,patch_size,patch_size])
        return X_final

    if len(X.shape)==5:
        X = tf.squeeze(X, axis=1)
    _, h, w, _ = X.shape.as_list()
    assert h%patch_size==0, f"Patch size must divide images height"
    assert w%patch_size==0, f"Patch size must divide images width"
    X = tf.transpose(X, perm=[0,3,1,2])
    patches_tf = tf.map_fn(fn=lambda y: tf.map_fn(fn = lambda z: patches_2d(z), elems = y),
                           elems = X,
                           )
    patches_tf = tf.transpose(patches_tf, [0,2,3,4,1])
    return patches_tf

def unflatten(flattened, num_channels):
    if len(flattened.shape)==2:
        n, p = flattened.shape.as_list()
    else:
        _, n, p = flattened.shape.as_list()
    unflattened = tf.reshape(flattened, (-1, n, int(np.sqrt(p//num_channels)), int(np.sqrt(p//num_channels)), num_channels))
    return unflattened

def unpatch(x, num_channels):
    if len(x.shape) < 5:
        _, num_patches, h, w, ch = unflatten(x, num_channels).shape.as_list()
    else:
        _, num_patches, h, w, ch = x.shape.as_list()
    assert ch==num_channels, f"Num. channels must agree"
    elem_per_axis = int(np.sqrt(num_patches))
    x = tf.stack(tf.split(x, elem_per_axis, axis = 1), axis = 1)
    patches_middle = tf.concat(tf.unstack(x, axis = 2), axis = -2)
    restored_images = tf.reshape(tf.concat(tf.unstack(patches_middle, axis = 1), axis = -3), shape=[-1,1,h*elem_per_axis,w*elem_per_axis,ch])
    return restored_images

def resampling(encoded_patches, num_patches:List[int]=[64,256], projection_dim:List[int]=[196,64], num_channels:int=1):
    new_patch_size = int(np.sqrt(projection_dim[1]))
    original_image = unpatch(unflatten(encoded_patches, num_channels), num_channels)
    new_patches = patches(tf.squeeze(original_image, axis=1), new_patch_size)
    new_patches_flattened = tf.reshape(new_patches, shape=[-1, num_patches[1], projection_dim[1]])
    return new_patches_flattened

# Layers
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int=128,
                 patch_size:int=16,
                 num_channels:int=1,
                 ):
        super(PatchEncoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (self.img_size//self.patch_size)**2
        self.projection_dim = self.num_channels*self.patch_size**2
        self.projection = tf.keras.layers.Dense(units=self.projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )

    def call(self, X:tf.Tensor):
        X = patches(X, self.patch_size)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(X) + self.position_embedding(positions)
        return encoded

class DeepPatchEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int=128,
                 patch_size:List[int]=[16,8],
                 num_channels:int=1,
                 dropout:float=.2,
                 ):
        super(DeepPatchEncoder, self).__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = [(self.img_size//patch)**2 for patch in self.patch_size]
        self.projection_dim = [self.num_channels*patch**2 for patch in self.patch_size]
        # Layers
        self.dense = tf.keras.layers.Dense(self.projection_dim[0])
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_patches[0], output_dim=self.projection_dim[0],
        )
        if self.patch_size[0]>self.patch_size[1]:
            self.position_embedding_2 = tf.keras.Sequential([
                      tf.keras.layers.Conv2D(self.num_patches[1], kernel_size = (3,3), strides = (2,2), padding='same'),
                      tf.keras.layers.BatchNormalization(),
                      tf.keras.layers.Dropout(dropout),
                      tf.keras.layers.LeakyReLU(),
            ])
        else:
            self.position_embedding_2 = tf.keras.Sequential([
                      tf.keras.layers.Conv2DTranspose(self.num_patches[1], kernel_size = (3,3), strides = (2,2), padding='same'),
                      tf.keras.layers.BatchNormalization(),
                      tf.keras.layers.Dropout(dropout),
                      tf.keras.layers.LeakyReLU(),
            ])

    def call(self, X:tf.Tensor):
        # Flat patches
        patch = patches(X,self.patch_size[1])
        flat = tf.reshape(patch, [-1, self.num_patches[1], self.projection_dim[1]])
        # Embedding 1
        positions = tf.range(start=0, limit=self.num_patches[0], delta=1)
        pos_enc_1 = self.position_embedding(positions)
        # Embedding 2
        pos_enc_2 = unflatten(pos_enc_1, self.num_channels)
        pos_enc_2 = tf.transpose(pos_enc_2, [0,4,2,3,1])
        pos_enc_2 = tf.map_fn(lambda y: self.position_embedding_2(y), elems = pos_enc_2)
        pos_enc_2 = tf.transpose(pos_enc_2, [0,4,2,3,1])
        pos_enc_2 = tf.reshape(pos_enc_2, [-1,self.num_patches[-1], self.projection_dim[-1]])
        # Encoded
        encoded = flat + pos_enc_2
        encoded = tf.reshape(patches(tf.squeeze(unpatch(unflatten(encoded, 1), 1), axis = 1), self.patch_size[0]), [-1,self.num_patches[0], self.projection_dim[0]])
        encoded = encoded + pos_enc_1
        encoded = self.dense(encoded)
        return encoded


class DeepPatchEncoder_CNN(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int=128,
                 patch_size:List[int]=[16,8],
                 num_channels:int=1,
                 dropout:float=.2,
                 bias:bool=False,
                 ):
        super(DeepPatchEncoder_CNN, self).__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.bias = bias
        self.num_patches = [(self.img_size//patch)**2 for patch in self.patch_size]
        self.projection_dim = [self.num_channels*patch**2 for patch in self.patch_size]
        self.kernel_size = [patch//2 for patch in self.patch_size]
        self.patch_size_extended = [self.img_size] + self.patch_size
        # Layers
        self.seq = tf.keras.Sequential([])
        if self.patch_size[0]>self.patch_size[-1]:
            self.strides_size = [self.patch_size_extended[i]//self.patch_size_extended[i+1] for i in range(len(self.patch_size))]
            for i in range(len(self.patch_size)):
                self.seq.add(tf.keras.layers.Conv2D(self.num_patches[i], kernel_size=self.kernel_size[i], strides=self.strides_size[i], padding='same', use_bias=self.bias))
                self.seq.add(tf.keras.layers.BatchNormalization())
        else:
            self.strides_size = [self.patch_size_extended[i+1]//self.patch_size_extended[i] for i in range(len(self.patch_size))]
            for i in range(len(self.patch_size)):
                self.seq.add(tf.keras.layers.Conv2DTranspose(self.num_patches[i], kernel_size=self.kernel_size[i], strides=self.strides_size[i], padding='same', use_bias=self.bias))
                self.seq.add(tf.keras.layers.BatchNormalization())

    def call(self, X:tf.Tensor):
        X = tf.transpose(tf.expand_dims(X, axis=1), perm=[0,4,2,3,1])
        X = self.seq(X)
        X = tf.reshape(tf.transpose(X, perm=[0,4,2,3,1]), shape=[-1, self.num_patches[-1], self.projection_dim[-1]])
        X = resampling(X, [self.num_patches[-1], self.num_patches[0]], [self.projection_dim[-1],self.projection_dim[0]])
        return X


class Resampling(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int=128,
                 patch_size:List[int]=[16,8],
                 num_channels:int=1,
                 dropout:float=0.,
                 trainable:bool=True,
                 ):
        super(Resampling, self).__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = [(self.img_size//patch)**2 for patch in self.patch_size]
        self.num_channels = num_channels
        self.projection_dim = [self.num_channels*patch**2 for patch in self.patch_size]
        self.trainable = trainable
        # Layers
        if trainable:
            self.BN = tf.keras.layers.BatchNormalization()
            self.LeakyReLU = tf.keras.layers.LeakyReLU()
            self.drop = tf.keras.layers.Dropout(dropout)
            if self.patch_size[0]>self.patch_size[1]:
                self.rs = tf.keras.layers.Conv2D(self.num_patches[1], (3,3), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)
            else:
                self.rs = tf.keras.layers.Conv2DTranspose(self.num_patches[1], (3,3), strides=(2,2), padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)

    def call(self, encoded:tf.Tensor):
        if self.trainable:
            X_patch = unflatten(encoded, self.num_channels)
            X_patch = tf.transpose(X_patch, perm=[0,4,2,3,1])
            X_patch = tf.map_fn(fn=lambda y: self.rs(y), elems = X_patch)
            X_patch = self.BN(X_patch)
            X_patch = self.drop(X_patch)
            X_patch = self.LeakyReLU(X_patch)
            X_patch = tf.transpose(X_patch, perm=[0,4,2,3,1])
            return tf.reshape(X_patch, [-1, self.num_patches[1], self.projection_dim[1]])
        else:
            return resampling(encoded, self.num_patches, self.projection_dim, self.num_channels)

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

    def call(self, x):
        x = self.D1(x)
        x = tf.keras.activations.gelu(x)
        x = self.Drop1(x)
        x = self.D2(x)
        x = tf.keras.activations.gelu(x)
        x = self.Drop2(x)
        return x

## ReAttention
class ReAttention(tf.keras.layers.Layer):
    def __init__(self,
                 dim,
                 num_patches,
                 num_channels=1,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.2,
                 proj_drop=0.2,
                 apply_transform=True,
                 transform_scale=False,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_size = int(np.sqrt(dim))

        head_dim = self.dim // self.num_heads
        self.apply_transform = apply_transform
        self.scale = qk_scale or head_dim ** -0.5
        if apply_transform:
            self.reatten_matrix = tf.keras.layers.Conv2D(self.num_patches, 1)
            self.var_norm = tf.keras.layers.BatchNormalization()
            self.qconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=qkv_bias)
            self.kconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=qkv_bias)
            self.vconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=qkv_bias)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.qconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=qkv_bias)
            self.kconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=qkv_bias)
            self.vconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=qkv_bias)
        
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)
    
    def create_queries(self, x, letter):
        if letter=='q':
            x = unflatten(x, self.num_channels)
            x = tf.map_fn(fn=lambda y: tf.keras.activations.gelu(self.qconv2d(y)), elems=x)
        if letter == 'k':
            x = unflatten(x, self.num_channels)
            x = tf.map_fn(fn=lambda y: tf.keras.activations.gelu(self.kconv2d(y)), elems=x)
        if letter == 'v':
            x = unflatten(x, self.num_channels)
            x = tf.map_fn(fn=lambda y: tf.keras.activations.gelu(self.vconv2d(y)), elems=x)

        x = tf.reshape(x, shape=[-1, self.num_patches, self.dim])
        x = tf.reshape(x, shape = [-1, self.num_patches, self.num_heads, self.dim//self.num_heads, 1])
        x = tf.transpose(x, perm = [4,0,2,1,3])
        return x[0]

    def call(self, x, atten=None):
        _, N, C = x.shape.as_list()
        q = self.create_queries(x, 'q')
        k = self.create_queries(x, 'k')
        v = self.create_queries(x, 'v')
        attn = (tf.linalg.matmul(q, k, transpose_b = True)) * self.scale
        attn = tf.keras.activations.softmax(attn, axis = -1)
        attn = self.attn_drop(attn)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        attn_next = attn
        x = tf.reshape(tf.transpose(tf.linalg.matmul(attn, v), perm = [0,2,1,3]), shape = [-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_next


## Transformer Encoder
class AttentionTransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int,
                 patch_size:int,
                 num_channels:int,
                 num_heads:int,
                 transformer_layers:int,
                 hidden_dim:int,
                 attn_drop:float,
                 proj_drop:float,
                 ):
        super().__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (self.img_size//self.patch_size)**2
        self.projection_dim = self.num_channels*self.patch_size**2
        self.transformer_layers = transformer_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        # Layers
        self.LN1 = []
        self.LN2 = []
        self.Attn = []
        self.FF = []
        for _ in range(self.transformer_layers):
            self.LN1.append(tf.keras.layers.LayerNormalization())
            self.LN2.append(tf.keras.layers.LayerNormalization())
            self.Attn.append(
                tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,
                                                   key_dim=self.projection_dim,
                                                   dropout=self.attn_drop,
                                                  )
            )
            self.FF.append(
                FeedForward(projection_dim = self.projection_dim,
                                       hidden_dim = self.hidden_dim,
                                       dropout = self.proj_drop,
                                       )
            )

    def call(self, encoded_patches):
        for i in range(self.transformer_layers):
            encoded_patch_attn = self.Attn[i](encoded_patches, encoded_patches)
            encoded_patches = tf.keras.layers.Add()([encoded_patch_attn, encoded_patches])
            encoded_patches = self.LN1[i](encoded_patches)
            encoded_patch_FF = self.FF[i](encoded_patches)
            encoded_patches = tf.keras.layers.Add()([encoded_patch_FF, encoded_patches])
            encoded_patches = self.LN2[i](encoded_patches)
        return encoded_patches


class ReAttentionTransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int,
                 patch_size:int,
                 num_channels:int,
                 num_heads:int,
                 transformer_layers:int,
                 hidden_dim:int,
                 attn_drop:float,
                 proj_drop:float,
                 ):
        super().__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (self.img_size//self.patch_size)**2
        self.projection_dim = self.num_channels*self.patch_size**2
        self.transformer_layers = transformer_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        # Layers
        self.LN1 = []
        self.LN2 = []
        self.ReAttn = []
        self.FF = []
        for _ in range(self.transformer_layers):
            self.LN1.append(tf.keras.layers.LayerNormalization())
            self.LN2.append(tf.keras.layers.LayerNormalization())
            self.ReAttn.append(
                ReAttention(dim = self.projection_dim,
                                  num_patches = self.num_patches,
                                  num_channels = self.num_channels,
                                  num_heads = self.num_heads,
                                  attn_drop = self.attn_drop,
                                  )
            )
            self.FF.append(
                FeedForward(projection_dim = self.projection_dim,
                                       hidden_dim = self.hidden_dim,
                                       dropout = self.proj_drop,
                                       )
            )

    def call(self, encoded_patches):
        for i in range(self.transformer_layers):
            encoded_patch_attn, _ = self.ReAttn[i](encoded_patches)
            encoded_patches = encoded_patch_attn + encoded_patches
            encoded_patches = self.LN1[i](encoded_patches)
            encoded_patches = self.FF[i](encoded_patches) + encoded_patches
            encoded_patches = self.LN2[i](encoded_patches)
        return encoded_patches

# Skip connection
class SkipConnection(tf.keras.layers.Layer):
    def __init__(self,
                 img_size,
                 patch_size,
                 num_channels=3,
                 num_heads=8,
                 attn_drop:float=.2,
                 ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.attn_drop = attn_drop
        self.num_patches = (self.img_size//self.patch_size)**2
        self.projection_dim = self.num_channels*self.patch_size**2
        self.Attn = tf.keras.layers.MultiHeadAttention(self.num_heads, self.projection_dim, self.projection_dim, self.attn_drop)

        
    def call(self, q, v):
        return self.Attn(q,v)


# Models
## HViT
class HViT_UNet(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int=128,
                 patch_size:List[int]=[16,8,4],
                 num_channels:int=1,
                 num_heads:int=8,
                 transformer_layers:List[int]=[5,5],
                 size_bottleneck:int=2,
                 hidden_unit_factor:float=.5,
                 drop_attn:float=.2,
                 drop_proj:float=.2,
                 drop_rs:float=.2,
                 drop_linear:float=.4,
                 pos_enc_type:str='linear',
                 trainable_rs:bool=True,
                 original_attn:bool=True,
                 ):
        super(HViT_UNet, self).__init__()
        # Validations
        assert pos_enc_type in ['linear','conv'], f"Positional encoding must either be 'linear' or 'conv'."
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
        self.drop_rs = drop_rs
        self.drop_linear = drop_linear
        self.pos_enc_type = pos_enc_type
        self.trainable_rs = trainable_rs
        self.original_attn = original_attn
        self.num_patches = [(self.img_size//patch)**2 for patch in self.patch_size]
        self.projection_dim = [self.num_channels*patch**2 for patch in self.patch_size]
        self.hidden_units = [int(hidden_unit_factor*proj) for proj in self.projection_dim]
        # Layers
        ##Positional Encoding
        if self.pos_enc_type=='linear':
            self.DPE = DeepPatchEncoder(self.img_size, self.patch_size, self.num_channels)
        elif self.pos_enc_type=='conv':
            self.DPE = DeepPatchEncoder_CNN(self.img_size, self.patch_size, self.num_channels)
        ##Encoder
        self.Encoder = []
        self.Encoder_RS = []        
        if self.original_attn:
            for i in range(len(patch_size)-1):
                self.Encoder.append(AttentionTransformerEncoder(self.img_size,self.patch_size[i],self.num_channels,self.num_heads,self.transformer_layers[i], self.hidden_units[i],self.drop_attn,self.drop_proj))
                self.Encoder_RS.append(Resampling(self.img_size, self.patch_size[i:i+2], self.num_channels, self.drop_rs, self.trainable_rs))
        else:
            for i in range(len(patch_size)-1):
                self.Encoder.append(ReAttentionTransformerEncoder(self.img_size,self.patch_size[i],self.num_channels,self.num_heads,self.transformer_layers[i], self.hidden_units[i],self.drop_attn,self.drop_proj))
                self.Encoder_RS.append(Resampling(self.img_size, self.patch_size[i:i+2], self.num_channels, self.drop_rs, self.trainable_rs))
        ##BottleNeck
        if self.original_attn:
            self.BottleNeck = tf.keras.Sequential([AttentionTransformerEncoder(self.img_size,self.patch_size[-1],self.num_channels,self.num_heads,self.size_bottleneck, self.hidden_units[-1],self.drop_attn,self.drop_proj)])
        else:
            self.BottleNeck = tf.keras.Sequential([ReAttentionTransformerEncoder(self.img_size,self.patch_size[-1],self.num_channels,self.num_heads,self.size_bottleneck, self.hidden_units[-1],self.drop_attn,self.drop_proj)])
        ##Decoder
        self.Decoder = []
        self.Decoder_RS = []         
        if self.original_attn:
            for i in range(len(patch_size)-1):
                self.Decoder_RS.append(Resampling(self.img_size, self.patch_size_rev[i:i+2], self.num_channels, self.drop_rs, self.trainable_rs))
                self.Decoder.append(AttentionTransformerEncoder(self.img_size,self.patch_size_rev[i+1],self.num_channels,self.num_heads,self.transformer_layers[len(patch_size)-(i+2)], self.hidden_units[len(patch_size)-(i+2)],self.drop_attn,self.drop_proj))
                
        else:
            for i in range(len(patch_size)-1):
                self.Decoder_RS.append(Resampling(self.img_size, self.patch_size_rev[i:i+2], self.num_channels, self.drop_rs, self.trainable_rs))
                self.Decoder.append(ReAttentionTransformerEncoder(self.img_size,self.patch_size_rev[i+1],self.num_channels,self.num_heads,self.transformer_layers[len(patch_size)-(i+2)], self.hidden_units[len(patch_size)-(i+2)],self.drop_attn,self.drop_proj))
        ## Skip connections
        self.SkipConnections = []
        for i in range(len(patch_size)-1):
            self.SkipConnections.append(SkipConnection(self.img_size, self.patch_size_rev[i+1], self.num_channels, self.num_heads, self.drop_attn))

    def call(self, X:tf.Tensor):
        # Patch
        encoded = self.DPE(X)
        # Encoder
        encoded_list = []
        for i in range(len(patch_size)-1):
            encoded = self.Encoder[i](encoded)
            encoded_list.append(encoded)
            encoded = self.Encoder_RS[i](encoded)
        print('Encoded done')
        # BottleNeck
        encoded = self.BottleNeck(encoded)
        print("BottleNeck done")
        # Decoder
        encoded_list = encoded_list[-1::-1]
        for i in range(len(patch_size)-1):
            print("Decoder iteration", i)
            encoded = self.Decoder_RS[i](encoded)
            encoded = self.Decoder[i](encoded)
            encoded = self.SkipConnections[i](encoded_list[i], encoded)
        # Return original image
        Y = X + tf.squeeze(unpatch(unflatten(encoded, self.num_channels), self.num_channels), axis = 1)
        return Y
