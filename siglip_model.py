import torch
import torch.nn as nn

class SiglipVisionConfig:
  def __init__(
      self,
      hidden_size = 768,
      intermediate_size = 3072,
      num_hidden_layers = 12,
      num_attention_heads = 12,
      num_channels = 3,
      image_size = 224,
      patch_size = 16,
      layer_norm_eps = 1e-6,
      attention_dropout = 0.0,
      num_image_token: int = None,
      **kwargs
  ):
    super(SiglipVisionConfig, self).__init__()
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.num_channels = num_channels
    self.image_size = image_size
    self.patch_size = patch_size
    self.layer_norm_eps = layer_norm_eps
    self.attention_dropout = attention_dropout
    self.num_image_token = num_image_token

class SiglipVisionEmbeddings(nn.Module):
  def __init__(self,config: SiglipVisionConfig):
    self.config = config
    self.embed_dim = config.hidden_size
    self.image_size - config.image_size
    self.patch_size = config.patch_size
    
    #Here padding = valid means no padding is added
    self.patch_embedding = nn.Conv2d(
      in_channels = config.num_channels,
      out_channels = self.embed_dim,
      kernel_size = self.patch_size,
      stride = self.patch_size,
      padding = "valid"
    )

    self.num_patches = (self.image_size // self.patch_size) ** 2
    #In a generic ViT we would require 1 more position to account for the class token
    self.num_positons = self.num_patches
    #In the Vanilla transformer we forced the model to learn the sinusoidal position representations
    #In this implmentation we are making the model learn the position representation on its own over time
    self.position_embedding = nn.Embedding(self.num_positons,self.embed_dim)
    """
    register_buffer(name, tensor, persistent=True)
      Add a buffer to the module.
      This is typically used to register a buffer that should not to be considered a model parameter. 
      For example, BatchNorms running_mean is not a parameter, but is part of the modules state.
        Buffers, by default, are persistent and will be saved alongside parameters. 
        This behavior can be changed by setting persistent to False. 
        The only difference between a persistent buffer and a non-persistent buffer is that the latter will not be a part of this modules state_dict.
      Buffers can be accessed as attributes using given names.
    """
    self.register_buffer("position_ids",torch.arange(self.num_positions).expand((1,-1)),persistent=False)
  
  def forward(self,img: torch.FloatTensor) -> torch.Tensor:
    batch_size,channel,height,width = img.shape
    #Now we must generate embeds using the Convolution operation
    # [ Batch_size,Embed_dim,num_patches_h,num_patches_w]
    patch_embeds = self.patch_embedding(img)
    # Now we must flatten the embeddings
    # [Batch_size,Embed_dim,num_patches_h*num_patches_w] OR
    # [Batch_size,Embed_dim,num_patches] where num_patches = num_patches_h * num_patches_w
    embeddings = patch_embeds.flatten(2)
    #Taking transpose
    # [Batch_size,num_patches,Embed_dim]
    embeddings = torch.permute(0,2,1)
    #Adding positional embeddings
    # [Batch_size,num_patches,Embed_dim] here num_patches = num_positions
    embeddings = embeddings + self.position_embedding(self.position_ids)

    return embeddings

# class SiglipEncoderLayer(nn.Module):
#   def __init__(self,config: SiglipVisionConfig):
#     super(SiglipEncoderLayer, self).__init__()
#     self.embed_dim = config.hidden_size
#     self.eps = config.layer_norm_eps
#     self.self_attn = SiglipAttention(config)
#     self.layer_norm1 =nn.LayerNorm(self.embed_dim,eps = self.eps)
#     self.mlp = SiglipMLP(config)
#     self.layer_norm2 =nn.LayerNorm(self.embed_dim,eps = self.eps)

#   def forward(self,hidden_states: torch.Tensor)->torch.Tensor:
#     residual = hidden_states
#     hidden_states = self.layer_norm1(hidden_states)
#     hidden_states,_ = self.self_attn(hidden_states)
#     residual = hidden_states + residual
#     hidden_states = self.layer_norm2(residual)
#     hidden_states = self.mlp(hidden_states)
#     hidden_states = hidden_states + residual
#     return hidden_states
    

class SiglipVisionTransformer(nn.Module):
  def __init__(self,config: SiglipVisionConfig):
    super(SiglipVisionTransformer, self).__init__()
    #embed_dim and 
    self.config = config
    embed_dim = config.hidden_size
    self.embeddings = SiglipVisionEmbeddings(config)
    self.encoder = SiglipVisionEncoder(config)
    self.post_layernorm = nn.Layer_norm(embed_dim, eps = config.layer_norm_eps)

  def forward(self, img: torch.Tensor) -> torch.Tensor:
    #This type of coding ensures that anyone reading the function knows what type of input is expected and what type of output will be returned.
    # img: [Batch_size,Channel,Height,Width] -> [Batch_size,Num_patches,Embed_dim]
    #extracting patches from the images, creates embeddings and adds position encoding
    embeds = self.embeddings(img)
    #runs it through Nx layers of the transformer encoder
    hidden_states = self.encoder(inputs_embeds = embeds)
    last_hidden_state = self.post_layernorm(hidden_states)
    return last_hidden_state

class SiglipVisionModule(nn.Module):
  def __init__(self,config:SiglipVisionConfig):
    super(SiglipVisionModule, self).__init__()
    self.config = config
    self.vision_model = SiglipVisionTransformer(config)
  
  def forward(self,img) -> tuple:
    #Abse aise hi krenge define
    #img == pixel_values
    # [Batch_size,Channels,Height,Width] -> [Batch_size,Num_patches,Embed_dim]
    return self.vision_model(img=img)



