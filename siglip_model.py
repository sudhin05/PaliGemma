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



