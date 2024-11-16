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
    super().__init__()
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

    
