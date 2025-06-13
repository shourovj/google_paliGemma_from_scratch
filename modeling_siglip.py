from typing import Optional, Tuple
import torch
from torch import nn
# from torchvision import models

class SiglipVisionConfig:
    def __init__(self,
                 hidden_size: int = 768,
                 intermediate_size: int=3072,
                 num_hidden_layers: int =12,
                 num_attention_heads: int =12,
                 num_channels: int =3,
                 image_size : int =224,
                 patch_size: int =16,
                 layer_norm_eps=1e-6,
                 attentin_dropout=0.0,
                 num_image_tokens: int = None,
                 **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attentin_dropout = attentin_dropout
        self.num_image_tokens = num_image_tokens 


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.cofig = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.image_size = config.image_size

        self.patch_embeddings = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = config.patch_size,
            stride = config.patch_size,
            padding = 'valid'
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.postional_embeddings = nn.Embedding(self.num_positions, self.embed_dim)

        self.register_buffer("position_ids", torch.arange(self.num_positions).expand([1,-1]), 
                             persistent=False)
        
    def forward(self, pixel_values) -> torch.Tensor:
        _, _, h, w = pixel_values.shape

        patch_embeddings = self.patch_embeddings(pixel_values)  # [b, embed_dim, h_patches', w_patches'] ()
        embeddings = patch_embeddings.flatten(2).transpose(1, 2)  # [b, num_patches, embed_dim]

        embeddings = embeddings + self.postional_embeddings(self.position_ids)

        return embeddings  # [b, num_patches, embed_dim]



class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.activation_fn = nn.GELU()
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size)
        

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states




class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.emb_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.emb_dim//self.num_heads
        self.scale = self.head_dim**-.5
        self.dropout = config.attentin_dropout

        self.key_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.query_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.value_proj = nn.Linear(self.emb_dim, self.emb_dim)


    def forward(self, hidden_states):
        b, num_patches, _ = hidden_states.size()
        #[b, num_patches, emb_dim]--> [b, num_patches, emb_dim]
        key_states = self.key_proj(hidden_states) 
        query_states = self.query_proj(hidden_states)
        value_states = self.value_proj(hidden_states)

        #now divide the emb_dims into multiple heads of head_dim
        #[b, num_patches, emb_dim]--> [b, num_patches, self.num_heads, self.head_dim] --> [b, self.num_heads, num_patches, self.head_dim]
        key_states = key_states.view(b, num_patches, self.num_heads, self.head_dim).transpose(1,2)
        query_states = query_states.view(b, num_patches, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(b, num_patches, self.num_heads, self.head_dim).transpose(1,2)

        #[b, self.num_heads, num_patches, self.head_dim] --> [b, self.num_heads, num_patches, num_patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)))*self.scale

        if attn_weights.size() != (b, self.num_heads, num_patches, num_patches):
            raise ValueError(
                f"Attention weights should be of size {(b, self.num_heads, num_patches, num_patches)}"
            )
        
        #did not apply causal masking because for image we want the relation between all the image_patches
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        #[b, self.num_heads, num_patches, num_patches] * [b, self.num_heads, num_patches, emb_dim] 
        outputs = (torch.matmul(attn_weights, value_states)).transpose(1,2).contiguous()
        attn_outputs = outputs.reshape(b, num_patches, self.emb_dim)

        return attn_outputs, attn_weights


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.embd_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm1 = nn.LayerNorm(self.embd_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(self.embd_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states  # residual connection
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states  # residual connection
        return hidden_states  # [b, num_patches, embd_dim]


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList(
            SiglipEncoderLayer(config) for _ in range(self.num_hidden_layers)
        )

    def forward(self, input):
        hidden_states = input
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embd_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder =  SiglipEncoder(config)
        self.layernorm = nn.LayerNorm(embd_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values) -> Tuple: 
        ##takes imags [b,c,h,w] and returns embeddings [b,num_patches, embd_dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(hidden_states)

        last_hidden_state = self.layernorm(last_hidden_state)

        return last_hidden_state
    

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)


    def forward(self, pixel_values) -> Tuple:
        #takes imags [b,c,h,w] and returns embeddings [b,num_patches, embd_dim]
        return self.vision_model(pixel_values)
    


if __name__ == '__main__':
    config = SiglipVisionConfig()
    vision_model = SiglipVisionModel(config)
    print(vision_model)
    total_params = sum(p.numel() for p in vision_model.parameters())
    total_trainable_params = sum(p.numel() for p in vision_model.parameters() if p.requires_grad)
    print(f'total params {total_params/10e6} and trainable {total_trainable_params/10e6}') 
    image = torch.randn(1,3,224,224)
    output = vision_model(image)
    print(output.shape)

