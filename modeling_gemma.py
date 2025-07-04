import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Tuple, Dict, Optional, Union, List
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel





class KVCache():
    def __init__(self)-> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self)-> int:
        if(len(self.key_cache)==0):
            return 0
        else:
            #[Batch_size, num_headsKV, seq_len, head_dim]
            return self.key_cache[0].shape[-2] #retrives the seq_len form this
        
    def update(
            self,
            key_states:torch.Tensor,
            value_states:torch.Tensor,
            layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) < layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx],key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]



class GemmaConfig():

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


class PaliGemmaMultimodalProjector(nn.Module):
    def __init__(self, config:PaliGemmaConfig):
        super().__init_()

        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        #[Batch_szie, num_patches, hidden_size] -> [Batch_size, num_patches, projection_dim]
        hidden_states = self.linear(image_features)
        return hidden_states




class GemmaRMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):

        return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):

        output = self._norm(x.float())
        output = (1 + self.weight.float()) * output

        return output.type_as(x)

        #only need to compute one statistics for RMS norm instead of two in LayerNorm



class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)


    def forward(self, x):
        
        y = self.gate_proj(x) #[[Batch_size, seq_len, intermediate_size]
        
        y = nn.functional.gelu(y, approximate='tanh') # [Batch_size, seq_len, intermediate_size]
        j = self.up_proj(x)  #[Batch_size, seq_len, intermediate_size]
        z = y * j  #[Batch_size, seq_len, intermediate_size]
        z = self.down_proj(z)  #[Batch_size, seq_len, hidden_size]

        return z





class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim # it is set to the head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate the theta according to the formula theta_i = base^(-2i/dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        # Copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded: [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Seq_Len, Head_Dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states:torch.Tensor, n_rep:int)->torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.size

    if n_rep==1:
        return hidden_states
    hidden_states = hidden_states[:,:,None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads*n_rep, slen, head_dim)

class GemmaAttention(nn.Module):
    def __init__(self, config:GemmaConfig, layer_idx:int):
        super().__init__()

        self.config = config

        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_key_value_heads // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_attention_heads"
        

        #so for grouped query attention we have smaller number of head dimension for key and value
        self.q_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_heads, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads, bias =True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.num_key_value_heads, bias=True)
        self.o_proj = nn.Linear( self.num_heads*self.head_dim, self.hidden_size, bias=True)

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings = self.max_position_embeddings,
            base = self.rope_theta
        )



    def forward(
            self,
            hidden_states:torch.Tensor,
            attention_mask: Optional[torch.Tensor]=None,
            position_ids: Optional[torch.Tensor]=None,
            kv_cache: Optional[KVCache]=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
       

       

        bsz, q_len, _ = hidden_states.size
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        #[Batch_size, num_q_heads, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1,2)
        #[Batch_size, num_KV_heads, seq_len, head_dim]
        key_states = key_states.view(bsz,  q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        #[Batch_size, num_KV_heads, seq_len, head_dim]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)

        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
                key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_scores = torch.matmul(query_states, key_states.transpose(2,3))/math.sqrt(self.head_dim)

        assert attention_mask is not None

        attn_scores = attn_scores + attention_mask
        attn_scores = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_scores = nn.functional.dropout(attn_scores, dim=-1, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_scores, value_states)


        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # Make sure the sequence length is the second dimension. # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Concatenate all the heads together. [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q * Head_Dim]
        attn_output = attn_output.view(bsz, q_len, -1)
        # Multiply by W_o. [Batch_Size, Seq_Len_Q, Hidden_Size]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_scores







                                

    
    






class GemmaDecoderLayer(nn.Module):
    def __init__(self, config:GemmaConfig, layer_idx:int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config=config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states:torch.Tensor,
            attention_mask: Optional[torch.Tensor]=None,
            position_ids: Optional[torch.Tensor]=None,
            kv_cache: Optional[KVCache]=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        # [Batch_size, seq_len, hidden_size]
        hidden_states = self.input_layernorm(hidden_states)
        # [Batch_size, seq_len, hidden_size]
        hidden_states  = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache   
        )

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        # [Batch_size, seq_len, hidden_size]
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states






class GemmaModel(nn.Module):
    def __init__(self, config:GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.voca_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)

        self.layers = nn.ModuleList([
            GemmaDecoderLayer(self.config, layer_idx) for layer_idx in range (config.num_hidden_layers)
        ])

        self.norm = nn.GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            input_embeds: Optional[torch.Tensor] = None,
            kv_cache: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        #[Batch_sisze, seq_len, hidden_size]
        hidden_states = input_embeds

        normalizer = torch.tensor(self.config.hidden_size**.5, dtype=hidden_states.dtype, device=hidden_states.device)

        hidden_states = hidden_states*normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )

        hideen_states = self.norm(hidden_states)

        return hidden_states
    





class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = self.config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True) #logits layer

    def get_input_embeddings(self):
        return self.model.embed_toekens
        
    def tie_weights(self):
        self.lm_head.weigth = self.model.embed_tokens.weight

    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            input_embeds: Optional[torch.Tensor] = None,
            kv_cache: Optional[torch.Tensor] = None,
            ) -> Tuple:
        
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
        )
        
        hidden_states = outputs
        logits = self.ln_head(hidden_states)
        logits = logits.float()

        return_data = {
            'logits': logits,
        }

        if kv_cache is not None:
            return_data ["kv_cache"] = kv_cache

        return return_data
    




class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states




class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config:PaliGemmaConfig):
        super().__init__()

        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config) #for projectin the visual features to the same shape as the text features
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def merge_input_ids_with_image_features(self, image_features:torch.Tensor, input_embeds:torch.Tensor, input_ids:torch.Tensor, attention_mask:torch.Tensor, kv_cache:torch.Tensor):

        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device

        # B, num_patches, hidden_size
        scaled_image_features = image_features*((self.config.hidden_size)**-.5)
        final_embeddings = torch.zeros(batch_size, sequence_length, embed_dim, dtype=dtype, device=device)
  
        #shape of these mask = [B, sequence_length]
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.config.pad_token_id)
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids==self.config.pad_token_id

        #Now everythis has the same sized (hidden_sized) placeholders
        text_mask = text_mask.unsqueeze(-1).expand(-1,-1, embed_dim)
        image_mask = image_mask.unsqueeze(-1).expand(-1,-1, embed_dim)
        pad_mask = pad_mask.unsqueeze(-1).expand(-1,-1, embed_dim)

        final_embeddings = torch.where(text_mask, input_embeds, final_embeddings)
        final_embeddings = torch.masked_scatter(image_mask, scaled_image_features, final_embeddings)
        final_embeddings = torch.where(pad_mask, torch.zeros_like(final_embeddings),final_embeddings)

        ##Creating attention mask
        dtype, device = input_embeds.dtype , input_embeds.device

        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:

            #for no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )

        else:
            #generating part (during inference no masking is needed veven in the generating tokesn as we are generating one single token at a time)
            #query must be a single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            #
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        #Adding the head dimension
        #[Batch_size, q_len, kv_len] -> [Batch_size, num_heads, q_len, kv_len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

        else:

            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask==0)).to(device) 

         


        return final_embeddings, causal_mask, position_ids


    def forward(
            self, 
            input_ids: torch.LongTensor=None,
            pixel_values: torch.FloatTensor=None,
            attention_mask: Optional[torch.Tensor]=None,
            kv_cache: Optional[KVCache]=None,
            ) -> Tuple:
        
        assert torch.all(attention_mask == 1), "the input can not be padded"

        #get the input embeddings [B, seq_len, hidden_size]
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        #get the input embeddings for the image patches [B, num_patches, Embd_dim]
        selected_image_features = self.vision_tower(pixel_values.to(input_embeds.dtype))

        # [B, num_patches, Embd_dim] -> [B, num_patches, hidden_size]
        image_features = self.multi_modal_projector(selected_image_features)

        input_embeds, attention_mask, position_ids = self.merge_input_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache
        )

        return outputs