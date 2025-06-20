import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Tuple, Dict, Optional, Union, List
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


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
        return self.model,embed_toekens
        
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