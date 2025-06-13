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

        return final_embeddings


    def forward(
            self, 
            input_ids: torch.LongTensor=None,
            pixel_values: torch.FloatTensor=None,
            attention_mask: Optional[torch.Tensor]=None,
            kv_cache: Optional[KVCache]=None,
            ) -> Tuple:
        
        #get the input embeddings [B, seq_len, hidden_size]
        input_embeds = self.language_model.get_input_embeddings(input_ids)

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