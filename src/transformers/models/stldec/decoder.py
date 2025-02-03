import copy
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from utils2 import STLAttention, STLSinusoidalPositionalEmbedding
from configuration import STLConfig

# copied from ...
class STLPreTrainedModel(PreTrainedModel):
    config_class = STLConfig 
    base_model_prefix = "model" 
    supports_gradient_checkpointing = True

    # initializes the weights of `nn.Linear`, `nn.Embedding` and `STLSinusoidalPositionalEmbedding`
    def _init_weights(self, module: Union[nn.Linear, nn.Embedding, STLSinusoidalPositionalEmbedding]):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, STLSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
        }
        return dummy_inputs


class STLDecoderBlock(nn.Module):
    
    def __init__(self, embed_dim: int, 
                num_decoder_attention_heads: int,
                num_decoder_ffn_dim: int,
                dropout: float = 0.0,
                attention_dropout: float = 0.0,
                activation_dropout: float = 0.0,
                ):
        
        super().__init__()
        
        self.embed_dim = embed_dim

        # first block
        self.self_attn = STLAttention(
            embed_dim=self.embed_dim, 
            num_heads=num_decoder_attention_heads,
            dropout=dropout,
            is_decoder=True, # not used, debugging purposes
            is_causal=True, # not used, debugging purposes
        )
        self.dropout = dropout
        self.activation_fn = nn.functional.gelu
        self.activation_dropout = activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # second block
        self.encoder_attn = STLAttention(
            self.embed_dim,
            num_decoder_attention_heads,
            dropout=attention_dropout,
            is_decoder=True, # not used, debugging purposes
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # third block
        self.fc1 = nn.Linear(self.embed_dim, num_decoder_ffn_dim)
        self.fc2 = nn.Linear(num_decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        
        ###################################################################
        
        # BLOCK 1: processing what has been previously generated 

        # previous state is stored into an auxiliary variable `residual`
        residual = hidden_states

        # tries to exploit previous K, V values if there are any 
        # (practically picks up to the first 2 values stored in `past_key_value` vector)
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        # masked MHSA on the already generated sequence
        # invokes `forward` method to transform the original vector accordingly 
        hidden_states, self_attn_weights, present_key_value = self.self_attn.forward(
            hidden_states=hidden_states, # Q
            past_key_value=self_attn_past_key_value, # K, V
            attention_mask=attention_mask, # passed as input of the decoder layer
            layer_head_mask=layer_head_mask, # to deactivate certain attn layers 
            output_attentions=output_attentions, 
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # residual connection
        hidden_states = residual + hidden_states

        # normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        ###################################################################

        # BLOCK 2: cross-attn between already generated input and previous information (from the encoder)

        # initialize K, Q, attn_weights for this new attn operation
        cross_attn_present_key_value = None 
        cross_attn_weights = None

        # the important condition is that the encoder carries some information
        if encoder_hidden_states is not None:

            # previous state is stored into an auxiliary variable `residual`
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3, 4 of PAST_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

            # MHSA in cross-attn
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn.forward(
                hidden_states=hidden_states, # Q = generated output
                key_value_states=encoder_hidden_states, # K, V = encoder memory (used only in the 1st step when `use_cache = True`)
                attention_mask=encoder_attention_mask, # just pads some elements (not causal this time!)
                layer_head_mask=cross_attn_layer_head_mask, # again to mask certain heads
                past_key_value=cross_attn_past_key_value, # K, V = encoder CACHED memory (used from the 2nd step on when `use_cache = True`)
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

            # residual connection
            hidden_states = residual + hidden_states

            # normalization
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3, 4 of PRESENT_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        ###################################################################

        # BLOCK 3: FFNN (transforming some merged generated output - encoder information)

        # previous state is stored into an auxiliary variable `residual`
        residual = hidden_states

        # FFNN - core
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # residual connection
        hidden_states = residual + hidden_states

        # normalization
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache: # otherwise it picks K and V each time
            outputs += (present_key_value,)

        return outputs


class STLDecoder(STLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Extract from `config` file
        embed_dim = config.d_model
        num_decoder_attention_heads = config.decoder_attention_heads
        num_decoder_ffn_dim = config.decoder_ffn_dim
        max_position_embeddings = config.max_position_embeddings
        decoder_vocab_size = config.vocab_size
        pad_token_id = config.pad_token_id
        num_decoder_layers = config.decoder_layers
        scale_embedding = config.scale_embedding
        dropout = config.dropout
        attention_dropout = config.attention_dropout
        activation_dropout = config.activation_dropout
        decoder_layerdrop = config.decoder_layerdrop
        
        self.dropout = dropout
        self.layerdrop = decoder_layerdrop
        self.padding_idx = pad_token_id
        self.max_target_positions = max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if scale_embedding else 1.0

        # Initialize the input embedding (if not passed already)
        self.embed_tokens = nn.Embedding(decoder_vocab_size, embed_dim, self.padding_idx)
        
        # Initialize positional embedding also
        self.embed_positions = STLSinusoidalPositionalEmbedding(
            max_position_embeddings, embed_dim, self.padding_idx
        )
        
        # Initialize decoder layers (of a prespecified number)
        self.layers = nn.ModuleList([STLDecoderBlock(embed_dim, num_decoder_attention_heads, 
                                                      num_decoder_ffn_dim, dropout, 
                                                      attention_dropout, activation_dropout) 
                                     for _ in range(num_decoder_layers)])

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
