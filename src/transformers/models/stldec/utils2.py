import copy
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from configuration import STLConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask


class STLSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)

class STLAttention(nn.Module):
    """ Multi-Head Attention as depicted from 'Attention is all you need' """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, 
                 is_decoder: bool = False, bias: bool = False, is_causal: bool = False):
        
        super().__init__()
        self.embed_dim = embed_dim  # overall embedding dimension -> to be divided between multiple heads
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads) == self.embed_dim 
        self.scaling = self.head_dim ** -0.5  # used to normalize values when projected using `W_` matrices
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 'roleplaying' matrices 
        self.W_k = nn.Linear(embed_dim, embed_dim, bias = bias) 
        self.W_q = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias = bias)

        # to project the heads' outputs into a single vector
        self.W_o = nn.Linear(embed_dim, embed_dim, bias = bias) 


    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    
    def forward(self, 
                hidden_states: torch.Tensor,  # previous values, passed to the multi-head attn layer
                key_value_states: Optional[torch.Tensor] = None,  # different key, value items (used in cross-attn)
                past_key_value: Optional[Tuple[torch.Tensor]] = None,  # stores the key and values of previous steps 
                attention_mask: Optional[torch.Tensor] = None,  # masks non-allowed items (padded or future ones)
                layer_head_mask: Optional[torch.Tensor] = None,  # used to de-activate specific attn heads
                output_attentions: bool = False  # flag to control the output of the attn values
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        is_cross_attention = key_value_states is not None  # cross-attn if key_value_states is not None

        batch_size, tgt_len, embed_dim = hidden_states.size()

        # Project the current input in the `query` role:
        query = self.W_q(hidden_states) * self.scaling

        if (is_cross_attention and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]):
            key = past_key_value[0]
            value = past_key_value[1]
        elif is_cross_attention:
            key = self._shape(self.W_k(key_value_states), -1, batch_size)
            value = self._shape(self.W_v(key_value_states), -1, batch_size)
        elif past_key_value is not None:
            key = self._shape(self.W_k(hidden_states), -1, batch_size)
            value = self._shape(self.W_v(hidden_states), -1, batch_size)
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        else:
            key = self._shape(self.W_k(hidden_states), -1, batch_size)
            value = self._shape(self.W_v(hidden_states), -1, batch_size)

        if self.is_decoder:
            past_key_value = (key, value)
        
        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)

        query = self._shape(query, tgt_len, batch_size).view(*proj_shape) 
        key = key.reshape(*proj_shape)
        value = value.reshape(*proj_shape)

        src_len = key.size(1)

        
        ######################################################################################################

        # 'traditional' attention computation
        # i.e. softmax(Q*K^T / sqrt(d_model) + self_attn_mask) * V

        # Batch-wise matrix multiplication between `query` and (TRANSPOSED) `key`
        attn_weights = torch.bmm(query, key.transpose(1, 2))

        if attention_mask is not None:
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)
        
        # Normalize values on the `key` axis (dim=-1)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # if layer_head_mask is not None:
        #     attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
        #     attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Batch-wise matrix multiplication between the resulting probs and the value
        attn_output = torch.bmm(attn_probs, value)

        ######################################################################################################

        attn_output = attn_output.view(batch_size, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(batch_size, tgt_len, self.embed_dim)
        attn_output = self.W_o(attn_output) 

        return attn_output, None, past_key_value
