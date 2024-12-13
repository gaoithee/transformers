import copy
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss


class STLAttention(nn.Module):
    """ Multi-Head Attention as depicted from 'Attention is all you need' """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, 
                 is_decoder: bool = False, bias: bool = False, is_causal: bool = False,):
        
        super().__init__()
        self.embed_dim = embed_dim # overall embedding dimension -> to be divided between multiple heads
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads) == self.embed_dim 
        self.scaling = self.head_dim ** -0.5 # used to normalized values when projected using `W_` matrices
        self.is_decoder = is_decoder # NOT USED
        self.is_causal = is_causal # NOT USED

        # 'roleplaying' matrices 
        # note: `embed_dim` refers to the overall number of embedding dimensions, BEFORE splitting them between the heads
        self.W_k = nn.Linear(embed_dim, embed_dim, bias = bias) 
        self.W_q = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias = bias)

        # to project the heads' outputs into a single vector!
        self.W_o = nn.Linear(embed_dim, embed_dim, bias = bias) 


    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        """ 
        Reshapes tensors to split the input (of dimension `embed_dim`) between multiple heads (of dimension `head_dim`) 
            Input: (batch_size, seq_len, embed_dim)
            Output: (batch_size, num_heads, seq_len, head_dim)
        """
        # `batch_size`` = number of sequences processed in parallel
        # `seq_len` = length of each sequence 
        # `num_heads`, `head_dim` = number of heads of the multi-attn mechanism and dimension of each of them
        # `.transpose(1, 2)` swaps the 2nd and the 3rd element
        # `.contiguous()` just asks to sotre the data in a contiguous block of memory
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    
    def forward(self, 
                hidden_states: torch.Tensor, # previous values, passed to the multi-head attn layer
                key_value_states: Optional[torch.Tensor] = None, # different key, value items (used in cross-attn)
                past_key_value: Optional[Tuple[torch.Tensor]] = None, # stores the key and values of previous steps 
                attention_mask: Optional[torch.Tensor] = None, # masks non-allowed items (padded or future ones)
                layer_head_mask: Optional[torch.Tensor] = None, # used to de-activate specific attn heads (?)
                output_attentions: bool = False # flag to control the output of the attn values
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        
        # if `key_value_states` is provided (i.e. is not None), then `is_cross_attention` is set to True:
        is_cross_attention = key_value_states is not None # imagine this as a 'context definition' step


        # `hidden_states` has dimensions `(batch_size, tgt_len, embed_dim)`, where:
        #   `batch_size` = number of items that are processed simultaneously
        #   `tgt_len` = number of tokens in the TARGET sequence
        #   `embed_dim` = embedding dimension (per token!)
        # batch_size, tgt_len = _ = hidden_states.size()
        batch_size, tgt_len, embed_dim = hidden_states.size()

        # project the current input in the `query` role:
        query = self.W_q(hidden_states) * self.scaling

        # to get `key` and `value` we have to differentiate the definition wrt the scenario: 

        # if we are using cross-attn AND there exist past key values AND the lengths match, then use them!
        if (is_cross_attention and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]):
            # then re-use K and V:
            key = past_key_value[0]
            value = past_key_value[1]

        # if we are using cross-attn BUT we do not have past key values, we compute them
        elif is_cross_attention:
            key = self._shape(self.W_k(key_value_states), -1, batch_size)
            value = self._shape(self.W_v(key_value_states), -1, batch_size)
        
        # if we do have past key values BUT we are not using cross-attn, then we project the current hidden state to new key and value items
        # and after that, concatenate them:
        elif past_key_value is not None:
            key = self._shape(self.W_k(hidden_states), -1, batch_size)
            value = self._shape(self.W_v(hidden_states), -1, batch_size)
            key = torch.cat([past_key_value[0], key], dim = 2)
            value = torch.cat([past_key_value[1], key], dim = 2)

        # if we are in a scenario in which we do not have anything pre-computed, compute them
        else:
            key = self._shape(self.W_k(hidden_states), -1, batch_size)
            value = self._shape(self.W_v(hidden_states), -1, batch_size)


        if self.is_decoder:
            past_key_value = (key, value)
        
        
        # final shape that we want the queries, keys and values to have BEFORE the attn computation
        # we do this because we want each head to be able to operate independently of the others on its set of params
        proj_shape = (batch_size * self.num_heads, -1, self.head_dim) # in fact, `head_dim` is the input dimension

        # projection: 
        query = self._shape(query, tgt_len, batch_size).view(*proj_shape)  # `._shape` prepares the vector for the proj
        key = key.reshape(*proj_shape)
        value = value.reshape(*proj_shape)

        src_len = key.size(1) 

        # bmm = batch(-wise) matrix multiplication between `query` and (TRANSPOSED) `key` 
        attn_weights = torch.bmm(query, key.transpose(1, 2))

        # check n°1
        if attn_weights.size() != (batch_size * self.num_heads, tgt_len, src_len):
            raise ValueError(f"Attention weights should be of size {(batch_size * self.num_heads, tgt_len, src_len)}, but is"
             f" {attn_weights.size()}"
            )
        # check n°2
        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, tgt_len, src_len):
                raise ValueError(
                f"Attention mask should be of size {(batch_size, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            # reshaping and application of the mask to the the upper-right part of `attn_weights` matrix
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len) + attention_mask
            # returned packed together, not divided among `num_heads` heads
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)
        
        # if it passed the checks, then normalize these values on the `key` axis (i.e. `dim = -1`)
        attn_weights = nn.functional.softmax(attn_weights, dim = -1)

        # `layer_head_mask` = 1D array of dimension `self.num_heads` that flags whether or not each head is activated
        # e.g. (0, 1, ..., 0.5) -> 1st head deactivated, 2nd activated, ..., last head partially activated (reduced contribution)
        if layer_head_mask is not None:
            # check 
            if layer_head_mask.size() != (self.num_heads, ):
                raise ValueError(f"Head mask for a single layer should be of size {(self.num_heads, )}, but is"
                                 f"{layer_head_mask.size()}"
                                 )
            # reshapes the mask from `(self.num_head, )` to `(1, self.num_heads, 1, 1)` to allow the broadcasting
            # also attn_weights is again reshaped, splitting for different heads
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
            # final reshape (aka merged again)
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)


        # if this component is asked to output the attention values, then reshape and go back (?)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None


        # apply the dropout to the `attn_weights`:
        attn_probs = nn.functional.dropout(attn_weights, p = self.dropout, training = self.training)
        # batch-wise matrix multiplication between the resulting probs and the value 
        attn_output = torch.bmm(attn_probs, value)
        # check
        if attn_output.size() != (batch_size * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        
        attn_output = attn_output.view(batch_size, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)


        attn_output = attn_output.reshape(batch_size, tgt_len, self.embed_dim)
        attn_output = self.W_o(attn_output) 

        return attn_output, attn_weights_reshaped, past_key_value


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
            is_decoder=True, # not used
            is_causal=True, # not used
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
            is_decoder=True, # not used
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


