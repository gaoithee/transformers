import ast
import copy
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers.modeling_utils import PreTrainedModel
from configuration import STLConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask

import copy
import pickle
import os
from collections import deque

from stl import *

############################################################################################################################

def load_pickle(path):
    with open(path, 'rb') as f:
        x = pickle.load(f)
    return x


def dump_pickle(name, thing):
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(thing, f)


def set_time_thresholds(st):
    unbound, right_unbound = [True, False]
    left_time_bound, right_time_bound = [0, 0]
    if st[-1] == ']':
        unbound = False
        time_thresholds = st[st.index('[')+1:-1].split(",")
        left_time_bound = int(time_thresholds[0])
        if time_thresholds[1] == 'inf':
            right_unbound = True
        else:
            right_time_bound = int(time_thresholds[1])-1
    return unbound, right_unbound, left_time_bound, right_time_bound


def from_string_to_formula(st):
    root_arity = 2 if st.startswith('(') else 1
    st_split = st.split()
    if root_arity <= 1:
        root_op_str = copy.deepcopy(st_split[0])
        if root_op_str.startswith('x'):
            atom_sign = True if st_split[1] == '<=' else False
            root_phi = Atom(var_index=int(st_split[0][2]), lte=atom_sign, threshold=float(st_split[2]))
            return root_phi
        else:
            assert (root_op_str.startswith('not') or root_op_str.startswith('eventually')
                    or root_op_str.startswith('always'))
            current_st = copy.deepcopy(st_split[2:-1])
            if root_op_str == 'not':
                root_phi = Not(child=from_string_to_formula(' '.join(current_st)))
            elif root_op_str.startswith('eventually'):
                unbound, right_unbound, left_time_bound, right_time_bound = set_time_thresholds(root_op_str)
                root_phi = Eventually(child=from_string_to_formula(' '.join(current_st)), unbound=unbound,
                                      right_unbound=right_unbound, left_time_bound=left_time_bound,
                                      right_time_bound=right_time_bound)
            else:
                unbound, right_unbound, left_time_bound, right_time_bound = set_time_thresholds(root_op_str)
                root_phi = Globally(child=from_string_to_formula(' '.join(current_st)), unbound=unbound,
                                    right_unbound=right_unbound, left_time_bound=left_time_bound,
                                    right_time_bound=right_time_bound)
    else:
        # 1 - delete everything which is contained in other sets of parenthesis (if any)
        current_st = copy.deepcopy(st_split[1:-1])
        if '(' in current_st:
            par_queue = deque()
            par_idx_list = []
            for i, sub in enumerate(current_st):
                if sub == '(':
                    par_queue.append(i)
                elif sub == ')':
                    par_idx_list.append(tuple([par_queue.pop(), i]))
            # open_par_idx, close_par_idx = [current_st.index(p) for p in ['(', ')']]
            # union of parentheses range --> from these we may extract the substrings to be the children!!!
            children_range = []
            for begin, end in sorted(par_idx_list):
                if children_range and children_range[-1][1] >= begin - 1:
                    children_range[-1][1] = max(children_range[-1][1], end)
                else:
                    children_range.append([begin, end])
            n_children = len(children_range)
            assert (n_children in [1, 2])
            if n_children == 1:
                # one of the children is a variable --> need to individuate it
                var_child_idx = 1 if children_range[0][0] <= 1 else 0  # 0 is left child, 1 is right child
                if children_range[0][0] != 0 and current_st[children_range[0][0] - 1][0:2] in ['no', 'ev', 'al']:
                    children_range[0][0] -= 1
                left_child_str = current_st[:3] if var_child_idx == 0 else \
                    current_st[children_range[0][0]:children_range[0][1] + 1]
                right_child_str = current_st[-3:] if var_child_idx == 1 else \
                    current_st[children_range[0][0]:children_range[0][1] + 1]
                root_op_str = current_st[children_range[0][1] + 1] if var_child_idx == 1 else \
                    current_st[children_range[0][0] - 1]
                assert (root_op_str[:2] in ['an', 'or', 'un'])
            else:
                if children_range[0][0] != 0 and current_st[children_range[0][0] - 1][0:2] in ['no', 'ev', 'al']:
                    children_range[0][0] -= 1
                if current_st[children_range[1][0] - 1][0:2] in ['no', 'ev', 'al']:
                    children_range[1][0] -= 1
                # if there are two children, with parentheses, the element in the middle is the root
                root_op_str = current_st[children_range[0][1] + 1]
                assert (root_op_str[:2] in ['an', 'or', 'un'])
                left_child_str = current_st[children_range[0][0]:children_range[0][1] + 1]
                right_child_str = current_st[children_range[1][0]:children_range[1][1] + 1]
        else:
            # no parentheses means that both children are variables
            left_child_str = current_st[:3]
            right_child_str = current_st[-3:]
            root_op_str = current_st[3]
        left_child_str = ' '.join(left_child_str)
        right_child_str = ' '.join(right_child_str)
        if root_op_str == 'and':
            root_phi = And(left_child=from_string_to_formula(left_child_str),
                           right_child=from_string_to_formula(right_child_str))
        elif root_op_str == 'or':
            root_phi = Or(left_child=from_string_to_formula(left_child_str),
                          right_child=from_string_to_formula(right_child_str))
        else:
            unbound, right_unbound, left_time_bound, right_time_bound = set_time_thresholds(root_op_str)
            root_phi = Until(left_child=from_string_to_formula(left_child_str),
                             right_child=from_string_to_formula(right_child_str),
                             unbound=unbound, right_unbound=right_unbound, left_time_bound=left_time_bound,
                             right_time_bound=right_time_bound)
    return root_phi


def scale_trajectories(traj):
    traj_min = torch.min(torch.min(traj, dim=0)[0], dim=0)[0]
    traj_max = torch.max(torch.max(traj, dim=0)[0], dim=0)[0]
    scaled_traj = -1 + 2*(traj - traj_min) / (traj_max - traj_min)
    return scaled_traj


def standardize_trajectories(traj_data, n_var):
    means, stds = [[] for _ in range(2)]
    for i in range(n_var):
        means.append(torch.mean(traj_data[:, i, :]))
        stds.append(torch.std(traj_data[:, i, :]))
    for i in range(n_var):
        traj_data[:, i, :] = (traj_data[:, i, :] - means[i]) / stds[i]
    return traj_data

############################################################################################################################

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


# Create a `CustomDataset` class to properly format input data with respect to 
# the `input_ids`, `labels`, and `attention_mask` attributes for model training.
class CustomDataset(Dataset):
    def __init__(self, df, device='cpu'):
        """
        Initializes the dataset by storing the DataFrame and setting the device.
        
        Args:
        - df: A pandas DataFrame containing the data (e.g., `Encoded_Formula`, `Embedding`).
        - device: The device ('cpu' or 'cuda') where the tensors will be moved for processing.
        """
        self.df = df
        self.device = device  

    def __len__(self):
        """
        Returns the length of the dataset, i.e., the number of examples in the DataFrame.
        
        Returns:
        - Length of the DataFrame (number of samples).
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves a specific example from the dataset, processes it, and formats it 
        into the required structure for the model (e.g., `input_ids`, `labels`, `attention_mask`).
        
        Args:
        - idx: Index of the example to retrieve.
        
        Returns:
        - A dictionary containing the formatted input data, including:
            - `input_ids`: The tokenized input sequence (excluding the last token).
            - `labels`: The tokenized target sequence (excluding the first token).
            - `attention_mask`: A mask indicating which tokens should be attended to.
            - `encoder_hidden_states`: Embedding for each formula (precomputed, used as hidden states).
        """
        # Extract the encoded formula (tokenized input sequence) from the DataFrame
        encoded_formula = self.df['Encoded_Formula'][idx]
        # Convert the string representation of a list back to a Python list using ast.literal_eval
        encoded_formula = ast.literal_eval(encoded_formula.strip())

        # Extract the precomputed formula embedding (hidden states) from the DataFrame
        formula_embedding = self.df['Embedding'][idx]
        # Clean the string and convert it back to a tensor
        formula_embedding = formula_embedding.replace("tensor(", "").rstrip(")")
        formula_embedding = eval(formula_embedding)
        
        # Define the input_ids by excluding the last token (shifted tokens for prediction)
        input_ids = encoded_formula[:-1]  # All tokens except the last
        # Define the labels by excluding the first token (shifted tokens for teacher forcing)
        labels = encoded_formula[1:]     # All tokens except the first

        # Create the attention mask to indicate which tokens should be attended to.
        # Tokens equal to '1' (typically padding tokens) will be masked (set to 0), 
        # and the rest will be visible (set to 1).
        attention_mask = [0 if token == '1' else 1 for token in input_ids]

        # Convert `input_ids`, `labels`, and `attention_mask` to tensors and move them to the desired device (e.g., GPU or CPU)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)

        # Convert the formula embedding (list of hidden states) to a tensor and move it to the device
        encoder_hidden_states = torch.tensor(formula_embedding, dtype=torch.float32).to(self.device)

        # Return the formatted data as a dictionary, which the model can use directly for training or evaluation
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'encoder_hidden_states': encoder_hidden_states
        }
