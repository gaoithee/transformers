import ast
import copy
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
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

from nltk.translate.bleu_score import sentence_bleu
from handcoded_tokenizer import STLTokenizer

import networkx as nx
import phis_generator_depth

from datasets import load_dataset

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


class DatasetProcessor:
    def __init__(self, dataset_name, split="train", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.original_dataset = pd.read_pickle(dataset_name)  # Load the dataset from the pickle file
        self.processed_dataset = self._create_processed_dataset()

    def _create_processed_dataset(self):
        # Transform a single entry
        def transform_entry(entry):
            # Convert 'Embedding' from string to list of floats if necessary
            formula_embedding = entry['Embedding512']
            encoder_hidden_states = torch.tensor(formula_embedding, dtype=torch.float32).to(self.device)

            # Convert 'Encoded_Formula' from string to list of integers
            encoded_formula = entry['Encoded_Formula']
            input_ids = encoded_formula[:-1]  # All tokens except the last
            labels = encoded_formula[1:]  # All tokens except the first
            attention_mask = [0 if token == 1 else 1 for token in input_ids]
            
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)            
            labels = torch.tensor(labels, dtype=torch.long).to(self.device)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)


            # Return only the transformed columns
            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
                'encoder_hidden_states': encoder_hidden_states
            }

        # Apply the transformation to each row in the dataset using pandas .apply()
        transformed_data = self.original_dataset.apply(transform_entry, axis=1)

        return transformed_data

    def get_processed_dataset(self):
        return self.processed_dataset


class DatasetProcessor2:
    def __init__(self, dataset_name, split="train", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
#        self.original_dataset = pd.read_pickle(dataset_name)
        self.original_dataset = load_dataset('csv', data_files=dataset_name, split=split)
        self.processed_dataset = self._create_processed_dataset()

    def _create_processed_dataset(self):
        def transform_entry(entry):
            # Convertire 'Embedding' da stringa a lista di float e poi a tensor
            formula_embedding = eval(entry['Embedding']) # Converti la stringa in lista
            encoder_hidden_states = torch.tensor(formula_embedding, dtype=torch.float32).to(self.device)

            # Convertire 'Encoded_Formula' da stringa a lista di interi
            encoded_formula = eval(entry['Encoded_Formula'])
            input_ids = encoded_formula[:-1]  # Tutti i token tranne l'ultimo
            labels = encoded_formula[1:]  # Tutti i token tranne il primo
            attention_mask = [0 if token == 1 else 1 for token in input_ids]

            # Restituiamo solo le nuove colonne
            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
                'encoder_hidden_states': encoder_hidden_states.tolist()  # Convertire tensor in lista per compatibilità Dataset
            }

        # Creiamo un dataset nuovo applicando `.map()` con tqdm
        # removed_columns = list(self.original_dataset.columns)
        new_dataset = self.original_dataset.map(
            transform_entry, num_proc=1, remove_columns=self.original_dataset.column_names
        )

        # Convertiamo input_ids, labels e attention_mask in tensori PyTorch
        def convert_to_tensors(batch):
            batch["input_ids"] = [torch.tensor(x, dtype=torch.long).to(self.device) for x in batch["input_ids"]]
            batch["labels"] = [torch.tensor(x, dtype=torch.long).to(self.device) for x in batch["labels"]]
            batch["attention_mask"] = [torch.tensor(x, dtype=torch.long).to(self.device) for x in batch["attention_mask"]]
            batch["encoder_hidden_states"] = [torch.tensor(x, dtype=torch.float32).to(self.device) for x in batch["encoder_hidden_states"]]
            return batch

        # Applichiamo la conversione in batch per efficienza
        new_dataset = new_dataset.map(convert_to_tensors, batched=True)

        return new_dataset

    def get_processed_dataset(self):
        return self.processed_dataset

class DatasetProcessorOLD:
    def __init__(self, dataset_name, split="train", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.original_dataset = load_dataset('csv', data_files=dataset_name, split=split)
        self.processed_dataset = self._create_processed_dataset()

    def _create_processed_dataset(self):
        def transform_entry(entry):
            # Convertire 'Embedding' da stringa a lista di float e poi a tensor
            formula_embedding = eval(entry['Embedding'])  # Converti la stringa in lista
            encoder_hidden_states = torch.tensor(formula_embedding, dtype=torch.float32).to(self.device)

            # Convertire 'Encoded_Formula' da stringa a lista di interi
            encoded_formula = eval(entry['Encoded_Formula'])  # Converti la stringa in lista
            input_ids = encoded_formula[:-1]  # Tutti i token tranne l'ultimo
            labels = encoded_formula[1:]  # Tutti i token tranne il primo
            attention_mask = [0 if token == 1 else 1 for token in input_ids]

            # Restituiamo solo le nuove colonne
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long).to(self.device),
                'labels': torch.tensor(labels, dtype=torch.long).to(self.device),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long).to(self.device),
                'encoder_hidden_states': encoder_hidden_states
            }

        # Applica la trasformazione con map() e mostra la barra di avanzamento
        processed_dataset = self.original_dataset.map(
            transform_entry, desc="Processing Dataset", num_proc=1
        )

        return processed_dataset

    def get_processed_dataset(self):
        return self.processed_dataset


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
        self.df = df['train']
        self.device = device

        encoded_formulae = []
        formulae_embeddings = []
        input_ids = []
        labels = []
        attention_masks = []

        for idx in range(len(self.df)):
            # Extract the encoded formula (tokenized input sequence) from the DataFrame
            # encoded_formula = self.df['Encoded_Formula'][idx]
            # Convert the string representation of a list back to a Python list using ast.literal_eval
            encoded_formula = ast.literal_eval(self.df['Encoded_Formula'][idx])
            # encoded_formula = [int(x) for x in encoded_formula.split()]
            encoded_formulae.append(encoded_formula)

            # Extract the precomputed formula embedding (hidden states) from the DataFrame
            formula_embedding = self.df['Embedding'][idx]

            # Clean the string and convert it back to a tensor
            # formula_embedding = formula_embedding.replace("tensor(", "").rstrip(")")
            # formula_embedding = eval(formula_embedding)
            formula_embedding = ast.literal_eval(formula_embedding.strip())
            encoder_hidden_states = torch.tensor(formula_embedding, dtype=torch.float32).to(self.device)
            formulae_embeddings.append(encoder_hidden_states)
            
            # Define the input_ids by excluding the last token (shifted tokens for prediction)
            input_ids.append(torch.tensor(encoded_formula[:-1], dtype=torch.long).to(self.device))  # All tokens except the last
            # Define the labels by excluding the first token (shifted tokens for teacher forcing)
            labels.append(torch.tensor(encoded_formula[1:], dtype=torch.long).to(self.device))     # All tokens except the first

            # Create the attention mask to indicate which tokens should be attended to.
            # Tokens equal to '1' (typically padding tokens) will be masked (set to 0), 
            # and the rest will be visible (set to 1).
            attention_mask = [0 if token == 1 else 1 for token in encoded_formula[:-1]]  # Use encoded_formula for mask
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)
            attention_masks.append(attention_mask)

        # Create the DataFrame with the processed tensors
        self.df = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_masks,
            'encoder_hidden_states': formulae_embeddings
        }

        # self.df = pd.DataFrame(temp, device=device) 

    def __len__(self):
        """
        Returns the length of the dataset, i.e., the number of examples in the DataFrame.
        
        Returns:
        - Length of the DataFrame (number of samples).
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves the dataset item at the given index.
        
        Args:
        - idx: The index of the sample to retrieve.
        
        Returns:
        - A dictionary containing the input data for the model.
        """
        return {
            'input_ids': self.df['input_ids'][idx],
            'labels': self.df['labels'][idx],
            'attention_mask': self.df['attention_mask'][idx],
            'encoder_hidden_states': self.df['encoder_hidden_states'][idx]
        }

############################################################################################################################

#               METRICS

def token_division(input_string):
    tokenizer = STLTokenizer('tokenizer_files/tokenizer.json')
    return [element for element in tokenizer.tokenize(input_string) if element != "pad"] 



def bleu_score(dataset):

    bleu_scores = []

    for idx in range(len(dataset)):
        gold = token_division(dataset["Gold Formula"][idx])
        generated = token_division(dataset["Generated Formula"][idx])

        bleu_scores.append(sentence_bleu(gold, generated))

    return np.min(bleu_scores), np.mean(bleu_scores), np.max(bleu_scores)



def exact_match(dataset, gold_formula_column: str, generated_formula_column: str):

    percentage = []

    for idx in range(len(dataset)):
        gold = token_division(dataset[gold_formula_column][idx])
        generated = token_division(dataset[generated_formula_column][idx])

        match_count = 0
        for gold_token, gen_token in zip(gold, generated):
            if gold_token == gen_token:
                match_count += 1

        percentage.append(match_count/len(gold))


        return np.mean(percentage)   
    


def cosine_similarity(dataset):
    
    similarities = []
    
    for idx in range(len(dataset)):
        gold = ast.literal_eval(dataset["Embedding Gold Formula"][idx])
        gen = ast.literal_eval(dataset["Embedding Generated Formula"][idx])

        dot_product = np.dot(gold, gen)
        gold_norm = np.linalg.norm(gold)
        gen_norm = np.linalg.norm(gen)

        similarities.append(dot_product / (gold_norm * gen_norm))

    return np.min(similarities), np.mean(similarities), np.max(similarities) 


def euclidean_distance(dataset):

    distances = []

    for idx in range(len(dataset)):

        gold = torch.tensor(ast.literal_eval(dataset["Embedding Gold Formula"][idx]))
        generated = torch.tensor(ast.literal_eval(dataset["Embedding Generated Formula"][idx]))

        distances.append(torch.dist(gold, generated))

    return np.min(distances), np.mean(distances), np.max(distances)     


#######################################################################################################

def get_name_given_type(formula):
    """
    Returns the type of node (as a string) of the top node of the formula/sub-formula
    """
    name_dict = {And: 'and', Or: 'or', Not: 'not', Eventually: 'F', Globally: 'G', Until: 'U',
                 Atom: 'x'}
    return name_dict[type(formula)]


def get_id(child_name, name, label_dict, idx):
    """
    Get unique identifier for a node
    """
    while child_name in label_dict.keys():  # if the name is already present
        idx += 1
        child_name = name + "(" + str(idx) + ")"
    return child_name, idx                  # returns both the child name and the identifier


def get_temporal_list(temporal_node):
    """
    Returns the features vector for temporal nodes (the two bounds of the temporal interval)
    Variant and num_arg modify the length of the list to return (3, 4 or 5)
    """
    left = float(temporal_node.left_time_bound) if temporal_node.unbound is False else 0.
    right = float(temporal_node.right_time_bound) if (temporal_node.unbound is False and
                                                      temporal_node.right_unbound is False) else -1.
    vector_l = [left, right, 0.]      # third slot for sign and fourth for threshold        # add another slot for argument number
    return vector_l


def add_internal_child(current_child, current_idx, label_dict):
    child_name = get_name_given_type(current_child) + '(' + str(current_idx) + ')'
    child_name, current_idx = get_id(child_name, get_name_given_type(current_child), label_dict, current_idx)
    return child_name, current_idx


def add_leaf_child(node, name, label_dict, idx):
    """
    Add the edges and update the label_dictionary and the identifier count for a leaf node (variable)
    variant = ['original', 'threshold-sign', 'all-in-var']
    shared_var = [True, False] denotes if shared variables for all the DAG or single variables (tree-like)
    num_arg = [True, False] if true argument number is one-hot encoded in the feature vector
    until_right is a flag to detect when the argument number encoding should be 1
    """
    new_e = []
    label_dict[name] = [0., 0., 0.]     # te
    atom_idx =str(node).split()[0] +  '(' + str(idx) + ')'
    # different names for the same variables (e.g. x_1(5), x_1(8))
    idx += 1
    if atom_idx not in label_dict.keys():
        label_dict[atom_idx] = [0., 0., 0.]

    if str(node).split()[1] == '<=':
        label_dict[name] = [0., 0., round(node.threshold, 4)]
    else:
        label_dict[name] = [0., 0., round(node.threshold, 4)]
    new_e.append([name, atom_idx])
    return new_e, label_dict, idx+1


def traverse_formula(formula, idx, label_dict):
    current_node = formula
    edges = []
    if type(current_node) is not Atom:
        current_name = get_name_given_type(current_node) + '(' + str(idx) + ')'
        if (type(current_node) is And) or (type(current_node) is Or) or (type(current_node) is Not):
            label_dict[current_name] = [0., 0., 0. ] # temp_left, temp_right, threshold
        else:
            label_dict[current_name] = get_temporal_list(current_node)
        if (type(current_node) is And) or (type(current_node) is Or) or (type(current_node) is Until):
            left_child_name, current_idx = add_internal_child(current_node.left_child, idx + 1, label_dict)
            edges.append([current_name, left_child_name])
            if type(current_node.left_child) is Atom:
                e, d, current_idx = add_leaf_child(current_node.left_child, left_child_name, label_dict, current_idx+1)
                edges += e
                label_dict.update(d)
            e, d = traverse_formula(current_node.left_child, current_idx, label_dict)
            edges += e
            label_dict.update(d)
            right_child_name, current_idx = add_internal_child(current_node.right_child, current_idx + 1, label_dict)
            edges.append([current_name, right_child_name])
            if type(current_node.right_child) is Atom:
                e, d, current_idx = add_leaf_child(current_node.right_child, right_child_name, label_dict,
                                                   current_idx+1)
                edges += e
                label_dict.update(d)
            e, d = traverse_formula(current_node.right_child, current_idx, label_dict)
            edges += e
            label_dict.update(d)
        else:
            # eventually, globally, not
            child_name, current_idx = add_internal_child(current_node.child, idx + 1, label_dict)
            edges.append([current_name, child_name])
            if type(current_node.child) is Atom:
                e, d, current_idx = add_leaf_child(current_node.child, child_name, label_dict, current_idx+1)
                edges += e
                label_dict.update(d)
            e, d = traverse_formula(current_node.child, current_idx, label_dict)
            edges += e
            label_dict.update(d)
    return edges, label_dict


def build_dag(formula):
    edges, label_dict = traverse_formula(formula, 0, {})
    graph = nx.from_edgelist(edges, create_using=nx.DiGraph)
    assert(nx.is_directed_acyclic_graph(graph))
    return graph, label_dict


def get_depth(formula):
    phi_g = build_dag(formula)[0]
    return len(nx.dag_longest_path(phi_g)) - 1

def get_n_nodes(str_phi):
    f_split = str_phi.split()
    f_nodes_list = [sub_f for sub_f in f_split if sub_f in ['not', 'and', 'or', 'always', 'eventually', '<=', '>=',
                                                            'until']]
    return len(f_nodes_list)


def get_n_leaves(str_phi):
    phi_split = str_phi.split()
    phi_var = [sub for sub in phi_split if sub.startswith('x_')]
    return len(phi_var)


def get_n_temp(str_phi):
    phi_split = str_phi.split()
    phi_temp = [sub for sub in phi_split if sub[:2] in ['ev', 'al', 'un']]
    return len(phi_temp)


def get_n_tokens(str_phi):
    tokenizer = STLTokenizer('tokenizer_files/tokenizer.json')
    return len(tokenizer.encode(str_phi))


def get_n_depth(str_phi):
    phi = from_string_to_formula(str_phi)
    return get_depth(phi)
