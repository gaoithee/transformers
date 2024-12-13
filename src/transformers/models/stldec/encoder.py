import os
import torch
from torch.nn.functional import normalize
import copy
import numpy as np
from typing import List, Optional, Tuple, Union

from phis_generator import StlGenerator
from traj_measure import BaseMeasure
from utils import from_string_to_formula, load_pickle, dump_pickle
from kernel import StlKernel

from anchor_set_generation import anchorGeneration


class STLEncoder():
    def __init__(self, 
                 embed_dim: int,
                 anchor_filename: Optional[str] = None,
                 n_vars: int = 3):
        
        self.n_vars = n_vars # passaglielo in input
        self.embed_dim = embed_dim
        self.anchorset_filename = anchor_filename
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mu = BaseMeasure(device=self.device)
        self.kernel = StlKernel(self.mu, varn=self.n_vars)

        if anchor_filename is None: 
            anchor_filename = anchorGeneration(diff_init = True, embed_dim = self.embed_dim, n_vars = self.n_vars)  
            anchor_filename+='.pickle'

        # TO DO: check on the dimensions of the anchor set and the `embed_dim` and `n_vars` values
        anchor_set = load_pickle(anchor_filename)
        if len(anchor_set) != self.embed_dim:
            raise ValueError("The anchor set and the embedding dimension do not match!")

        self.anchor_set = anchor_set

    def compute_embeddings(self, formula: List[str]):
        converted_formula = list(map(from_string_to_formula, formula))
        return self.kernel.compute_bag_bag(converted_formula, self.anchor_set)


# EXAMPLE OF USAGE
# formulae_to_embed = [
#     'not ( x_1 <= 0.0956 )', 
#     'not ( x_2 >= 1.118 )', 
#     'not ( ( not ( x_0 <= -0.692 ) and ( eventually[8,19] ( x_2 <= -1.5116 ) until[6,inf] x_2 >= -0.3382 ) ) )', 
#     '( ( x_2 >= -0.4612 or x_1 <= -1.1656 ) or x_0 <= -0.8679 )']

# here we do not pass an anchor set so the Encoder creates a new one of dimension set to `embed_dim` 
# encoder = STLEncoder(embed_dim=10)

# another option is the following: 
# encoder = STLEncoder(embed_dim=10, anchor_filename='anchor_set_10_dim.pickle')
# formulae_embeddings = encoder.compute_embeddings(formulae_to_embed)

# for i in range(len(formulae_to_embed)):
#     print(f"Formula: {formulae_to_embed[i]}")
#     embedding_str = ', '.join([f"{x:.4f}" for x in formulae_embeddings[i]])
#     print(f"Embedding: tensor([{embedding_str}])\n")

