import os
import torch
from torch.nn.functional import normalize
import copy
import numpy as np
import pandas as pd
import stl 
from encoder import STLEncoder
from handcoded_tokenizer import STLTokenizer

from phis_generator_depth import StlGenerator
from traj_measure import BaseMeasure
from utils import from_string_to_formula, load_pickle, dump_pickle, get_depth
from kernel import StlKernel


encoder = STLEncoder(embed_dim = 512, anchor_filename = "anchor_set_512.pickle")
tokenizer = STLTokenizer('tokenizer_files/tokenizer.json')


# Helper function to generate and filter formulae
def generate_and_filter(n_phis, n_vars, depth):
    sampler = StlGenerator(max_depth=depth, min_depth=depth)
    sampled_objs = sampler.bag_sample(bag_size=n_phis, nvars=n_vars)

    # convert to string
    sampled_objs = list(map(str, sampled_objs))

    lengths = []
    for obj in sampled_objs:
        lengths.append(len(tokenizer.encode(obj)))

    # filter sampled_objs where length is less than 500
    
    filtered_objs = [obj for i, obj in enumerate(sampled_objs) if lengths[i] < 500]
    return filtered_objs
#     return sampled_objs

def generate_formulae_depth(n_phis, n_vars, depth):

    # Generate initial batch of formulae
    formulae = generate_and_filter(n_phis, n_vars, depth)

    # If we don't have enough formulae, regenerate until we meet the required number
    while len(formulae) < n_phis:
        delta = n_phis - len(formulae)
        additional_formulae = generate_and_filter(delta, n_vars, depth)
        formulae.extend(additional_formulae)

    # Truncate the list to exactly n_phis formulae if needed
    # formulae = formulae[:n_phis]

    return formulae

def embed_generated_formulae(df):
    # sampled_formulae = list(map(str, df['Formula Obj']))
    formulae_embeddings = encoder.compute_embeddings(df)
    return formulae_embeddings.tolist()


# df = pd.read_csv('datasets/train_set.csv')
df = pd.read_pickle('datasets/easysk_train_set.pkl')
formulae_to_embed = df['Formula']
# formulae_to_embed = ["always ( eventually[6,20] ( x_1 >= -0.7636 ) )", "always ( eventually[1,25] ( x_1 >= -0.5783 ) )"]

# here we do not pass an anchor set so the Encoder creates a new one of dimension set to `embed_dim` 
# encoder = STLEncoder(embed_dim=1024, anchor_filename='anchor_set_1024_dim.pickle')
# print('computing embeddings')

formulae_embeddings = encoder.compute_embeddings(formulae_to_embed)

# print(formulae_embeddings.tolist())

df['Embedding512'] = formulae_embeddings.tolist()


# print('produce new file')
df.to_pickle('datasets/easysk_train_set.pkl')
