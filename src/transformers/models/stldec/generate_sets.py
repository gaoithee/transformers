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


encoder = STLEncoder(embed_dim = 1024, anchor_filename = "anchor_set_1024_dim.pickle")
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

formula = generate_formulae_depth(20000, 1, 1)
embedding = embed_generated_formulae(formula)
depth_5 = pd.DataFrame({'Formula': formula, 'Embedding': embedding})
depth_5.to_csv('datasets/depth_2_formulae.csv')

# depth_3 = generate_formulae_depth(20000, 2, 2)
# depth_3['Embedding'] = embed_generated_formulae(depth_3)
# depth_3.to_csv('depth_3_formulae.csv')

# depth_4 = generate_formulae_depth(20000, 3, 3)
# depth_4['Embedding'] = embed_generated_formulae(depth_4)
# depth_4.to_csv('depth_4_formulae.csv')

# depth_5 = generate_formulae_depth(20000, 4, 4)
# depth_5['Embedding'] = embed_generated_formulae(depth_5)
# depth_5.to_csv('depth_5_formulae.csv')
# depth_5 = pd.read_csv('datasets/depth_5_formulae.csv')
# depth_5['Embedding'] = embed_generated_formulae(depth_5)
# depth_5.to_csv('datasets/depth_5_formulae.csv')




# depth_6 = generate_formulae_depth(20, 3, 5)
# depth_6['Embedding'] = embed_generated_formulae(depth_6)
# depth_6.to_csv('test_formulae.csv')

# depth_7 = generate_formulae_depth(20000, 6, 6)
# depth_7['Embedding'] = embed_generated_formulae(depth_7)
# depth_7.to_csv('depth_7_formulae.csv')

