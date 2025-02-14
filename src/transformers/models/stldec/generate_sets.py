import os
import torch
from torch.nn.functional import normalize
import copy
import numpy as np
import pandas as pd
import stl 
from encoder import STLEncoder

from phis_generator_depth import StlGenerator
from traj_measure import BaseMeasure
from utils import from_string_to_formula, load_pickle, dump_pickle, get_depth
from kernel import StlKernel


encoder = STLEncoder(embed_dim = 1024, anchor_filename = "anchor_set_1024_dim.pickle")
def generate_formulae_depth(n_phis, n_vars, depth):

    # generate formulae of depth equal to (depth + 1) 
    sampler = StlGenerator(max_depth = depth, min_depth = depth)
    sampled_objs = sampler.bag_sample(bag_size = n_phis, nvars = n_vars)

    # check on the formulae depths
    for i in range(len(sampled_objs)):
        assert get_depth(sampled_objs[i]) == (depth+1)

    # converted into strings
    # sampled_formulae = map(str, sampled_objs)
    # encoded_formulae = encoder.compute_embeddings(list(map(str, sampled_formulae)))

    df = pd.DataFrame({'Formula Obj': sampled_objs})
    
    return df

def embed_generated_formulae(df, encoder):
    sampled_formulae = list(map(str, df['Formula Obj'])
    



# depth_3 = generate_formulae_depth(20000, 2, 2)
# depth_3.to_csv('depth_3_formulae.csv')
# depth_4 = generate_formulae_depth(20000, 3, 3)
# depth_4.to_csv('depth_4_formulae.csv')
# depth_5 = generate_formulae_depth(20000, 4, 4)
# depth_5.to_csv('depth_5_formulae.csv')
# depth_6 = generate_formulae_depth(20000, 5, 5)
# depth_6.to_csv('depth_6_formulae.csv')
# depth_7 = generate_formulae_depth(20000, 6, 6)
# depth_7.to_csv('depth_7_formulae.csv')


