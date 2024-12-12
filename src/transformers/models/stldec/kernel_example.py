import os
import torch
from torch.nn.functional import normalize
import copy
import numpy as np

from phis_generator import StlGenerator
from traj_measure import BaseMeasure
from utils import from_string_to_formula, load_pickle, dump_pickle
from kernel import StlKernel

# compute kernel embeddings
# 1. Fix an anchor set of STL formulae, against which you compute all embeddings
# We randomly sample it using the following:
# leaf_prob determines the syntactic complexity of formulae
sampler = StlGenerator(leaf_prob=0.4)
# the number of generated formulae determines the dimension of your embeddings
n_phis = 3000
# the number of variables determines the maximum allowed dimension of signals
n_vars = 3
anchor_set = sampler.bag_sample(bag_size=n_phis, nvars=n_vars) # list of formulae
# IMPORTANT: the anchor set should never change within the same application, better to store it in memory!
# dump_pickle('anchor_set_{}_vars'.format(n_vars), anchor_set)
# to recover it
# anchor_set = load_pickle(os.getcwd() + os.path.sep + 'anchor_set.pkl')
# if it is too heavy you can keep the string in memory
anchor_set_string = list(map(str, anchor_set))
anchor_set_from_string = list(map(from_string_to_formula, anchor_set_string))
# 2. Instantiate the kernel
# need the measure on the space of trajectories over which to integrate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mu = BaseMeasure(device=device)
# instantiate the kernel given the measure and the number of variables (should be the same of the anchor set)
kernel = StlKernel(mu, varn=n_vars)
# 3. Embed random STL formulae
phis = sampler.bag_sample(1000, nvars=n_vars)  # in principle, you can use a sampler with different parameters
# the rows of the following are the embeddings of the corresponding formula
gram_phis = kernel.compute_bag_bag(phis, anchor_set)


# CAVEAT: you can sample a semantically diverse set of formulae as follows
# just-copy paste the following in place of the previous plain anchor set sampling
cos_threshold = 0.9 # lower it if it struggles to fin enough formulae
diverse_anchor_set = [sampler.sample(nvars=n_vars)]
signals = mu.sample(samples=10000, varn=n_vars)
# always use normalized robustness with mu0
anchor_rob_vectors = torch.cat(
    [phi.quantitative(signals, normalize=True).unsqueeze(0) for phi in diverse_anchor_set],0)
while len(diverse_anchor_set) < n_phis:
    # sample candidate anchor formulae
    candidate_anchors = sampler.bag_sample(n_phis - len(diverse_anchor_set), nvars=n_vars)
    # compute robustness of candidate anchor formulae on the same signals as previous anchor set
    candidate_robs = torch.cat(
        [phi.quantitative(signals, normalize=True).unsqueeze(0) for phi in candidate_anchors],0)
    # compute cosine similarity between current anchor set and candidate new formulae
    cos_simil = torch.tril(normalize(candidate_robs) @ normalize(anchor_rob_vectors).t(), diagonal=-1)
    # check which formulae are similar (i.e. greater cosine similarity then threshold) w.r.t. current anchors
    similar_idx = [np.where(cos_simil[r, :] > cos_threshold)[0].tolist() for r in range(cos_simil.shape[0])]
    # keep only those who are semantically distant
    keep_idx = list(set(np.arange(len(candidate_anchors)).tolist()).difference(set(
        [i for sublist in similar_idx for i in sublist])))
    diverse_anchor_set += [copy.deepcopy(candidate_anchors[i]) for i in keep_idx]
    anchor_rob_vectors = torch.cat([anchor_rob_vectors, copy.deepcopy(
        torch.index_select(candidate_robs, 0, torch.tensor(keep_idx)))], 0)
anchor_set = diverse_anchor_set[:n_phis]
