#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.generation import get_best_candidates, gen_candidates_torch
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import gen_batch_initial_conditions
import gpytorch
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import copy
import numpy as np
from sklearn.model_selection import KFold
import itertools
import warnings

from trajectories import TrajectoryDataset, get_dataset
from utils import from_string_to_formula, get_leaves_idx, from_str_to_n_nodes, dump_pickle, execution_time
from phis_generator import StlGenerator
from phisearch import get_embeddings, search_from_embeddings
from kernel_embeddings import rescale_var_thresholds


warnings.filterwarnings("ignore")


def get_pos_neg_robustness(phi, traj_data):
    # this requires that positive examples are labeled as 1
    pos_idx = np.where(traj_data.labels.cpu().numpy() == 1)[0].tolist()
    neg_idx = np.where(traj_data.labels.cpu().numpy() != 1)[0].tolist()
    traj_pos = traj_data.trajectories[pos_idx, :, :]
    traj_neg = traj_data.trajectories[neg_idx, :, :]
    phi = traj_data.time_scaling(phi)
    rob_pos = phi.quantitative(traj_pos, normalize=True)
    rob_neg = phi.quantitative(traj_neg, normalize=True)
    return rob_pos, rob_neg


# we want to fit a model of this objective function
def objective(phi, traj_data):  # this is the objective to maximize
    if type(phi) is str:
        phi = from_string_to_formula(phi)
    rob_pos, rob_neg = get_pos_neg_robustness(phi, traj_data)
    exp_pos, std_pos = [torch.mean(rob_pos), torch.std(rob_pos)]
    exp_neg, std_neg = [torch.mean(rob_neg), torch.std(rob_neg)]
    return torch.div(exp_pos - exp_neg, std_pos + std_neg)

# TODO: need to adapt get_embeddings
'''
def get_embeddings(folder_index, max_n_vars, device, phis, n_pc=-1):
    train_phis = load_pickle(folder_index, 'train_phis_{}_vars.pickle'.format(max_n_vars))
    mu0 = BaseMeasure(device=device, sigma0=1.0, sigma1=1.0, q=0.1)
    kernel = StlKernel(mu0, varn=max_n_vars, sigma2=0.44, samples=10000)
    gram_phis = kernel.compute_bag_bag(phis, train_phis)
    embeddings = gram_phis
    if n_pc != -1:
        pca_matrix = torch.from_numpy(load_pickle(folder_index, 'pca_proj_{}_vars.pickle'.format(
            max_n_vars))).to(device)
        embeddings = torch.matmul(gram_phis.to(device).float(), pca_matrix[:, :n_pc].float())
    return embeddings
'''


def generate_initial_data(traj_data, folder_index, max_n_vars, device, n_pc=-1, n=10, seed=None):
    sampler_phis = StlGenerator(leaf_prob=0.5, seed=seed)
    init_phis = []
    while len(init_phis) < n:
        sampled_phis = list(
            filter(lambda p: from_str_to_n_nodes(p) <= 2, sampler_phis.bag_sample(n, nvars=traj_data.nvars)))
        init_phis += copy.deepcopy(sampled_phis)
    init_phis = init_phis[:n]
    init_x = get_embeddings(folder_index, max_n_vars, device, init_phis, n_pc=n_pc)
    def current_obj(p): return objective(p, traj_data)
    init_y = torch.tensor(list(map(current_obj, init_phis)))
    return init_phis, init_x.to(device), init_y.to(device)


# TODO: need to adapt custom search
def custom_search(candidate_embeddings, traj_data, folder_index, k, neigh_n, device, max_n_vars,
                  n_pc=-1, timespan=None, nodes=None):
    retrieved_str = search_from_embeddings(candidate_embeddings, traj_data.nvars, folder_index, k, neigh_n, device,
                                           n_pc=n_pc, timespan=timespan, nodes=nodes)[0]
    retrieved = [list(map(from_string_to_formula, sub_l)) for sub_l in retrieved_str]
    new_f = []
    for results in retrieved:
        results_valid_idx = list(filter(lambda i: max(get_leaves_idx(results[i])[1]) < traj_data.nvars,
                                        list(range(len(results)))))
        results_valid = [results[i] for i in results_valid_idx]
        new_f += results_valid
    new_emb = get_embeddings(folder_index, max_n_vars, device, new_f, n_pc=n_pc)
    # TODO: should we return only phis whose actual embedding is close to that searched?
    # TODO: e.g. with some sort of filtering on distances computed by the index?
    # TODO: for this we need a threshold on euclidean distances vs similarity
    # TODO: it can depend on other search parameters such as nprobe
    # TODO: what if none of the results is below the threshold?
    return new_emb.to(device), new_f


def get_fitted_gp(train_x, train_y, state_d=None, covar=None):
    gp = SingleTaskGP(train_X=train_x, train_Y=train_y, covar_module=covar)
    if state_d is not None:
        gp.load_state_dict(state_d)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    mll.to(train_x)
    fit_gpytorch_model(mll)
    return gp


def get_candidates_from_acquisition(acq_f, bound, npts, nbatches, optim):
    batch_init = gen_batch_initial_conditions(acq_function=acq_f, bounds=bound, q=npts, num_restarts=nbatches,
                                              raw_samples=100)

    batch_candidates, batch_acq_values = gen_candidates_torch(
        initial_conditions=batch_init, acquisition_function=acq_f, lower_bounds=bound[0], upper_bounds=bound[1],
        optimizer=optim, options={"maxiter": 500})
    # candidates dimension is [n batches, n points per batch q, dimension emb]
    # best candidates is a batch of points (best among previously generated)
    best_candidates = get_best_candidates(batch_candidates, batch_acq_values)
    return best_candidates.detach()


def get_binary_performance(phi, traj_data, refine=False):
    alpha = 0.
    if refine:
        phi, alpha = refine_phi(phi, traj_data)
    rob_on_pos, rob_on_neg = get_pos_neg_robustness(phi, traj_data)
    true_pos = sum([int(rob >= 0) for rob in rob_on_pos])
    false_neg = len(rob_on_pos) - true_pos
    false_pos = sum([int(rob >= 0) for rob in rob_on_neg])
    true_neg = len(rob_on_neg) - false_pos
    accuracy = (true_pos + true_neg)/traj_data.trajectories.shape[0]
    misclassification = (false_pos + false_neg)/traj_data.trajectories.shape[0]
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos != 0 else 0.
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg != 0 else 0.
    return np.round(accuracy, 4), np.round(precision, 4), np.round(recall, 4), np.round(misclassification, 4), alpha


def refine_phi(phi, traj_data, n_alphas=50):
    def current_rescale(a): return copy.deepcopy(rescale_var_thresholds(phi, [a]*traj_data.nvars, [1.]*traj_data.nvars))
    def current_accuracy(p): return get_binary_performance(p, traj_data, refine=False)[0]
    rob_on_pos, rob_on_neg = get_pos_neg_robustness(phi, traj_data)
    alphas = torch.linspace(torch.mean(rob_on_neg).item(), torch.mean(rob_on_pos).item(), n_alphas).tolist()
    phi_alphas = list(map(current_rescale, alphas))
    phi_acc = torch.tensor(list(map(current_accuracy, phi_alphas)))
    return phi_alphas[phi_acc.argmax()], alphas[phi_acc.argmax()]


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on {device}".format(device=dev))
# import dataset of trajectories
dataset = TrajectoryDataset(data_fn=get_dataset, dataname='maritime', indexes=None, device=dev)
dataset.normalize()
# set up for search engine
base_folder = os.getcwd()
index_folder = base_folder + os.path.sep + 'index'
maxvars = 3
topk = 5
n_neigh = 64
# set up for bayes opt
# RBFKernel()  LinearKernel() MaternKernel()
gp_covar = gpytorch.kernels.MaternKernel()
# bayes opt loop
npc = -1  # [25, 50, 100, 250, 500] -1
bounds_dim = 1000 if npc == -1 else npc
bounds = torch.stack([torch.zeros(bounds_dim), torch.ones(bounds_dim)]).to(dev)
n_init_points = 100
n_batches_acq = 1  # 100
n_pts_batch = 25  # 1
n_tried_batches = 1
n_seeds = 10
best_acc = 0.
best_config = {}
for _ in range(n_seeds):
    current_seed = np.random.randint(low=1, high=666)
    np.random.seed(current_seed)
    kfold = KFold(n_splits=5, random_state=current_seed, shuffle=True)
    current_phis, current_scaled_phis, current_alphas, current_acc, current_mcr = [[] for _ in range(5)]
    for (train_index, test_index) in kfold.split(dataset.trajectories):
        train_dataset = TrajectoryDataset(dev, x=dataset.trajectories[train_index], y=dataset.labels[train_index])
        train_dataset.normalize()
        def current_objective(p): return objective(p, train_dataset)
        test_dataset = TrajectoryDataset(dev, x=dataset.trajectories[test_index], y=dataset.labels[test_index])
        test_dataset.normalize()
        phis, x, y = generate_initial_data(train_dataset, index_folder, maxvars, dev, n_pc=npc, n=n_init_points,
                                           seed=current_seed)
        start_optim = time.time()
        emulator = get_fitted_gp(x, y.unsqueeze(-1), state_d=None, covar=gp_covar)
        # possible acquisition functions
        ei = LogExpectedImprovement(emulator, y.max())
        ucb = UpperConfidenceBound(emulator, beta=0.1)
        # bayesian optimization loop
        used_phis, used_x, used_y = [copy.deepcopy(item) for item in [phis, x, y]]
        for new_batch in range(n_batches_acq):
            sampler = SobolQMCNormalSampler(torch.Size([bounds_dim]), seed=current_seed)
            qEI = qLogExpectedImprovement(emulator, used_y.max(), sampler=sampler)  # log too
            qUCB = qUpperConfidenceBound(emulator, beta=0.2, sampler=sampler)
            # torch.optim.SGD
            candidates_x = get_candidates_from_acquisition(qUCB, bounds, n_pts_batch, n_tried_batches, torch.optim.Adam)
            candidates_x, new_phis = custom_search(candidates_x, train_dataset, index_folder, topk, n_neigh, dev,
                                                   maxvars, n_pc=npc, timespan=None, nodes=None)
            new_obj = torch.tensor(list(map(current_objective, new_phis))).to(dev)
            used_phis += copy.deepcopy(new_phis)
            used_x = torch.cat([used_x, copy.deepcopy(candidates_x)], dim=0).to(dev)
            used_y = torch.cat([used_y, copy.deepcopy(new_obj)]).to(dev)
            emulator = get_fitted_gp(used_x, used_y.unsqueeze(-1), state_d=emulator.state_dict(), covar=gp_covar)
            if new_batch == n_batches_acq - 1:
                current_best_idx = torch.argmax(used_y).item()
                best_phi = used_phis[current_best_idx]
                acc, current_prec, current_rec, mcr, al = \
                    get_binary_performance(best_phi, test_dataset, refine=True)
                print('Current seed: ', current_seed)
                print('Accuracy: ', np.round(acc, 4), 'Precision: ', np.round(current_prec, 4), 'Recall: ',
                      np.round(current_rec, 4), 'Misclassification Rate: ', np.round(mcr, 4))
                print('Best formula: ', str(best_phi), 'obj value: ', used_y[current_best_idx].item())
                best_temporal_scaled = test_dataset.time_scaling(best_phi)
                best_scaled = \
                    rescale_var_thresholds(best_temporal_scaled, test_dataset.mean.tolist(), test_dataset.std.tolist())
                best_scaled_alpha = \
                    rescale_var_thresholds(best_scaled, (-al*test_dataset.std).tolist(), [1.]*test_dataset.nvars)
                print('Best formula with scaled parameters: ', str(best_scaled_alpha))
                current_phis.append(used_phis[current_best_idx])
                current_scaled_phis.append(best_scaled_alpha)
                current_alphas.append(al)
                current_acc.append(acc)
                current_mcr.append(mcr)
        end_optim = time.time()
        execution_time(start_optim, end_optim, p=True)
        del used_x
        del used_y
    if torch.median(torch.tensor(current_acc)).item() >= best_acc:
        best_config = {'seed': current_seed, 'phis': current_phis, 'scaled phis': current_scaled_phis,
                       'alphas': current_alphas, 'accuracy': current_acc, 'mcr': current_mcr}
        best_acc = torch.median(torch.tensor(current_acc)).item()
        # dump_pickle(os.getcwd() + os.path.sep + 'results' + os.path.sep + 'naval_results_kpca', best_config)
