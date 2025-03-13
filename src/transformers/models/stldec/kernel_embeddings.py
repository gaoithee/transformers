import copy
from sklearn.decomposition import KernelPCA

import stl
from kernel import StlKernel
from traj_measure import BaseMeasure
# from utils import get_leaves_idx
# from phiselect import filter_time_depth


def get_leaves_idx(phi):
    # needed when one only wants to set var thresholds
    phi_str = str(phi)
    phi_split = phi_str.split()
    phi_var = [sub for sub in phi_split if sub.startswith('x_')]
    var_idx = [int(sub[2:]) for sub in phi_var]
    return len(phi_var), var_idx
    
def filter_time_depth(phi_list, max_time_depth):
    def check_time_depth(phi): return phi.time_depth() < max_time_depth
    phi_ok = list(filter(check_time_depth, phi_list))
    return phi_ok


def rescale_var_thresholds(formula, mu_list, std_list):
    # DFS traversal of the tree
    current_node = formula
    if type(current_node) is not stl.Atom:
        if (type(current_node) is stl.And) or (type(current_node) is stl.Or) or (type(current_node) is stl.Until):
            left_child = rescale_var_thresholds(current_node.left_child, mu_list, std_list)
            current_node.left_child = copy.deepcopy(left_child)
        else:
            if (type(current_node) is stl.Eventually) or (type(current_node) is stl.Globally):
                child = rescale_var_thresholds(current_node.child, mu_list, std_list)
                current_node.child = copy.deepcopy(child)
        if (type(current_node) is stl.And) or (type(current_node) is stl.Or) or (type(current_node) is stl.Until):
            right_child = rescale_var_thresholds(current_node.right_child, mu_list, std_list)
            current_node.right_child = copy.deepcopy(right_child)
    else:
        current_node.threshold = round((current_node.threshold * std_list[current_node.var_index]) +
                                       mu_list[current_node.var_index], 4)
    return current_node


class StlKernelEmbeddings:
    def __init__(self, traj_dataset, all_phis, device, pca=None):
        self.device = device
        self.trajectories = traj_dataset.trajectories
        self.nvars = traj_dataset.nvars
        self.timespan = traj_dataset.npoints
        self.phis = self.filter_phis(all_phis)
        # if None plain embeddings, else oca is the number of retained dimensions
        self.pca = pca
        self.embeddings = None  # maybe useless
        self.mean, self.std = [copy.deepcopy(traj_dataset.mean), copy.deepcopy(traj_dataset.std)]
        if traj_dataset.normalized is False:
            traj_dataset.normalize()
            self.mean, self.std = [traj_dataset.mean, traj_dataset.std]
            traj_dataset.inverse_normalize()
        self.normalized = False

    def filter_phis(self, phi_list):
        def get_max_var_idx(phi): max(get_leaves_idx(phi)[1])
        def filter_max_var(phi): return get_max_var_idx(phi) < self.nvars
        var_filtered_phis = list(filter(filter_max_var, phi_list))
        return filter_time_depth(var_filtered_phis, self.timespan)

    def get_kernel_embedding(self, train_phis):
        # assuming train phis are fixed and have the right number of variables
        mu0 = BaseMeasure(self.device, sigma0=1., sigma1=1., q=0.1)
        kernel = StlKernel(mu0, sigma2=0.44, varn=self.nvars)
        if self.pca is None:
            kernel_embeddings = kernel.compute_bag_bag(self.phis, train_phis)
        else:
            gram_train = kernel.compute_bag_bag(train_phis, train_phis)
            gram_phis = kernel.compute_bag_bag(self.phis, train_phis)
            kpca = KernelPCA(n_components=self.pca, kernel='precomputed')
            kpca.fit(gram_train)
            kernel_embeddings = kpca.transform(gram_phis)
        self.embeddings = kernel_embeddings
        return kernel_embeddings

    def rescale_thresholds(self, phis):
        def current_rescale(phi): return rescale_var_thresholds(phi, self.mean, self.std)
        self.normalized = True
        return list(map(current_rescale, phis))
