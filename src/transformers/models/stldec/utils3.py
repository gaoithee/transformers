import copy
import pickle
import os
from collections import deque

from stl import *


def load_pickle(folder, name):
    with open(folder + os.path.sep + name, 'rb') as f:
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


def rescale_trajectories(traj, min_traj=[0, 0, 0], max_traj=[99, 100, 97]):
    rescaled_traj = torch.zeros_like(traj)
    for i in range(traj.shape[1]):
        new_i = min_traj[i] + (traj[:, i, :] + 1)*(max_traj[i] - min_traj[i])/2
        rescaled_traj[:, i, :] = new_i
    return rescaled_traj


def scale_trajectories(traj):
    traj_min = torch.min(torch.min(traj, dim=0)[0], dim=0)[0]
    traj_max = torch.max(torch.max(traj, dim=0)[0], dim=0)[0]
    scaled_traj = -1 + 2*(traj - traj_min) / (traj_max - traj_min)
    return scaled_traj


def execution_time(start, end, p=False):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    if p:
        print("Execution time = {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))
    return int(hours), int(minutes), int(seconds)


def from_str_to_n_nodes(f):
    f_str = str(f)
    f_split = f_str.split()
    f_nodes_list = [sub_f for sub_f in f_split if sub_f in ['not', 'and', 'or', 'always', 'eventually', '<=', '>=',
                                                            'until']]
    return len(f_nodes_list)


def get_leaves_idx(phi):
    # needed when one only wants to set var thresholds
    phi_str = str(phi)
    phi_split = phi_str.split()
    phi_var = [sub for sub in phi_split if sub.startswith('x_')]
    var_idx = [int(sub[2:]) for sub in phi_var]
    return len(phi_var), var_idx

