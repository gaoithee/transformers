import networkx as nx
import numpy as np
import pandas as pd
import stl
from utils2 import from_string_to_formula
from handcoded_tokenizer import STLTokenizer

tokenizer = STLTokenizer('tokenizer_files/tokenizer.json')

def mean_formulae_depth(dataset):
    formulae_depths = []
    for idx in range(len(dataset)):
        object_formula = from_string_to_formula(dataset['Formula'][idx])
        formulae_depths.append(get_depth(object_formula))
    return np.mean(formulae_depths)

def get_depth(formula):
    formula = from_string_to_formula(formula)
    phi_g = build_dag(formula)[0]
    return len(nx.dag_longest_path(phi_g)) - 1

#######################################################################

def get_name_given_type(formula):
    """
    Returns the type of node (as a string) of the top node of the formula/sub-formula
    """
    name_dict = {stl.And: 'and', stl.Or: 'or', stl.Not: 'not', stl.Eventually: 'F', stl.Globally: 'G', stl.Until: 'U',
                 stl.Atom: 'x'}
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
    if type(current_node) is not stl.Atom:
        current_name = get_name_given_type(current_node) + '(' + str(idx) + ')'
        if (type(current_node) is stl.And) or (type(current_node) is stl.Or) or (type(current_node) is stl.Not):
            label_dict[current_name] = [0., 0., 0. ] # temp_left, temp_right, threshold
        else:
            label_dict[current_name] = get_temporal_list(current_node)
        if (type(current_node) is stl.And) or (type(current_node) is stl.Or) or (type(current_node) is stl.Until):
            left_child_name, current_idx = add_internal_child(current_node.left_child, idx + 1, label_dict)
            edges.append([current_name, left_child_name])
            if type(current_node.left_child) is stl.Atom:
                e, d, current_idx = add_leaf_child(current_node.left_child, left_child_name, label_dict, current_idx+1)
                edges += e
                label_dict.update(d)
            e, d = traverse_formula(current_node.left_child, current_idx, label_dict)
            edges += e
            label_dict.update(d)
            right_child_name, current_idx = add_internal_child(current_node.right_child, current_idx + 1, label_dict)
            edges.append([current_name, right_child_name])
            if type(current_node.right_child) is stl.Atom:
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
            if type(current_node.child) is stl.Atom:
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

#######################################################################

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
    return len(tokenizer.encode(str_phi))


def get_n_depth(str_phi):
    phi = from_string_to_formula(str_phi)
    return get_depth(phi)

#######################################################################

study = pd.read_pickle('datasets/hardsk_train_set.pkl')

depths = study['Formula'].apply(get_depth)
tokens = study['Formula'].apply(get_n_tokens)
nodes = study['Formula'].apply(get_n_nodes)
temps = study['Formula'].apply(get_n_temp)
leaves = study['Formula'].apply(get_n_leaves)

descr = pd.DataFrame({
    'Formula': study['Formula'],
    'Depths': depths,
    'Tokens': tokens,
    'Nodes': nodes,
    'Temps': temps,
    'Leaves': leaves
})

descr.to_pickle('descriptive/hardsk_train_set.pkl')
