import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_dataset(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    old = np.load
    np.load = lambda *a,**k: old(*a,**k,allow_pickle=True)
    # with np.load('../../tensorflow2.5-graph2gauss/data/'+file_name) as loader:
    with np.load('../../test/'+file_name) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        # X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
        #                    loader['attr_indptr']), shape=loader['attr_shape'])
        X = sp.csr_matrix(loader.get('x_dis'))
        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    print(graph)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj, features

def load_ppi(file_name):
    input_filename = '../../test/datasets/'+file_name+'.txt'
    # read ppi data
    ppi = []
    with open(input_filename, 'r') as f:
        for line in f:
            ppi.append(line.split())

    # determine unique proteins, i.e. nodes
    nodes = list(set([r[0] for r in ppi] + [r[1] for r in ppi]))
    nodes.sort()
    n = len(nodes)

    A = np.identity(n)
    for r in ppi:
        row_index = nodes.index(r[0])
        col_index = nodes.index(r[1])
        if len(r)>2:
            A[row_index][col_index] = float(r[2])
        else:
            A[row_index][col_index] = float(1)
        A[col_index][row_index] = A[row_index][col_index]
    print(A[0])
    X = np.identity(n)
    return sp.csr_matrix(A),sp.csr_matrix(X),nodes

def read_gt(gt_file, nodes):
    with open(('../../test/gold_standard/' + gt_file), 'r') as f:
        gt_raw = []
        for item in f:
            gt_raw.append(item.split())
        # append to "nodes" the nodes not in it but found in gt_nodes
        gt_nodes = set()
        for g in gt_raw:
            gt_nodes.update(g)
        nodes_comm = list(set(nodes) & gt_nodes)
        # form ground truth theta
        gt_theta = np.identity(len(nodes_comm))
        for j in range(len(gt_raw)):
            gt_raw_list = gt_raw[j]
            for k in range(len(gt_raw_list)):
                node1 = gt_raw_list[k]
                if node1 in nodes_comm:
                    index1 = nodes_comm.index(node1)
                    for i in range(k+1,len(gt_raw_list)):
                        node2 = gt_raw_list[i]
                        if node2 in nodes_comm:
                            index2 = nodes_comm.index(node2)
                            gt_theta[index1, index2] = 1.0
                            gt_theta[index2, index1] = 1.0
    return gt_theta,nodes_comm