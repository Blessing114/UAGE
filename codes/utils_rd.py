import os
import sys
import numpy as np
import tensorflow as tf
import logging
import numpy as np
import scipy.sparse as sp

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import normalize
import math
import networkx as nx


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger

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
    with np.load(file_name) as loader:
        loader = dict(loader)
        # print(loader.keys())
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])
        P = sp.csr_matrix((loader['adj_data_weighted'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])
        # if mode=='0':
        # X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
        #                 loader['attr_indptr']), shape=loader['attr_shape'])
        # else:
        X = sp.csr_matrix(loader.get('x_dis'))

        z = loader.get('labels')

        graph = {
            'A': A,
            'P': P,
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


def Cni(n, i):
    return math.factorial(n)//(math.factorial(i)*math.factorial(n-i))

def P_power(P, i):
    result = np.identity(P.shape[0])
    if i == 0:
        return result
    else:
        result = P
    for j in range(1,i):
        result=np.matmul(result,P)
    return result

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
    
# ===========================================todo====================================
def compute_PSM(P,c=0.6,t=20):
    # 论文DGCU: A new deep directed method based on Gaussian embedding for clustering uncertain graphs，公式（10）
    # 公式与论文SimRank*: effective and scalable pairwise similarity search based on graph topology有所出入，需要核实修改
    PSM = 0.0
    for i in range(t+1):
        mid=0.0
        for j in range(i+1):
            mid+=Cni(i,j)*np.matmul(P_power(P,j),P_power(P.T,i-j))
        PSM+=math.pow(c,i)/math.pow(2,i)*mid
    PSM=(1-c)*PSM
    return PSM

class Data_sampling:
    def __init__(self, PSM, is_all=True):

        self.A = sp.csr_matrix(PSM)
        # if not is_all and 'val_edges' in loader.keys():
        #     self.val_edges = loader['val_edges']
        #     self.val_ground_truth = loader['val_ground_truth']
        #     self.test_edges = loader['test_edges']
        #     self.test_ground_truth = loader['test_ground_truth']

        self.g = nx.from_scipy_sparse_matrix(self.A)

        self.num_of_nodes = self.g.number_of_nodes()
        self.num_of_edges = self.g.number_of_edges()
        self.edges_raw = self.g.edges(data=True)
        self.nodes_raw = self.g.nodes(data=True)

        self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.edges_raw], dtype=np.float32)
        self.edge_distribution /= np.sum(self.edge_distribution)
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)
        self.node_negative_distribution = np.power(
            np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw], dtype=np.float32), 0.75)
        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

        self.node_index = {}
        self.node_index_reversed = {}
        for index, (node, _) in enumerate(self.nodes_raw):
            self.node_index[node] = index
            self.node_index_reversed[index] = node
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v, _ in self.edges_raw]

    def fetch_next_batch(self, batch_size=16, K=10):
        edge_batch_index = self.edge_sampling.sampling(batch_size)
        u_i = []
        u_j = []
        label = []
        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
            if self.g.__class__ == nx.Graph:
                if np.random.rand() > 0.5:
                    edge = (edge[1], edge[0])
            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            for i in range(K):
                while True:
                    negative_node = self.node_sampling.sampling()
                    if not self.g.has_edge(self.node_index_reversed[negative_node], self.node_index_reversed[edge[0]]):
                        break
                u_i.append(edge[0])
                u_j.append(negative_node)
                label.append(-1)
        u_k = list(np.random.randint(0, self.num_of_nodes, batch_size))
        return u_i, u_j, u_k, label

    # def embedding_mapping(self, embedding):
    #     return {node: embedding[self.node_index[node]] for node, _ in self.nodes_raw}

class AliasSampling:
    # Reference: LINE source code from https://github.com/snowkylin/line
    # Reference: https://en.wikipedia.org/wiki/Alias_method
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res
            
def Wasserstein(mu, sigma, idx1, idx2):
    delta_mu = tf.gather(mu,idx1) - tf.gather(mu,idx2)
    p1 = tf.reduce_sum(delta_mu * delta_mu, axis=1)
    delta_sigma = tf.pow(tf.gather(sigma,idx1),1/2) - tf.pow(tf.gather(sigma,idx2),1/2)
    p2 = tf.reduce_sum(delta_sigma * delta_sigma, axis=1)
    return p1+p2

def energy_kl(mu, sigma, idx1, idx2, embedding_dim):
    mu_i = tf.gather(mu, idx1)
    sigma_i = tf.gather(sigma, idx1)
    mu_j = tf.gather(mu, idx2)
    sigma_j = tf.gather(sigma, idx2)

    sigma_ratio = sigma_j / sigma_i
    trace_fac = tf.reduce_sum(sigma_ratio, 1)
    log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), 1)

    mu_diff_sq = tf.reduce_sum(tf.square(mu_i - mu_j) / sigma_i, 1)
    ij_kl = 0.5 * (trace_fac + mu_diff_sq - embedding_dim - log_det)

    sigma_ratio = sigma_i / sigma_j
    trace_fac = tf.reduce_sum(sigma_ratio, 1)
    log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), 1)

    mu_diff_sq = tf.reduce_sum(tf.square(mu_j - mu_i) / sigma_j, 1)
    ji_kl = 0.5 * (trace_fac + mu_diff_sq - embedding_dim - log_det)
    kl_distance = 0.5 * (ij_kl + ji_kl)
    return kl_distance

def kl(x, y):
    X = tf.distributions.Categorical(probs=x)
    Y = tf.distributions.Categorical(probs=y)
    return tf.distributions.kl_divergence(X, Y)

def count_trainable_vars():
    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters
 
    print("Total number of trainable parameters-----------------------------------------: %d" % total_parameters)
    
# def energy_kl(self, u_i, u_j, proximity):
#     mu_i = tf.gather(self.embedding, u_i)
#     sigma_i = tf.gather(self.sigma, u_i)

#     if proximity == 'first-order':
#         mu_j = tf.gather(self.embedding, u_j)
#         sigma_j = tf.gather(self.sigma, u_j)
#     elif proximity == 'second-order':
#         mu_j = tf.gather(self.ctx_mu, u_j)
#         sigma_j = tf.gather(self.ctx_sigma, u_j)

#     sigma_ratio = sigma_j / sigma_i
#     trace_fac = tf.reduce_sum(sigma_ratio, 1)
#     log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), 1)

#     mu_diff_sq = tf.reduce_sum(tf.square(mu_i - mu_j) / sigma_i, 1)

#     ij_kl = 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

#     sigma_ratio = sigma_i / sigma_j
#     trace_fac = tf.reduce_sum(sigma_ratio, 1)
#     log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), 1)

#     mu_diff_sq = tf.reduce_sum(tf.square(mu_j - mu_i) / sigma_j, 1)

#     ji_kl = 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

#     kl_distance = 0.5 * (ij_kl + ji_kl)

#     return kl_distance

def one_hot(y_):
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    features = tf.sparse.to_dense(tf.sparse.reorder(features), default_value=0)
    rowsum = tf.reduce_sum(features,axis=1)
    d_inv_sqrt = tf.pow(rowsum, -1)
    d_mat_inv_sqrt = tf.diag(d_inv_sqrt)
    return tf.matmul(d_mat_inv_sqrt,features)   

def normalize_adj(adj, alpha):
    """Symmetrically normalize adjacency matrix."""
    rowsum = tf.reduce_sum(adj,axis=1)
    d_inv_sqrt = tf.pow(rowsum, alpha)
    d_mat_inv_sqrt = tf.diag(d_inv_sqrt)
    return tf.matmul(tf.matmul(d_mat_inv_sqrt,adj),d_mat_inv_sqrt)

def preprocess_adj(adj, alpha):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(tf.sparse.to_dense(tf.sparse.reorder(adj), default_value=0) + tf.diag(tf.ones([adj.shape[0]])), alpha)
    return adj_normalized

# ======================节点分类函数，后期可以考虑将不确定性加入sample weight中===============
def score_node_classification(mu, sigma, z, p_labeled=0.1, n_repeat=1, norm=True):
    """
    Train a classifier using the node embeddings as features and reports the performance.

    Parameters
    ----------
    mu,sigma : array-like, shape [N, L]
        The features used to train the classifier, i.e. the node embeddings
    z : array-like, shape [N]
        The ground truth labels
    p_labeled : float
        Percentage of nodes to use for training the classifier
    n_repeat : int
        Number of times to repeat the experiment
    norm

    Returns
    -------
    f1_micro: float
        F_1 Score (micro) averaged of n_repeat trials.
    f1_micro : float
        F_1 Score (macro) averaged of n_repeat trials.
    """
    # features = mu + tf.random_normal(mu.shape) * tf.sqrt(sigma)
    # with tf.Session() as sess:
    #     features = sess.run(features)
    features = mu
    lrcv = LogisticRegressionCV(max_iter=10000)

    if norm:
        features = normalize(features, axis=1)

    trace = []
    for seed in range(n_repeat):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled, random_state=seed)
        sigma_mean = sigma.mean(axis=1)
        sigma_exp = np.exp(-1*sigma_mean)
        split_train, split_test = next(sss.split(features, z, sigma_exp))
        # lrcv.fit(features[split_train], z[split_train],sample_weight=sigma_exp[split_train])
        
        lrcv.fit(features[split_train], z[split_train])
        
        predicted = lrcv.predict(features[split_test])
        scores = lrcv.predict_proba(features[split_test])
        acc = accuracy_score(z[split_test], predicted)
        f1_micro = f1_score(z[split_test], predicted, average='micro')
        f1_macro = f1_score(z[split_test], predicted, average='macro')

        auc = roc_auc_score(one_hot(z[split_test]), scores)
        auprc = average_precision_score(one_hot(z[split_test]), scores)
        prec_macro = precision_score(z[split_test], predicted, average='macro', )
        prec_micro = precision_score(z[split_test], predicted, average='micro', )
        recall_macro = recall_score(z[split_test], predicted, average='macro', )
        recall_micro = recall_score(z[split_test], predicted, average='micro', )
        trace.append((f1_micro, f1_macro, prec_micro, prec_macro, recall_micro, recall_macro,auc,auprc, acc))

    return np.array(trace).mean(0)
















def random_split(n=11988, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Use 8:1:1 split-random split"""
    p_train = train_ratio
    p_val = val_ratio
    p_test = test_ratio

    n_train = round(n * p_train)
    n_val = round(n * p_val)
    n_test = n - (n_train + n_val)
    p = np.random.permutation(n)
    idx_train = p[:n_train]
    idx_val = p[n_train:n_train + n_val]
    idx_test = p[n_train + n_val:]
    return idx_train, idx_val, idx_test

def random_sample(idx_0, idx_1, B, replace=False):
    """ Returns a balanced sample of tensors by randomly sampling without replacement. """
    idx0_batch = np.random.choice(idx_0, size=int(B / 2), replace=replace)
    idx1_batch = np.random.choice(idx_1, size=int(B / 2), replace=replace)
    idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
    return idx


