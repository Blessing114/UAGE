import networkx as nx
import numpy as np
import pickle
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 

class DBLPDataLoader:
    def __init__(self, graph_file):
        # old = np.load
        # np.load = lambda *a,**k: old(*a,**k,allow_pickle=True)
        # with np.load('../tensorflow2.5-graph2gauss/data/'+graph_file+'.npz') as loader:
        #     loader = dict(loader)
        # A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
        #                loader['adj_indptr']), shape=loader['adj_shape'])
        # z = loader.get('labels')
        # temp = A.todense()
        # edgeList = []
        # x = np.nonzero(temp)
        # for i in range(len(x[0])):
        #     edgeList.append((x[0][i],x[1][i]))
        # G = nx.Graph()   
        # G.add_edges_from(edgeList, weight = 1)

        input_filename = '../test/datasets/'+graph_file+'.txt'
        ppi = []
        G = nx.Graph() 
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
                weight = float(r[2])
            else:
                weight = float(1)
            G.add_edges_from([(row_index,col_index)], weight = weight)

        self.nodes = nodes
        self.g = G
        # self.g = nx.read_gpickle(graph_file)
        print(self.g.__len__())
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

    def fetch_batch(self, batch_size=16, K=10, edge_sampling='atlas', node_sampling='atlas'):
        if edge_sampling == 'numpy':
            edge_batch_index = np.random.choice(self.num_of_edges, size=batch_size, p=self.edge_distribution)
        elif edge_sampling == 'atlas':
            edge_batch_index = self.edge_sampling.sampling(batch_size)
        elif edge_sampling == 'uniform':
            edge_batch_index = np.random.randint(0, self.num_of_edges, size=batch_size)
        u_i = []
        u_j = []
        label = []
        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
            if self.g.__class__ == nx.Graph:
                if np.random.rand() > 0.5:      # important: second-order proximity is for directed edge
                    edge = (edge[1], edge[0])
            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            for i in range(K):
                while True:
                    if node_sampling == 'numpy':
                        negative_node = np.random.choice(self.num_of_nodes, p=self.node_negative_distribution)
                    elif node_sampling == 'atlas':
                        negative_node = self.node_sampling.sampling()
                    elif node_sampling == 'uniform':
                        negative_node = np.random.randint(0, self.num_of_nodes)
                    if not self.g.has_edge(self.node_index_reversed[negative_node], self.node_index_reversed[edge[0]]):
                        break
                u_i.append(edge[0])
                u_j.append(negative_node)
                label.append(-1)
        return u_i, u_j, label
        

    def embedding_mapping(self, embedding):
        return {node: embedding[self.node_index[node]] for node, _ in self.nodes_raw}


class AliasSampling:

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

def read_gt(gt_file, nodes):
    with open(('../test/gold_standard/' + gt_file+'.txt'), 'r') as f:
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

def my_cluster(X,gt_comms,nodes_comm,nodes):
    n_clusters = 30
    x = [0]*len(X.keys())
    for i in X.keys():
        x[int(i)]=X[i]
    X = StandardScaler().fit_transform(x)  #标准化
    cluster=KMeans(n_clusters=n_clusters).fit(X) 
    res = cluster.labels_
    result_mat = np.identity(len(nodes_comm))
    for n in range(n_clusters):
        for i in range(len(res)):
            node1 = nodes[i]
            if res[i]==n and node1 in nodes_comm:
                index1 = nodes_comm.index(node1)
                for j in range(i+1,len(res)):
                        node2 = nodes[j]
                        if res[j]==n and node2 in nodes_comm:
                            index2 = nodes_comm.index(node2)
                            result_mat[index1, index2] = 1.0
                            result_mat[index2, index1] = 1.0
    jc,ri,fmi,F1,MCC = cal_indicators(gt_comms,result_mat,nodes_comm)
    return jc,ri,fmi,F1,MCC

def cal_indicators(gt_comms,result_mat,nodes_comm):
    tp = fp = tn = fn = 0.0
    for i in range(len(nodes_comm)):
        true_set = gt_comms[i][i+1:]
        pre_set = result_mat[i][i+1:]
        temp_tp_fn = np.sum(true_set==1)
        temp_tn_fp = len(true_set)- temp_tp_fn
        temp_fn = 0
        temp_tn = 0
        for j in range(len(nodes_comm)-i-1):
            if pre_set[j] == 0:
                if true_set[j]==1: 
                    temp_fn+=1
                if true_set[j]==0:
                    temp_tn+=1 
        fn+=temp_fn
        tn+=temp_tn
        tp += (temp_tp_fn - temp_fn)
        fp += (temp_tn_fp - temp_tn)
    print(tp+fp+fn+tn)
    print('TP: {:.4f},TN: {:.4f},FP:{:.4f},FN:{:4f}'.format(tp, tn,fp, fn))
    #JC
    JC = tp/(tp+fp+fn)
    #RI
    RI = (tp+tn)/(tp+fp+fn+tn)
    #FMI
    FMI = np.sqrt((tp/(tp+fn))*(tp/(tp+fp)))
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    #F-score
    F1 = 2*(precision*recall)/(precision+recall)
    #MCC
    MCC = (tn*tp-fp*fn)/(np.sqrt((fn+tn)*(fp+tp)*(tn+fp)*(fn+tp)))
    return JC,RI,FMI,F1,MCC