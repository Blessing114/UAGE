import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
def load_dataset(filename):
    old = np.load
    np.load = lambda *a,**k: old(*a,**k,allow_pickle=True)
    # with np.load('../tensorflow2.5-graph2gauss/data/'+filename) as loader:
    with np.load('../test/'+filename) as loader:
    # with np.load('test_test.npz') as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                        loader['adj_indptr']), shape=loader['adj_shape'])

        # X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
        #                 loader['attr_indptr']), shape=loader['attr_shape'])
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
def score_node_classification(features, z, p_labeled=0.1, n_repeat=10, norm=True):
    lrcv = LogisticRegressionCV()

    if norm:
        features = normalize(features)

    trace = []
    for seed in range(n_repeat):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled, random_state=seed)
        split_train, split_test = next(sss.split(features, z))

        lrcv.fit(features[split_train], z[split_train])
        predicted = lrcv.predict(features[split_test])
        #one-hot multi-class
        labels = list(set(z[split_test]))
        scores = lrcv.predict_proba(features[split_test])
        y_binarize = label_binarize(z[split_test], classes=labels)

        f1_micro = f1_score(z[split_test], predicted, average='micro')
        f1_macro = f1_score(z[split_test], predicted, average='macro')
        precision_micro = precision_score(z[split_test], predicted, average='micro')
        precision_macro = precision_score(z[split_test], predicted, average='macro')
        accuracy= accuracy_score(z[split_test], predicted)
        recall_micro = recall_score(z[split_test], predicted, average='micro')
        recall_macro = recall_score(z[split_test], predicted, average='macro')

        average_precision_micro = average_precision_score(y_binarize, scores, average='micro')
        average_precision_macro = average_precision_score(y_binarize, scores, average='macro')
        roc_auc_micro = roc_auc_score(y_binarize, scores, average='micro', multi_class='ovr')
        roc_auc_macro = roc_auc_score(y_binarize, scores, average='macro', multi_class='ovr')
        
        trace.append((f1_macro, accuracy,precision_macro,recall_macro,average_precision_macro,roc_auc_macro))
    return np.array(trace).mean(0)
filename = 'citeseer-disturbed30.npz'
g = load_dataset(filename)
A, X, z = g['A'], g['X'], g['z']

f1_macro, accuracy,precision_macro,recall_macro,average_precision_macro,roc_auc_macro = score_node_classification(X, z, n_repeat=1, norm=True)
print('accuracy: {:.4f}, f1_macro: {:.4f}'.format(accuracy, f1_macro))
print('recall_macro: {:.4f}, precision_macro: {:.4f}'.format(recall_macro, precision_macro))
print('roc_auc_macro: {:.4f}, average_precision_macro: {:.4f}'.format(roc_auc_macro, average_precision_macro))
# res = pd.DataFrame()
# for i in range(1,10,1):
#     f1_macro, accuracy,precision_macro,recall_macro,average_precision_macro,roc_auc_macro = score_node_classification(X, z, p_labeled = 0.1*i, n_repeat=1, norm=True)
#     if i == 1:
#         print('accuracy: {:.4f}, f1_macro: {:.4f}'.format(accuracy, f1_macro))
#         print('recall_macro: {:.4f}, precision_macro: {:.4f}'.format(recall_macro, precision_macro))
#         print('roc_auc_macro: {:.4f}, average_precision_macro: {:.4f}'.format(roc_auc_macro, average_precision_macro))
#     res = res.append([[f1_macro,average_precision_macro]])
# res.to_csv('res_%s.csv'%(filename.split('/')[-1].split('.')[0]),index=False,header=False)