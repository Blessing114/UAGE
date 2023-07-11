from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "5"

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,accuracy_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import label_binarize,StandardScaler

from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data,load_ppi,read_gt
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
import pickle
import pandas as pd
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 5000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'pubmed-disturbed30', 'Dataset string.')
flags.DEFINE_string('ppiName', 'krogan2006_extended.txt', 'ppi Dataset Name.')
flags.DEFINE_string('gt_dataset', 'sgd.txt', 'ppi groudtruth file.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

def score_node_classification(features, z, p_labeled=0.1, n_repeat=10, norm=True):
    """
    Train a classifier using the node embeddings as features and reports the performance.

    Parameters
    ----------
    features : array-like, shape [N, L]
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
    lrcv = LogisticRegressionCV()

    if norm:
        features = normalize(features)

    trace = []
    for seed in range(n_repeat):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled, random_state=seed)
        split_train, split_test = next(sss.split(features, z))

        lrcv.fit(features[split_train], z[split_train])
        predicted = lrcv.predict(features[split_test])
        scores = lrcv.predict_proba(features[split_test])
        #one-hot multi-class
        labels = list(set(z[split_test]))
        y_binarize = label_binarize(z[split_test], classes=labels)
        # predicted_binarize = label_binarize(predicted, classes=labels)

        # f1_micro = f1_score(z[split_test], predicted, average='micro')
        f1_macro = f1_score(z[split_test], predicted, average='macro')
        # precision_micro = precision_score(z[split_test], predicted, average='micro')
        precision_macro = precision_score(z[split_test], predicted, average='macro')
        accuracy= accuracy_score(z[split_test], predicted)
        # recall_micro = recall_score(z[split_test], predicted, average='micro')
        recall_macro = recall_score(z[split_test], predicted, average='macro')

        # average_precision_micro = average_precision_score(y_binarize, scores, average='micro')
        average_precision_macro = average_precision_score(y_binarize, scores, average='macro')
        # roc_auc_micro = roc_auc_score(y_binarize, scores, average='micro', multi_class='ovr')
        roc_auc_macro = roc_auc_score(y_binarize, scores, average='macro', multi_class='ovr')
        
        trace.append((f1_macro, accuracy,precision_macro,recall_macro,average_precision_macro,roc_auc_macro))

    return np.array(trace).mean(0)
def my_cluster(x,gt_comms,nodes_comm,nodes):
    n_clusters = 30
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

def train (adj,features):
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    tf.disable_eager_execution()
    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                        validate_indices=False), [-1]),
                            pos_weight=pos_weight,
                            norm=norm)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                        validate_indices=False), [-1]),
                            model=model, num_nodes=num_nodes,
                            pos_weight=pos_weight,
                            norm=norm)

    def get_roc_score(edges_pos, edges_neg, emb=None):
        if emb is None:
            feed_dict.update({placeholders['dropout']: 0})
            emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    cost_val = []
    acc_val = []

    val_roc_score = []

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    early_stopping_score_max = -float('inf')
    tolerance = 100
    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy,opt.mu,opt.sigma], feed_dict=feed_dict)
        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
        early_stopping_score = avg_accuracy + ap_curr
        val_roc_score.append(roc_curr)
        if early_stopping_score > early_stopping_score_max:
            early_stopping_score_max = early_stopping_score
            tolerance = 100
        else:
            tolerance -= 1
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
            "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
            "val_ap=", "{:.5f}".format(ap_curr),
            "time=", "{:.5f}".format(time.time() - t))
        if tolerance == 0:
            break
    pickle.dump({'mu': outs[3], 'sigma':outs[4] },
                open('vgae_embedding_%s'%(dataset_str), 'wb'))
    print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))

model_str = FLAGS.model
dataset_str = FLAGS.ppiName.split('/')[-1].split('.')[0]
A, X,nodes = load_ppi(dataset_str)
# g = load_dataset(dataset_str)
# a, x, z = g['A'], g['X'], g['z']
# adj, features = load_data(dataset_str)

train(A,X)
result=pickle.load(open('vgae_embedding_%s'%(dataset_str), 'rb'))
print(len(nodes))
gt_comms,nodes_comm = read_gt(FLAGS.gt_dataset, nodes)
print(len(nodes_comm))
JC, RI, FMI, F1, MCC = my_cluster(result['mu'],gt_comms,nodes_comm,nodes)
print('JC: {:.4f}, RI: {:.4f},FMI:{:.4f}'.format(JC, RI,FMI))
print('F-score: {:.4f}, MCC: {:.4f}'.format(F1, MCC))
# f1_macro, accuracy,precision_macro,recall_macro,average_precision_macro,roc_auc_macro = score_node_classification(result['mu'], z, n_repeat=1, norm=True)
# print('accuracy: {:.4f}, f1_macro: {:.4f}'.format(accuracy, f1_macro))
# print('recall_macro: {:.4f}, precision_macro: {:.4f}'.format(recall_macro, precision_macro))
# print('roc_auc_macro: {:.4f}, average_precision_macro: {:.4f}'.format(roc_auc_macro, average_precision_macro))
# res = pd.DataFrame()
# for i in range(1,10,1):
#     f1_macro, accuracy,precision_macro,recall_macro,average_precision_macro,roc_auc_macro = score_node_classification(result['mu'], z, p_labeled = 0.1*i, n_repeat=1, norm=True)
#     if i == 1:
#         print('accuracy: {:.4f}, f1_macro: {:.4f}'.format(accuracy, f1_macro))
#         print('recall_macro: {:.4f}, precision_macro: {:.4f}'.format(recall_macro, precision_macro))
#         print('roc_auc_macro: {:.4f}, average_precision_macro: {:.4f}'.format(roc_auc_macro, average_precision_macro))
#     res = res.append([[f1_macro,average_precision_macro]])
# res.to_csv('res_%s.csv'%(dataset_str),index=False,header=False)