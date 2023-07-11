import os
os.environ['CUDA_VISIBLE_DEVICES']  = '3'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import scipy.sparse as sp
from model import Graph2Gauss
from util_sub import load_dataset, load_ppi,score_link_prediction, score_node_classification, read_gt, my_cluster
import pickle
import pandas as pd
import numpy as np

fileName = '../../test/pubmed-disturbed10.npz'
# fileName = '../data/cora_ml.npz'
# g = load_dataset(fileName)
# # g = load_dataset('../../test/cora_ml.npz')
# A, X, z = g['A'], g['X'], g['z']
ppiName = 'collins2007.txt'
gt_dataset = 'sgd.txt'
gt_comms,nodes_gt = read_gt(gt_dataset)
A, X,nodes = load_ppi(ppiName,nodes_gt)

# 链接预测
# g2g = Graph2Gauss(A=A, X=X, L=64, verbose=True, p_val=0.10, p_test=0.05, p_nodes=0)
# sess = g2g.train()
# test_auc, test_ap = score_link_prediction(g2g.test_ground_truth, sess.run(g2g.neg_test_energy))
# print('test_auc: {:.4f}, test_ap: {:.4f}'.format(test_auc, test_ap))

#分类
# g2g = Graph2Gauss(A=A, X=X, L=64, max_iter = 10000,verbose=True, p_val=0.0, p_test=0.00)
# sess = g2g.train()
# mu, sigma = sess.run([g2g.mu, g2g.sigma])
# pickle.dump({'mu': mu, 'sigma':sigma },
#             open('g2g_test_embedding_%s'%(ppiName.split('/')[-1].split('.')[0]), 'wb'))
# print(mu.shape)

result=pickle.load(open('g2g_test_embedding_%s'%(ppiName.split('/')[-1].split('.')[0]), 'rb'))
JC, RI, FMI, F1, MCC = my_cluster(result['mu'],gt_comms,nodes,nodes_gt)
print('JC: {:.4f}, RI: {:.4f},FMI:{:.4f}'.format(JC, RI,FMI))
print('F-score: {:.4f}, MCC: {:.4f}'.format(F1, MCC))

# f1_macro, accuracy,precision_macro,recall_macro,average_precision_macro,roc_auc_macro = score_node_classification(result['mu'], z, n_repeat=1, norm=True)
# print('accuracy: {:.4f}, f1_macro: {:.4f}'.format(accuracy, f1_macro))
# print('recall_macro: {:.4f}, precision_macro: {:.4f}'.format(recall_macro, precision_macro))
# print('roc_auc_macro: {:.4f}, average_precision_macro: {:.4f}'.format(roc_auc_macro, average_precision_macro))

# res = pd.DataFrame()
# for i in range(1,10,1):
#     f1_macro, accuracy,precision_macro,recall_macro,average_precision_macro,roc_auc_macro = score_node_classification(result['mu'], z, p_labeled = 0.1*i, n_repeat=1, norm=True)
#     res = res.append([[f1_macro,average_precision_macro]])
# res.to_csv('res_%s.csv'%(fileName.split('/')[-1].split('.')[0]),index=False,header=False)
# # 无属性
# g2g = Graph2Gauss(A=A, X=A+sp.eye(A.shape[0]), L=64, verbose=True, p_val=0.0, p_test=0.00)
# sess = g2g.train()
# mu, sigma = sess.run([g2g.mu, g2g.sigma])
# f1_micro, f1_macro = score_node_classification(mu, z, n_repeat=1, norm=True)
# print('f1_micro: {:.4f}, f1_macro: {:.4f}'.format(f1_micro, f1_macro))