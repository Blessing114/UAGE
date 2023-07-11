import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
import pickle
from cProfile import label
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import argparse
import pickle
# from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score, precision_score, recall_score, f1_score
from models_rd import *
from utils_rd import *

def load_ppi(file_name):
    input_filename = file_name
    # read ppi data
    ppi = []
    with open(input_filename, 'r') as f:
        for line in f:
            ppi.append(line.split())

    # determine unique proteins, i.e. nodes
    nodes = list(set([r[0] for r in ppi] + [r[1] for r in ppi]))
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
    with open( gt_file, 'r') as f:
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
    print('TP: {:.4f},TN: {:.4f},FP:{:.4f},FN:{:4f}'.format(tp, tn,fp, fn))
    print(tp+fp+fn+tn)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='biogrid_yeast_physical_unweighted', choices=['collins2007','biogrid_yeast_physical_unweighted','gavin2006_socioaffinities_rescaled' 'krogan2006_core', 'krogan2006_extended']) #
    parser.add_argument('--model', default='Our-model', help='Ours 1,2,3...') #
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.1)  #loss function 0.1
    parser.add_argument('--batch_size', type=int, default=128)  #128
    parser.add_argument('--num_batches', type=int, default=10000) #10000
    parser.add_argument('--t', type=int, default=1) #compute PSM
    parser.add_argument('--K', type=int, default=10)   #5
    parser.add_argument('--para_var', default=1.0, help='Parameter of variance-based attention') 
    parser.add_argument('--learning_rate', default=0.001)  #0.001
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--distance', default="Wasserstein", choices=['Wasserstein', 'KL-divergence'], help="the distance bettween two distributions")

    parser.add_argument('--is_all', default=True)  # train with all edges; no validation or test set
    parser.add_argument('--p_labeled',default=0.5, type=float, help='Percentage of nodes to use for training the classifier') 
    args, unknown = parser.parse_known_args()
    # args, unknown = parser.parse_args()

    model_path = '../models/'
    log_dir = '../logs'
    _logger = get_logger(log_dir, __name__, str(args.dataset)+"_"+str(args.t)+'-result.log', level='INFO')

    dataset = args.dataset
    _logger.info('Dataset used: {}'.format(dataset))
    train(args,_logger)


def train(args,_logger):
    Train=True
    if Train:
        graph_file='../../Data/%s.txt' % args.dataset
        gt_dataset = '../../Data/sgd.txt'

        args.A, args.X,nodes = load_ppi(graph_file)
        gt_comms,nodes_comm = read_gt(gt_dataset, nodes)
        print(len(nodes_comm))
        
        args.features_nonzero = sparse_feeder(args.X)[1].shape[0]
        #row-normalize feature
        # x_feature = tf.SparseTensor(*sparse_feeder(args.X))
        # args.X = preprocess_features(x_feature)
        # 邻接矩阵归一化
        P = tf.SparseTensor(*sparse_feeder(args.A))
        # with tf.Session() as sess:
        #     pp=sess.run(P)
        args.support=[]
        args.support.append(preprocess_adj(P, -0.5))
        args.support.append(preprocess_adj(P, -1.0))

        # =============计算相似矩阵==================  
        if not os.path.exists('../PSM/%s%s_PSM_%s.pkl' % (args.dataset, '_all' if args.is_all else '', str(args.t))):
            with tf.compat.v1.Session() as sess:
                a=tf.sparse.to_dense(tf.sparse.reorder(P), default_value=0)
                P = sess.run(a)
            PSM = compute_PSM(P,c=0.6,t=args.t)  #self.n_sample * self.n_sample
            pickle.dump(PSM, open('../PSM/%s%s_PSM_%s.pkl' % (args.dataset, '_all' if args.is_all else '', str(args.t)), 'wb'))
        else:
            PSM=pickle.load(open('../PSM/%s%s_PSM_%s.pkl' % (args.dataset, '_all' if args.is_all else '', str(args.t)), 'rb'))
        Data_sample = Data_sampling(PSM, args.is_all)
        # =============计算相似矩阵==================


        initial_tolerance = 200
        early_stopping_score_max = -1.0
        tolerance = initial_tolerance
        

        if args.model == "Our-model":
            model = Ourmodel(args)
            count_trainable_vars()

            with tf.Session() as sess:
                tf.compat.v1.global_variables_initializer().run()

                sampling_time, training_time = 0, 0

                for b in range(args.num_batches):
                    t1=time.time()
                    u_i, u_j, u_k, label = Data_sample.fetch_next_batch(batch_size=args.batch_size, K=args.K)
                    feed_dict = {model.u_i: u_i, model.u_j: u_j, model.u_k: u_k, model.label: label}
                    t2=time.time()
                    _,loss,L_s,L_reg,distance = sess.run([model.train_op,model.loss,model.L_s,model.L_reg,model.distance], feed_dict=feed_dict)
                    mu, sigma = sess.run([model.mu, model.sigma])
                    t3=time.time()
                    sampling_time = sampling_time+ t2-t1
                    training_time = training_time+t3-t2
                    # print("=================",b)
                    early_stopping_score = -loss
                    if b % 50 == 0:
                        _logger.info('batches: {}\tloss: {:.4f}\tsampling_time: {:.2f}\ttraining_time: {:.2f}'.format (b, loss, sampling_time, training_time))
                        sampling_time,training_time = 0, 0
                    if early_stopping_score > early_stopping_score_max:
                        early_stopping_score_max = early_stopping_score
                        tolerance = initial_tolerance
                    # if b % 50 == 0 or b == (args.num_batches - 1):
                        mu, sigma = sess.run([model.mu, model.sigma])
                        # print(mu,sigma)
                        pickle.dump({'mu': mu,
                                    'sigma': sigma},
                                    open('../emb/%s_%s%s_embedding.pkl' % (args.dataset, str(args.t),'_all' if args.is_all else ''), 'wb'))
                    else:
                        tolerance -= 1
                    
                    if tolerance == 0:
                        mu, sigma = sess.run([model.mu, model.sigma])
                        pickle.dump({'mu': mu,
                                    'sigma': sigma},
                                    open('../emb/%s_%s%s_embedding.pkl' % (args.dataset, str(args.t), '_all' if args.is_all else ''), 'wb'))
                        break

                if tolerance > 0:
                    _logger.info('The model has not been converged.. Exit due to number of batches..')

                result=pickle.load(open('../emb/%s_%s%s_embedding.pkl' % (args.dataset, str(args.t), '_all' if args.is_all else ''), 'rb'))   #####读取embedding
                JC, RI, FMI, F1, MCC = my_cluster(result['mu'],gt_comms,nodes_comm,nodes)
                print('JC: {:.4f}, RI: {:.4f},FMI:{:.4f},F-score: {:.4f}, MCC: {:.4f}'.format(JC, RI,FMI,F1, MCC))
                _logger.info('JC: {:.4f}, RI: {:.4f},FMI:{:.4f},F-score: {:.4f}, MCC: {:.4f}'.format(JC, RI,FMI,F1, MCC))
    else:
        graph_file='../../Data/%s.txt' % args.dataset
        gt_dataset = '../../Data/sgd.txt'
        args.A, args.X,nodes = load_ppi(graph_file)
        gt_comms,nodes_comm = read_gt(gt_dataset, nodes)
        
        result=pickle.load(open('../emb/%s_%s%s_embedding.pkl' % (args.dataset, str(args.t), '_all' if args.is_all else ''), 'rb'))   #####读取embedding
        JC, RI, FMI, F1, MCC = my_cluster(result['mu'],gt_comms,nodes_comm,nodes)
        print('JC: {:.4f}, RI: {:.4f},FMI:{:.4f},F-score: {:.4f}, MCC: {:.4f}'.format(JC, RI,FMI,F1, MCC))
        _logger.info('JC: {:.4f}, RI: {:.4f},FMI:{:.4f},F-score: {:.4f}, MCC: {:.4f}'.format(JC, RI,FMI,F1, MCC))
if __name__ == '__main__':
    main()