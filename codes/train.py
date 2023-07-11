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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmed-disturbed30', choices=['cora_ml','cora_ml-disturbed10' 'cora', 'citeseer', 'dblp', 'pubmed']) #
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
        graph_file='../../Data/%s.npz' % args.dataset
        g = load_dataset(graph_file)
        args.A, args.X, args.z, args.P = g['A'], g['X'], g['z'],g['P']   #邻接矩阵A;节点属性X;节点标签z
    
        # args.A, args.X, args.z = g['A'], g['X'], g['z']   #邻接矩阵A;节点属性X;节点标签z
        args.features_nonzero = sparse_feeder(args.X)[1].shape[0]
        #row-normalize feature
        # x_feature = tf.SparseTensor(*sparse_feeder(args.X))
        # args.X = preprocess_features(x_feature)
        # 邻接矩阵归一化
        P = tf.SparseTensor(*sparse_feeder(args.P))
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

        z = tf.SparseTensor(*sparse_feeder(args.z))
        with tf.Session() as sess:
            z = sess.run(tf.sparse_tensor_to_dense(tf.sparse.reorder(z), default_value=0))

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

                result=pickle.load(open('../emb/%s_%s%s_embedding.pkl' % (args.dataset, str(args.t),'_all' if args.is_all else ''), 'rb'))
                f1_micro, f1_macro, prec_micro, prec_macro, recall_micro, recall_macro,auc,auprc, acc = score_node_classification(result["mu"], result["sigma"], z.reshape(-1,1).ravel(), p_labeled=args.p_labeled, n_repeat=10, norm=True)
                _logger.info('f1_micro: {:.4f}, f1_macro: {:.4f},\n prec_micro: {:.4f}, prec_macro: {:.4f},\n recall_micro: {:.4f}, recall_macro: {:.4f},\n auc: {:.4f},auprc: {:.4f},acc: {:.4f}'.format(f1_micro, f1_macro, prec_micro, prec_macro, recall_micro, recall_macro,auc,auprc,acc))
    else:
        graph_file='../../Data/%s.npz' % args.dataset
        g = load_dataset(graph_file)
        args.z = g['z']   #节点标签z
        z = tf.SparseTensor(*sparse_feeder(args.z))
        with tf.Session() as sess:
            z = sess.run(tf.sparse.to_dense(tf.sparse.reorder(z), default_value=0))
        result=pickle.load(open('../emb/%s_%s%s_embedding.pkl' % (args.dataset, str(args.t), '_all' if args.is_all else ''), 'rb'))
        f1_micro, f1_macro, prec_micro, prec_macro, recall_micro, recall_macro,auc,auprc, acc = score_node_classification(result["mu"], result["sigma"], z.reshape(-1,1).ravel(), p_labeled=args.p_labeled, n_repeat=10, norm=True)
        _logger.info('f1_micro: {:.4f}, f1_macro: {:.4f},\n prec_micro: {:.4f}, prec_macro: {:.4f},\n recall_micro: {:.4f}, recall_macro: {:.4f},\n auc: {:.4f},auprc: {:.4f},acc: {:.4f}'.format(f1_micro, f1_macro, prec_micro, prec_macro, recall_micro, recall_macro,auc,auprc,acc))
if __name__ == '__main__':
    main()
