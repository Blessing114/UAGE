import tensorflow as tf
import numpy as np
import argparse
from model import LINEModel
from utils import DBLPDataLoader,read_gt,my_cluster
import pickle
import time
import os
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES']  = '4'# import tensorflow as tf
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,accuracy_score
from sklearn.preprocessing import normalize
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
    x = [0]*len(features.keys())
    for i in features.keys():
        x[int(i)]=features[i]
    if norm:
        features = normalize(x)
    trace = []
    for seed in range(n_repeat):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled, random_state=seed)
        split_train, split_test = next(sss.split(features, z))

        lrcv.fit(features[split_train], z[split_train])
        predicted = lrcv.predict(features[split_test])
        scores = lrcv.predict_proba(features[split_test])
        #one-hot multi-class
        labels = list(set(z[split_test]))
        # predicted_binarize = label_binarize(predicted, classes=labels)
        y_binarize = label_binarize(z[split_test], classes=labels)
        # f1_micro = f1_score(z[split_test], predicted, average='micro')
        f1_macro = f1_score(z[split_test], predicted, average='macro')
        # precision_micro = precision_score(z[split_test], predicted, average='micro')
        precision_macro = precision_score(z[split_test], predicted, average='macro')
        accuracy= accuracy_score(z[split_test], predicted)
        # recall_micro = recall_score(z[split_test], predicted, average='micro')
        recall_macro = recall_score(z[split_test], predicted, average='macro')

        average_precision_macro = average_precision_score(y_binarize, scores, average='macro')
        # roc_auc_micro = roc_auc_score(y_binarize, scores, average='micro', multi_class='ovr')
        roc_auc_macro = roc_auc_score(y_binarize, scores, average='macro', multi_class='ovr')
        
        
        trace.append((f1_macro, accuracy,precision_macro,recall_macro,average_precision_macro,roc_auc_macro))

    return np.array(trace).mean(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', default=64)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--K', default=5)
    parser.add_argument('--proximity', default='first-order', help='first-order or second-order')
    parser.add_argument('--learning_rate', default=0.025)
    parser.add_argument('--mode', default='train')

    parser.add_argument('--num_batches', default=50000)
    parser.add_argument('--total_graph', default=True)
    parser.add_argument('--graph_file', default='cora_ml-disturbed10')
    parser.add_argument('--ppiName', default='krogan2006_extended')
    parser.add_argument('--gt_dataset', default='sgd')
    args = parser.parse_args()
    data_loader = DBLPDataLoader(graph_file=args.ppiName)
    if args.mode == 'train':
        train(args,data_loader)
    elif args.mode == 'test':
        test(args)
    result=pickle.load(open('embedding_%s_%s.pkl' % (args.ppiName, args.proximity), 'rb'))
    gt_comms,nodes_comm = read_gt(args.gt_dataset, data_loader.nodes)
    print(len(nodes_comm))
    JC, RI, FMI, F1, MCC = my_cluster(result,gt_comms,nodes_comm,data_loader.nodes)
    print('JC: {:.4f}, RI: {:.4f},FMI:{:.4f}'.format(JC, RI,FMI))
    print('F-score: {:.4f}, MCC: {:.4f}'.format(F1, MCC))
    # f1_macro, accuracy,precision_macro,recall_macro,average_precision_macro,roc_auc_macro = score_node_classification(result, data_loader.label, n_repeat=1, norm=True)
    # print('accuracy: {:.4f}, f1_macro: {:.4f}'.format(accuracy, f1_macro))
    # print('recall_macro: {:.4f}, precision_macro: {:.4f}'.format(recall_macro, precision_macro))
    # print('roc_auc_macro: {:.4f}, average_precision_macro: {:.4f}'.format(roc_auc_macro, average_precision_macro))
    # res = pd.DataFrame()
    # for i in range(1,10,1):
    #     f1_macro, accuracy,precision_macro,recall_macro,average_precision_macro,roc_auc_macro = score_node_classification(result, data_loader.label, p_labeled = 0.1*i, n_repeat=1, norm=True)
    #     if i == 1:
    #         print('accuracy: {:.4f}, f1_macro: {:.4f}'.format(accuracy, f1_macro))
    #         print('recall_macro: {:.4f}, precision_macro: {:.4f}'.format(recall_macro, precision_macro))
    #         print('roc_auc_macro: {:.4f}, average_precision_macro: {:.4f}'.format(roc_auc_macro, average_precision_macro))
    #     res = res.append([[f1_macro,average_precision_macro]])
    # res.to_csv('res_%s.csv'%(args.graph_file),index=False,header=False)

def train(args,data_loader):
    suffix = args.proximity
    args.num_of_nodes = data_loader.num_of_nodes
    model = LINEModel(args)
    with tf.Session() as sess:
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        for b in range(args.num_batches):
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            t2 = time.time()
            sampling_time += t2 - t1
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - b / args.num_batches)
                else:
                    learning_rate = args.learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0
            if b % 1000 == 0 or b == (args.num_batches - 1):
                embedding = sess.run(model.embedding)
                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                            open('embedding_%s_%s.pkl' % (args.ppiName, args.proximity), 'wb'))


def test(args):
    pass

if __name__ == '__main__':
    main()