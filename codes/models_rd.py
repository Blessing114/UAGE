import numpy as np
import tensorflow as tf
import scipy.sparse as sp
# import os
# import pickle

from utils_rd import *
from layers import *

seed = 42
def sparse_feeder(M):
    M = sp.coo_matrix(M, dtype=np.float32)
    return np.vstack((M.row, M.col)).T, M.data, M.shape

def sparse_gather(indices, values, selected_indices, axis=0):
    mask = tf.equal(indices[:, axis][tf.newaxis, :], selected_indices[:, tf.newaxis])
    to_select = tf.where(mask)[:, 1]
    return tf.gather(indices, to_select, axis=0), tf.gather(values, to_select, axis=0)

class Ourmodel:
    def __init__(self, args):
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)
        self.features_nonzero = args.features_nonzero

        self.X = tf.SparseTensor(*sparse_feeder(args.X))
        # zero = tf.constant(0, dtype=tf.float32)
        # where = tf.not_equal(args.X, zero)
        # indices = tf.where(where)
        # values = tf.gather_nd(args.X, indices)
        # self.X = tf.SparseTensor(indices, values, args.X.shape)
        # self.A = tf.SparseTensor(*sparse_feeder(args.A))
        self.support = args.support

        self.n_samples, self.input_dim = args.X.shape
        # self.n_samples, self.input_dim = args.X.shape[0].value,args.X.shape[1].value
        self.embedding_dim = args.embedding_dim
        self.para_var = args.para_var
        self.alpha = args.alpha
        self.dropout = args.dropout
        self.n_hidden =  512

        self.u_i = tf.compat.v1.placeholder(name='u_i', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.u_j = tf.compat.v1.placeholder(name='u_j', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.u_k = tf.compat.v1.placeholder(name='u_k', dtype=tf.int32, shape=[None])
        self.label = tf.compat.v1.placeholder(name='label', dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])

        self.__build_model()

        if not args.is_all:
            self.val_edges = args.val_edges
            self.val_ground_truth = args.val_ground_truth
            self.neg_val_energy = -self.energy_kl(self.val_edges[:, 0], self.val_edges[:, 1], args.proximity)
            self.val_set = True
        else:
            self.val_set = False

        # =============loss function============
        # attributes loss
        X_hat = tf.gather(self.X_hat, self.u_k)  # decoded node attribute vectors
        slice_indices, slice_values = sparse_gather(self.X.indices, self.X.values, tf.cast(self.u_k, tf.int64))
        X = tf.gather(tf.sparse.to_dense(tf.SparseTensor(slice_indices, slice_values,
                                                         tf.cast(tf.shape(self.X), tf.int64)), validate_indices=False),
                      self.u_k)  # original node attribute vectors
        var_out = tf.gather(self.sigma, self.u_k)
        var_out_mean = tf.reduce_mean(var_out,axis=1)
        node_weight = tf.exp(-var_out_mean * 0.05) #0.05
        L_a = tf.reduce_mean(node_weight * tf.reduce_mean(tf.square(tf.subtract(X, X_hat)),axis=1))  # Euclidean distance
        # L_a = tf.reduce_mean(tf.square(tf.subtract(X, X_hat)))  # Euclidean distance
        # attributes loss

        # structure loss
        if args.distance == "Wasserstein":
            self.distance = Wasserstein(self.mu, self.sigma, self.u_i, self.u_j)
        elif args.distance == "KL-divergence":
            self.distance = energy_kl(self.mu, self.sigma, self.u_i, self.u_j,self.embedding_dim)
            # tf.log(tf.clip_by_value(cvr, 1e-10, 0.9999))
        L_s = -tf.reduce_mean(tf.math.log_sigmoid(-self.label * self.distance))
        # structure loss

        L_reg = tf.reduce_mean(-0.5 * tf.reduce_sum(1 - self.sigma - tf.square(self.mu) +tf.math.log(self.sigma+1e-8), axis=1))
        

        # l1-regularizer loss
        # l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005)
        # L_1 = tf.reduce_mean(tf.contrib.layers.apply_regularization(l1_regularizer, self.trained_variables))
        # l1-regularizer loss

        # self.loss = L_s + self.alpha * L_a
        self.L_s = L_s
        self.L_a = L_a
        self.L_reg = L_reg
        self.loss = L_s + self.alpha * L_a + 1e-5 * L_reg
        # self.loss = L_s + 1e-3 * L_reg
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def __build_model(self):
        w_init = tf.contrib.layers.xavier_initializer

        # ==============================encoder2========================================
        # sizes = [self.input_dim]
        # sizes.append(self.n_hidden)
        # trained_variables = []
        
        # for i in range(1, len(sizes)):
        #     W_mu = tf.compat.v1.get_variable(name='W_mu_encoder{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float32, initializer=w_init())
        #     b_mu = tf.compat.v1.get_variable(name='b__mu_encoder{}'.format(i), shape=[sizes[i]], dtype=tf.float32, initializer=w_init())
        #     trained_variables.extend([W_mu])
        #     trained_variables.extend([b_mu])
        #     W_sigma = tf.compat.v1.get_variable(name='W_sigma_encoder{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float32, initializer=w_init())
        #     b_sigma = tf.compat.v1.get_variable(name='b__sigma_encoder{}'.format(i), shape=[sizes[i]], dtype=tf.float32, initializer=w_init())
        #     trained_variables.extend([W_sigma])
        #     trained_variables.extend([b_sigma])
        #     if i == 1:
        #         self.mu = tf.nn.relu(tf.sparse_tensor_dense_matmul(self.X, W_mu) + b_mu)
        #         self.sigma = tf.nn.elu(tf.sparse_tensor_dense_matmul(self.X, W_sigma) + b_sigma) + 1 + 1e-14
        #     else:
        #         self.mu = tf.nn.relu(tf.sparse_tensor_dense_matmul(self.mu, W_mu) + b_mu)
        #         self.sigma = tf.nn.elu(tf.sparse_tensor_dense_matmul(self.sigma, W_sigma) + b_sigma) + 1 + 1e-14

        # self.mu, self.sigma= GraphConvolution(input_dim=sizes[-1],
        #                                         output_dim=self.embedding_dim,
        #                                         support = self.support,
        #                                         act=tf.nn.elu,
        #                                         para_var=self.para_var,
        #                                         dropout=self.dropout)([self.mu, self.sigma])
        # ==============================encoder2========================================
        
        # ==============================encoder1========================================
        sizes = [self.input_dim]
        sizes.append(self.n_hidden)
        trained_variables = []
        for i in range(1, len(sizes)):
            W = tf.compat.v1.get_variable(name='W_encoder{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float32, initializer=w_init())
            b = tf.compat.v1.get_variable(name='b_encoder{}'.format(i), shape=[sizes[i]], dtype=tf.float32, initializer=w_init())
            trained_variables.extend([W])
            trained_variables.extend([b])
            if i == 1:
                encoded = tf.sparse_tensor_dense_matmul(self.X, W) + b
            else:
                encoded = tf.matmul(encoded, W) + b
            encoded = tf.nn.relu(encoded)

        W_mu = tf.compat.v1.get_variable(name='W_mu', shape=[sizes[-1], self.embedding_dim], dtype=tf.float32, initializer=w_init())
        b_mu = tf.compat.v1.get_variable(name='b_mu', shape=[self.embedding_dim], dtype=tf.float32, initializer=w_init())
        self.mu = tf.matmul(encoded, W_mu) + b_mu

        W_sigma = tf.compat.v1.get_variable(name='W_sigma', shape=[sizes[-1], self.embedding_dim], dtype=tf.float32, initializer=w_init())
        b_sigma = tf.compat.v1.get_variable(name='b_sigma', shape=[self.embedding_dim], dtype=tf.float32, initializer=w_init())
        log_sigma = tf.matmul(encoded, W_sigma) + b_sigma
        self.sigma = tf.nn.elu(log_sigma) + 1 + 1e-14

        trained_variables.extend([W_mu])
        trained_variables.extend([b_mu])
        # ==============================encoder1========================================


        # self.embedding = self.mu + tf.random_normal([self.n_samples, sizes[-1]]) * tf.exp(self.sigma)  #tf.exp?
        # self.embedding = self.mu + tf.random_normal([self.n_samples, self.embedding_dim],0,1) * tf.sqrt(self.sigma+1e-8)
        # ==============================decoder1========================================
        self.embedding = self.mu
        sizes.append(self.embedding_dim)   #encoder1 add
        for i in range(len(sizes)-1,0,-1):
            W = tf.compat.v1.get_variable(name='W_decoder{}'.format(i), shape=[sizes[i], sizes[i-1]], dtype=tf.float32, initializer=w_init())
            b = tf.compat.v1.get_variable(name='b_decoder{}'.format(i), shape=[sizes[i-1]], dtype=tf.float32, initializer=w_init())
            trained_variables.extend([W])
            trained_variables.extend([b])

            if i == len(sizes)-1:
                decoded = tf.matmul(self.embedding, W) + b
            else:
                decoded = tf.matmul(decoded, W) + b
            if i==1:
                decoded = tf.nn.elu(decoded)
            else:
                decoded = tf.nn.relu(decoded)
        self.X_hat = decoded

        self.trained_variables = trained_variables
        # ==============================decoder1========================================




