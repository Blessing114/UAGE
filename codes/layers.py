# from initializations import *
import tensorflow as tf
import numpy as np
from utils_rd import *

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}
def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs"""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.means = {}
        self.vars = {}
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


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
class GraphConvolution_first(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, support, features_nonzero, dropout=0., act=tf.nn.relu, para_var=1.0, **kwargs):
        super(GraphConvolution_first, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.means['weights'] = weight_variable_glorot(input_dim, output_dim*2, name="weights")
            # self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.output_dim = output_dim
        self.dropout = dropout
        self.features_nonzero = features_nonzero
        self.support = support
        self.act = act
        self.para_var = para_var

    def _call(self, input):
        input = dropout_sparse(input, 1-self.dropout, self.features_nonzero)
        encoded = tf.sparse_tensor_dense_matmul(input, self.means['weights'])
        mean_out = tf.nn.relu(tf.slice(encoded, [0, 0], [-1, self.output_dim]))
        var_out = tf.nn.relu(tf.slice(encoded, [0, self.output_dim], [-1, self.output_dim]))
        node_weight = tf.exp(-var_out*self.para_var)
        mean_out = tf.nn.relu(tf.matmul(self.support[0], mean_out * node_weight))
        var_out = tf.nn.elu(tf.matmul(self.support[1], var_out * node_weight * node_weight)) + 1 + 1e-14
        return mean_out, var_out

class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, support, dropout=0., act=tf.nn.relu, para_var=1.0, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.means['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.input_dim = input_dim
        self.dropout = dropout
        self.support = support
        self.act = act
        self.para_var = para_var

    def _call(self, input):
        x = tf.concat([input[0],input[1]], axis=1)
        # mean_vector, var_vector = input[0],input[1]
        x = tf.nn.dropout(x, 1-self.dropout)
        mean_vector = tf.slice(x, [0, 0], [-1, self.input_dim])
        var_vector = tf.slice(x, [0, self.input_dim], [-1, self.input_dim])
        # mean_vector = tf.nn.dropout(mean_vector, 1-self.dropout)
        # var_vector = tf.nn.dropout(var_vector, 1-self.dropout)
        mean_out = tf.nn.relu(tf.matmul(mean_vector, self.means['weights']))
        var_out = tf.nn.relu(tf.matmul(var_vector, self.vars['weights']))
        node_weight = tf.exp(-var_out*self.para_var)
        mean_out = tf.nn.relu(tf.matmul(self.support[0], mean_out * node_weight))
        var_out = tf.nn.elu(tf.matmul(self.support[1], var_out * node_weight * node_weight)) + 1 + 1e-14
        return mean_out, var_out

# class GraphConvolutionSparse(Layer):
#     """Graph convolution layer for sparse inputs."""
#     def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
#         super(GraphConvolutionSparse, self).__init__(**kwargs)
#         with tf.variable_scope(self.name + '_vars'):
#             self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
#         self.dropout = dropout
#         self.adj = adj
#         self.act = act
#         self.issparse = True
#         self.features_nonzero = features_nonzero

#     def _call(self, inputs):
#         x = inputs
#         x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
#         x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
#         x = tf.sparse_tensor_dense_matmul(self.adj, x)
#         outputs = self.act(x)
#         return outputs

# class GraphConvolution(Layer):
#     """Basic graph convolution layer for undirected graph without edge labels."""
#     def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
#         super(GraphConvolution, self).__init__(**kwargs)
#         with tf.variable_scope(self.name + '_vars'):
#             self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
#         self.dropout = dropout
#         self.adj = adj
#         self.act = act

#     def _call(self, inputs):
#         x = inputs
#         x = tf.nn.dropout(x, 1-self.dropout)
#         x = tf.matmul(x, self.vars['weights'])
#         x = tf.sparse_tensor_dense_matmul(self.adj, x)
#         outputs = self.act(x)
#         return outputs
