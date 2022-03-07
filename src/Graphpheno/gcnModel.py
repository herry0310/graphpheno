from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf
import os
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F




class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, hidden1, hidden2, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.hidden1_dim = hidden1
        self.hidden2_dim = hidden2
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=self.hidden1_dim,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.embeddings = GraphConvolution(input_dim=self.hidden1_dim,
                                           output_dim=self.hidden2_dim,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)

        self.z_mean = self.embeddings

        self.reconstructions = InnerProductDecoder(input_dim=self.hidden2_dim,
                                        act=lambda x: x,
                                      logging=self.logging)(self.embeddings)


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, hidden1, hidden2, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.hidden1_dim = hidden1
        self.hidden2_dim = hidden2
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=self.hidden1_dim,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              # act=tf.nn.relu,
                                              act=tf.nn.tanh,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim=self.hidden1_dim,
                                       output_dim=self.hidden2_dim,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_log_std = GraphConvolution(input_dim=self.hidden1_dim,
                                          output_dim=self.hidden2_dim,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z = self.z_mean + tf.random_normal([self.n_samples, self.hidden2_dim], dtype=tf.float64) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(input_dim=self.hidden2_dim,
                                        act=lambda x: x,
                                      logging=self.logging)(self.z)


# class VAE(nn.Module):
#     def __init__(self,input_feat_dim,hidden_dim1,hidden_dim2,dropout):
#         super(VAE, self).__init__()
#
#         self.fc1 = nn.Linear(input_feat_dim, hidden_dim1,dropout)
#         self.fc21 = nn.Linear(hidden_dim1, hidden_dim2,dropout)
#         self.fc22 = nn.Linear(hidden_dim1, hidden_dim2,dropout)
#         self.dc = InnerProductDecoder(input_dim=hidden_dim2,
#                                         act=lambda x: x,
#                                       logging=self.logging)(self.z)
#
#     def encoder(self, x):
#         h1 = F.relu(self.fc1(x))
#         mu = self.fc21(h1)
#         logvar = self.fc22(h1)
#         return mu, logvar
#
#     def reparametrize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()   # 计算标准差
#         eps = torch.FloatTensor(std.size()).normal_()
#         eps = Variable(eps)
#         return eps.mul(std).add_(mu)
#
#     def forward(self, x):
#         mu, logvar = self.encoder(x)
#         z = self.reparametrize(mu, logvar)
#         reconstructions = self.dc(z)
#         return reconstructions,z,mu, logvar


class VAE(nn.Module):
    def __init__(self,  features_dim, hidden_dim1, hidden_dim2,  dropout):
        super(VAE, self).__init__()


        self.fc1 = nn.Linear(features_dim, hidden_dim1, dropout)
        self.fc21 = nn.Linear(hidden_dim1, hidden_dim2, dropout)
        self.fc22 = nn.Linear(hidden_dim1, hidden_dim2, dropout)    # 方差
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim1, dropout)
        self.fc4 = nn.Linear(hidden_dim1, features_dim,dropout)

    def encoder(self, inputs):
        h1 = F.relu(self.fc1(inputs))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def decoder(self, z):
        h3 = F.relu(self.fc3(z))
        x = F.tanh(self.fc4(h3))
        return x
    # 重新参数化
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()   # 计算标准差
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, inputs):
        mu, logvar = self.encoder(inputs)
        z = self.reparametrize(mu, logvar)
        reconstructions = self.decoder(z)
        return reconstructions, z, mu, logvar

