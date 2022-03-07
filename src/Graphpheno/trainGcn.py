from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from optimizer import OptimizerAE, OptimizerVAE
from gcnModel import GCNModelAE, GCNModelVAE, VAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim


def train_gcn(features, adj_train, args, graph_type):
    model_str = args.model

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj_train
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_train)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    # Define placeholders
    placeholders = {
        'features':tf.compat.v1.sparse_placeholder(tf.float64),
        'adj': tf.compat.v1.sparse_placeholder(tf.float64),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float64),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero, args.hidden1, args.hidden2)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero, args.hidden1, args.hidden2)
    elif model_str == 'vae':
        model = VAE(placeholders, num_features, num_nodes, features_nonzero, args.hidden1, args.hidden2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                          validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm,
                          lr=args.lr)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                           validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           lr=args.lr)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)


    # Train model
    # use different epochs for ppi and similarity network
    if graph_type == "similarity":
        epochs = args.epochs_simi
    else:
        epochs = args.epochs_ppi

    for epoch in range(epochs):

        t = time.time()
        # Construct feed dictionary，feed_dict的作用是给使用placeholder创建出来的tensor赋值，feed使用一个值临时替换一个op的输出结果
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: args.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)#执行整个定义好的计算图

        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch+1), "train_loss=", "{:.5f}".format(outs[1]))


    print("Optimization Finished!")


    #return embedding for each protein
    emb = sess.run(model.z_mean,feed_dict=feed_dict)

    return emb


def loss_function(recon_x, x, mu, logvar):
    reconstruction_function = nn.MSELoss()
    MSE = reconstruction_function(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return MSE + KLD


def train_vae(features, adj, args, graph_type):
    n_nodes, feat_dim = features.shape
    scaler = MinMaxScaler()
    scaler.fit(features)
    features_normorlize = scaler.transform(features)
    print(n_nodes, feat_dim)
    features_norm = torch.FloatTensor(features_normorlize)
    features_label = torch.FloatTensor(features)

    # Some preprocessing
    # adj = sp.coo_matrix(adj)
    # adj_ = adj + sp.eye(adj.shape[0])
    # rowsum = np.array(adj_.sum(1))
    # degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # adj_normorlize = torch.FloatTensor(adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).toarray())
    # n_nodes, feat_dim = adj.shape
    # print(n_nodes, feat_dim)
    # adj_label = adj + sp.eye(adj.shape[0])
    # adj_label = adj_label.toarray()
    # adj_label = torch.FloatTensor(adj_label)
    # adj_norm = torch.FloatTensor(adj_normorlize)

    # Some preprocessing

    # features_norm_con = np.concatenate((adj_normorlize, features_normorlize), axis=1)  # concatenate
    # n_nodes, feat_dim = features_norm_con.shape
    # print(n_nodes, feat_dim)
    # features_con_label = np.concatenate((adj_label, features), axis=1)
    #
    # features_norm_con = torch.FloatTensor(features_norm_con)
    # features_con_label = torch.FloatTensor(features_con_label)


    model = VAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    # use different epochs for ppi and similarity network
    if graph_type == "similarity":
        epochs = args.epochs_simi
    else:
        epochs = args.epochs_ppi

    hidden_emb = None
    for epoch in range(epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        reconstructions, z, mu, logvar = model(features_norm)
        loss = loss_function(recon_x=reconstructions, x=features_label,
                             mu=mu, logvar=logvar)
        # reconstructions, z, mu, logvar = model(adj_norm)
        #
        # loss = loss_function(recon_x=reconstructions, x=adj_label,
        #                      mu=mu, logvar=logvar)
        # reconstructions, z, mu, logvar = model(features_norm_con)
        #
        # loss = loss_function(recon_x=reconstructions, x=features_con_label,
        #                      mu=mu, logvar=logvar)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()

        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss, time.time() - t))

    print("Optimization Finished!")

    # return embedding for each protein
    emb = hidden_emb

    return emb

