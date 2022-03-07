import pandas as pd
import os,json
import argparse
import numpy as np
from numpy import *
import pickle as pkl
from trainNN import train_nn
from evaluation import get_results, roc_thr


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default="../../data/", help="path storing data.")
parser.add_argument('--species', type=str, default="hpo_2019", help="which species to use.")
parser.add_argument('--ppi_attributes', type=int, default=0, help="types of attributes used by ppi.")
parser.add_argument('--model', type=str, default="gcn_vae", help="Model string.")
parser.add_argument('--supervised', type=str, default="nn", help="neural networks or svm")
parser.add_argument('--graph', type=str, default="similarity", help="lists of graphs to use.")
parser.add_argument('--thr_ppi', type=float, default=0.3, help="threshold for combiend ppi network.")
parser.add_argument('--epochs_ppi', type=int, default=80, help="Number of epochs to train ppi.")
parser.add_argument('--save_results', type=int, default=1, help="whether to save the performance results")
args = parser.parse_args()


uniprot = pd.read_pickle(os.path.join(args.data_path, args.species + "/features.pkl"))
uniprot_label = pd.read_pickle(os.path.join(args.data_path, args.species + "/features_label.pkl"))
idx = uniprot[~uniprot["HPO-Term-ID"].isna()].index.tolist()
uniprot_pre = pd.read_pickle(os.path.join(args.data_path, args.species + "/features_ind.pkl"))
uniprot_pre_index = uniprot[uniprot["HPO-Term-ID"].isna()].index.tolist()

hp = uniprot_label['hp_label'].values
hp = np.hstack(hp).reshape((len(hp), len(hp[0])))
hp_t = uniprot_pre['hp_label'].values
hp_t = np.hstack(hp_t).reshape((len(hp_t), len(hp_t[0])))

np.random.seed(5000)
embedding = pd.read_pickle(
    os.path.join(args.data_path, args.species + "/networks/" + args.model +'_'+args.graph+ "_" + str(
        args.ppi_attributes) + "_embeddings_" + args.species + ".pkl"))
embeddings = embedding[idx, :]

X_train = embeddings
Y_train = hp

X_pre = embedding[uniprot_pre_index, :]
Y_pre =  hp_t

y_pre = train_nn(X_train, Y_train, X_pre,Y_pre)


hp_label_list = pd.read_table(os.path.join(args.data_path, args.species + "/hp_list.txt"), header=None)
pd.DataFrame(y_pre).to_csv(
    os.path.join(args.data_path, args.species, args.data_result, "gcn_vae_1_phe_pre.txt"), index=True,
    sep='\t', index_label='id', header=[x[0] for x in hp_label_list.values])
pd.DataFrame(Y_pre).to_csv(
    os.path.join(args.data_path, args.species, args.data_result, "gcn_vae_1_phe_true.txt"), index=True,
    sep='\t', index_label='id', header=[x[0] for x in hp_label_list.values])


# predict, roc_thr_list, roc_auc_list = roc_thr(Y_hp_data,y_hp_data)
predict, roc_thr_list, roc_auc_list = roc_thr(Y_pre, y_pre)
pd.DataFrame(predict).to_csv(
    os.path.join(args.data_path, args.species, args.data_result, "gcn_vae_1_phe_pre01.txt"), index=True,
    sep='\t', index_label='id', header=[x[0] for x in hp_label_list.values])

