import pandas as pd
import os,json
import argparse
import numpy as np
from trainNN import train_nn
from sklearn.model_selection import KFold
from evaluation import get_results, roc_thr


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default="../../data/", help="path storing data.")
parser.add_argument('--species', type=str, default="hpo_2021", help="which species to use.")
parser.add_argument('--ppi_attributes', type=int, default=1, help="types of attributes used by ppi.")
parser.add_argument('--model', type=str, default="gcn_vae", help="Model string.")
parser.add_argument('--supervised', type=str, default="nn", help="neural networks or svm")
parser.add_argument('--graph', type=str, default="similarity", help="lists of graphs to use.")
parser.add_argument('--thr_ppi', type=float, default=0.3, help="threshold for combiend ppi network.")
parser.add_argument('--epochs_ppi', type=int, default=80, help="Number of epochs to train ppi.")
parser.add_argument('--save_results', type=int, default=1, help="whether to save the performance results")
args = parser.parse_args()


uniprot = pd.read_pickle(os.path.join(args.data_path, args.species + "/features.pkl"))
uniprot_label = pd.read_pickle(os.path.join(args.data_path, args.species + "/test1/features_label.pkl"))
idx = uniprot[~uniprot["HPO-Term-ID"].isna()].index.tolist()

hp = uniprot_label['hp_label'].values
hp = np.hstack(hp).reshape((len(hp), len(hp[0])))
hp_t = uniprot['hp_label'].values
test_hp =  np.hstack(hp_t).reshape((len(hp_t), len(hp_t[0])))


np.random.seed(5000)
# vgae embedding
embedding = pd.read_pickle(
    os.path.join(args.data_path, args.species + "/networks/" + args.model + "_" + str(args.lr) + "_" + str(
        args.thr_ppi) + "_" + args.graph + "_" + str(args.ppi_attributes) + "_embeddings_" + args.species + "_" + str(
        args.epochs_ppi) + ".pkl"))
embeddings = embedding[idx, :]
all_idx = list(range(hp.shape[0]))
np.random.shuffle(all_idx)

X_hp_data = embeddings[all_idx]
kf = KFold(n_splits=5)
y_score_hp_list = []
Y_test_hp_list = []
n = 0
for train, test in kf.split(X_hp_data):
    Y_train_hp = hp[train]
    Y_test_hp = hp[test]
    X_train = embeddings[train]
    X_test = embeddings[test]
    n = n + 1
    # Y_val_cell = cell[val_idx]
    y_score_hp = train_nn(X_train, Y_train_hp, X_test, Y_test_hp)
    y_score_hp_list.append(y_score_hp)
    Y_test_hp_list.append(Y_test_hp)
    y_hp_data = np.vstack(y_score_hp_list)
    Y_hp_data = np.vstack(Y_test_hp_list)
predict, roc_thr_list, roc_auc_list = roc_thr(Y_hp_data.values, y_hp_data.values)
perf = dict()
perf['roc_auc'] = [x[0] for x in roc_auc_list]
perf['roc_thr'] = [x[0] for x in roc_thr_list]
f = pd.DataFrame.from_dict(perf, orient='index', columns=None)
f = f.reset_index().rename(columns={'index': 'evaluation index'})

train_embedding = embedding[idx, :]
test_embedding = embedding

#预测打分
y_pre = train_nn(train_embedding, hp, test_embedding, test_hp)
for i in range( y_pre.shape[1]):
    y_pre[:, i][y_pre[:, i] >= perf['thr'][i]] = 1
    y_pre[:, i][y_pre[:, i] < perf['thr'][i]] = 0
pd.DataFrame(y_pre).to_csv(os.path.join(args.data_path, args.species, args.data_result, "gcn_vae_1_phe_pre01.txt"), index=False,
                         sep = '\t')