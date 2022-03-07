import pandas as pd
import os,json
import argparse
import numpy as np
from numpy import *
import pickle as pkl
from trainNN import train_nn
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score,roc_auc_score,roc_curve,auc
from deeppheno_evaluation import get_result
import heapq
import math

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default="../../data/", help="path storing data.")
parser.add_argument('--species', type=str, default="CAFA2", help="which species to use.")
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
uniprot_pre = pd.read_pickle(os.path.join(args.data_path, args.species + "/test1/features_pre.pkl"))
train = pd.read_pickle(os.path.join(args.data_path, args.species, "human.pkl"))
test = pd.read_pickle(os.path.join(args.data_path, args.species, "human_test.pkl"))

uniprot_label_index = uniprot[uniprot['Entry name'].isin(train['proteins'].values)].index
uniprot_pre_index = uniprot[uniprot['Entry name'].isin(test['proteins'].values)].index

hp = uniprot_label['hp_label'].values
hp = np.hstack(hp).reshape((len(hp), len(hp[0])))
test_hp = uniprot_pre['hp_label'].values
test_hp = np.hstack(test_hp).reshape((len(test_hp), len(test_hp[0])))
np.random.seed(5000)
embedding = pd.read_pickle(
    os.path.join(args.data_path, args.species + "/networks/gcn_vae_0.001_0.0_combined_1_embeddings_CAFA2(5)_80.pkl"))
train_embedding = embedding[uniprot_label_index, :]
test_embedding = embedding[uniprot_pre_index, :]

#预测打分
y_score_hp = train_nn(train_embedding, hp, test_embedding, test_hp)
#评估指标
perf_hp_all = get_result(hp, test_hp, y_score_hp)

print("Start running supervised model...")
rand_str = np.random.randint(10)  # 产生均匀分布的随机整数矩阵
save_path = os.path.join(args.data_path,
                         args.species + "/networks/results_gcn_vae_0.001_combined_1_embeddings_CAFA2(5)_80")

if args.save_results:
    with open(save_path + "_CAFA2.json", "w") as f:
        json.dump(perf_hp_all, f)