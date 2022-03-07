from input_data import load_data,load_labels
from trainGcn import train_gcn,train_vae
import pandas as pd
import os,json
import argparse
import numpy as np
import pickle as pkl
from trainNN import train_nn
from sklearn.model_selection import KFold
from evaluation import get_results, roc_thr



def reshape(features):
    return np.hstack(features).reshape((len(features),len(features[0])))


def train(args):
    # load feature dataframe
    print("loading features...") 
    uniprot = pd.read_pickle(os.path.join(args.data_path, args.species + "/features.pkl"))
    uniprot_label = pd.read_pickle(os.path.join(args.data_path, args.species + "/features_label.pkl"))
    idx = uniprot[~uniprot["HPO-Term-ID"].isna()].index.tolist()

    print("#############################")
    print("Training",args.graph)
    adj, features = load_data(args.graph, uniprot, args)
    embeddings = train_gcn(features, adj, args, args.graph)


    path = os.path.join(args.data_path, args.species + "/networks/" + args.model + "_" + str(args.lr) + "_" + str(args.thr_ppi) + "_" + args.graph + "_" + str(args.ppi_attributes) +"_embeddings_" + args.species + "_" + str(args.epochs_ppi) + ".pkl")
    with open(path, 'wb') as file:
        pkl.dump(embeddings, file)
    file.close()


    if args.only_gcn == 1:
        return


    hp = uniprot_label['hp_label'].values
    hp = np.hstack(hp).reshape((len(hp), len(hp[0])))
    np.random.seed(5000)
    # vgae embedding
    embedding = pd.read_pickle(
        os.path.join(args.data_path, args.species + "/networks/" + args.model + "_" + str(args.lr) + "_" + str(args.thr_ppi) + "_" + args.graph + "_" + str(args.ppi_attributes) +"_embeddings_" + args.species + "_" + str(args.epochs_ppi) + ".pkl"))
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

    print("###################################")
    print('----------------------------------')
    print('5hp')

    # save predicted score
    hp_label_list = pd.read_table(os.path.join(args.data_path, args.species + "/hp_list.txt"), header=None)
    pd.DataFrame(y_hp_data).to_csv(
        os.path.join(args.data_path, args.species, args.data_result, "gcn_vae_1_phe_pre.txt"), index=True,
        sep='\t', index_label='id', header=[x[0] for x in hp_label_list.values])
    pd.DataFrame(Y_hp_data).to_csv(
        os.path.join(args.data_path, args.species, args.data_result, "gcn_vae_1_phe_true.txt"), index=True,
        sep='\t', index_label='id', header=[x[0] for x in hp_label_list.values])



    print("Start running supervised model...")
    rand_str = np.random.randint(10)  # 产生均匀分布的随机整数矩阵
    save_path = os.path.join(args.data_path,
                             args.species + "/results_new/results_graphpheno_" + args.supervised + "_" +
                             args.graph + "_" + str(args.ppi_attributes) + "_" + str(args.thr_ppi) + "_" + str(args.lr) + "_" + str(
                                 args.epochs_ppi))

    print("###################################")
    print('----------------------------------')
    print('5hp')

    perf_hp_all = get_results(hp, Y_hp_data, y_hp_data)
    if args.save_results:
        with open(save_path + "_5hp.json", "w") as f:
            json.dump(perf_hp_all, f)

    # predict, roc_thr_list, roc_auc_list = roc_thr(Y_hp_data,y_hp_data)
    predict, roc_thr_list, roc_auc_list = roc_thr(Y_hp_data.values, y_hp_data.values)
    pd.DataFrame(predict).to_csv(
        os.path.join(args.data_path, args.species, args.data_result, "gcn_vae_1_phe_pre01.txt"), index=True,
        sep='\t', index_label='id', header=[x[0] for x in hp_label_list.values])

    perf = dict()
    perf['roc_auc'] = [x[0] for x in roc_auc_list]
    perf['roc_thr'] = [x[0] for x in roc_thr_list]
    f = pd.DataFrame.from_dict(perf, orient='index', columns=None)
    f = f.reset_index().rename(columns={'index': 'evaluation index'})
    # f = f.reset_index().rename(columns=[x[0] for x in hp_label_list.values])
    f.to_csv(os.path.join(args.data_path, args.species, args.data_result, "2019_phe_fre.txt"), index=False, sep='\t')




if __name__ == "__main__":
    parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter)#创建解析器
    #global parameters,  添加参数
    parser.add_argument('--ppi_attributes', type=int, default=1, help="types of attributes used by ppi.")
    parser.add_argument('--simi_attributes', type=int, default=0, help="types of attributes used by simi.")
    parser.add_argument('--graph', type=str, default="combined", help="lists of graphs to use.")
    parser.add_argument('--species', type=str, default="CAFA2", help="which species to use.")
    parser.add_argument('--data_path', type=str, default="../../data/", help="path storing data.")
    parser.add_argument('--thr_ppi', type=float, default=0.0, help="threshold for combiend ppi network.")
    parser.add_argument('--thr_evalue', type=float, default=1e-4, help="threshold for similarity network.")
    parser.add_argument('--supervised', type=str, default="nn", help="neural networks or svm")
    parser.add_argument('--only_gcn', type=int, default=1, help="0 for training all, 1 for only embeddings.")
    parser.add_argument('--save_results', type=int, default=1, help="whether to save the performance results")
    
    #parameters for traing GCN
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--epochs_ppi', type=int, default=80, help="Number of epochs to train ppi.")
    parser.add_argument('--epochs_simi', type=int, default=300, help="Number of epochs to train similarity network.")
    parser.add_argument('--hidden1', type=int, default=800, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=400, help="Number of units in hidden layer 2.")
    parser.add_argument('--weight_decay', type=float, default=0., help="Weight for L2 loss on embedding matrix.")
    parser.add_argument('--dropout', type=float, default=0., help="Dropout rate (1 - keep probability).")
    parser.add_argument('--model', type=str, default="gcn_vae", help="Model string.")
    parser.add_argument('--data_result', type=str, default="networks/5-fold_cross/ppi/",
                        help="path storing data.")


    args = parser.parse_args()#解析参数
    print(args)
    train(args)

