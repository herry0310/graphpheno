from collections import defaultdict
import pandas as pd
import argparse
import os, json
import numpy as np
from numpy import *
from sklearn.metrics import f1_score,auc,roc_curve
import math
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default="../../data/", help="path storing data.")
parser.add_argument('--species', type=str, default="hpo_2019", help="which species to use.")
parser.add_argument('--ppi_attributes', type=int, default=1, help="types of attributes used by ppi.")
parser.add_argument('--model', type=str, default="gcn_vae", help="Model string.")
parser.add_argument('--supervised', type=str, default="nn", help="neural networks or svm")
parser.add_argument('--graph', type=str, default="combined", help="lists of graphs to use.")
parser.add_argument('--thr_ppi', type=float, default=0.3, help="threshold for combiend ppi network.")
parser.add_argument('--epochs_ppi', type=int, default=80, help="Number of epochs to train ppi.")
parser.add_argument('--save_results', type=int, default=1, help="whether to save the performance results")
parser.add_argument('--data_result', type=str, default="networks/五倍交叉验证/ppi+seq/", help="path storing data.")
args = parser.parse_args()


def get_label_frequency(ontology):
    col_sums = ontology.sum(0)
    index_11_30 = np.where((col_sums >= 10) & (col_sums <= 30))[0]
    index_31_100 = np.where((col_sums >= 31) & (col_sums <= 100))[0]
    index_101_300 = np.where((col_sums >= 101) & (col_sums <= 300))[0]
    index_larger_300 = np.where(col_sums >= 301)[0]
    return index_11_30, index_31_100, index_101_300, index_larger_300

def evaluate_genes(labels, preds, threshold):
    total = 0
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    p = 0.0
    r = 0.0
    p_total= 0
    roc_auc = 0.0
    # for i in range(len(labels)):
    for i in range(0,len(labels)):
        predictions = (preds[i] > threshold).astype(np.int32)
        tpn = np.sum(predictions * labels[i])
        fpn = np.sum(predictions) - tpn
        fnn = np.sum(labels[i]) - tpn
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if np.sum(predictions) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    auc = compute_roc(labels[i], preds[i])
    if not math.isnan(auc):
        roc_auc += auc
    else:
        roc_auc += 1
    roc_auc /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    return f, p, r,roc_auc

def evaluate_annotations(labels, preds, threshold):#以term为中心
    total = 0
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    p = 0.0
    r = 0.0
    p_total= 0
    roc_auc = 0.0
    for i in range(0,preds.shape[1]):
        num_pos = sum(labels[:, i])
        num_pos = num_pos.astype(float)
        if num_pos == 0:
            continue
        preds = np.round(preds, 2)#浮点数向下取整
        labels = labels.astype(np.int32)
        predictions = (preds[:, i] > threshold).astype(np.int32)
        # p0 = (preds < threshold).astype(np.int32)
        tpn = np.sum(predictions * labels[:,i])
        fpn = np.sum(predictions) - tpn
        fnn = np.sum(labels[:,i]) - tpn
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if np.sum(predictions) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
        auc = compute_roc(labels[:, i], preds[:, i])
        if not math.isnan(auc):
            roc_auc += auc
        else:
            roc_auc += 1
    roc_auc /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    return f, p, r,roc_auc

def evaluate_performance(labels, preds):
    perf = dict()
    fmax = 0.0
    tmax = 0.0
    pmax = 0.0
    rmax = 0.0
    roc_auc = 0.0
    p_fmax = 0.0
    p_tmax = 0.0
    p_pmax = 0.0
    p_rmax = 0.0
    p_roc_auc = 0.0
    precisions = []
    recalls = []
    p_precisions = []
    p_recalls = []

    for t in range(0, 101):
        threshold = t / 100.0
        fscore, prec, rec,roc_auc = evaluate_annotations(labels, preds, threshold)
        precisions.append(prec)
        recalls.append(rec)
        print(f'Fscore: {fscore}, threshold: {threshold}')
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
            # max_preds = preds
            pmax = prec
            rmax = rec
        threshold = t / 100.0
        p_fscore, p_prec, p_rec,p_roc_auc = evaluate_genes(labels, preds, threshold)
        p_precisions.append(p_prec)
        p_recalls.append(p_rec)
        print(f'Fscore: {p_fscore}, threshold: {threshold}')
        if p_fmax < p_fscore:
            p_fmax = p_fscore
            p_tmax = threshold
            # max_preds = preds
            p_pmax = p_prec
            p_rmax = p_rec
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    # print(f'AUPR: {aupr:0.3f}, Fmax: {fmax:0.3f}, Prec: {pmax:0.3f}, Rec: {rmax:0.3f},  threshold: {tmax}')

    p_precisions = np.array(p_precisions)
    p_recalls = np.array(p_recalls)
    p_sorted_index = np.argsort(p_recalls)
    p_recalls = p_recalls[p_sorted_index]
    p_precisions = p_precisions[p_sorted_index]
    p_aupr = np.trapz(p_precisions, p_recalls)
    # plt.figure()
    # lw = 2
    # plt.plot(recalls, precisions, color='darkorange',
    #          lw=lw, label=f'AUPR curve (area = {aupr:0.2f})')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Area Under the Precision-Recall curve')
    # plt.legend(loc="lower right")
    # df = pd.DataFrame({'precisions': precisions, 'recalls': recalls})
    # df.to_pickle(f'PR.pkl')
    perf['fmax'] = fmax
    perf['pmax'] = pmax
    perf['rmax'] = rmax
    # perf['tmax'] = tmax
    # perf['precision'] = precisions
    # perf['recall'] = recalls
    perf["aupr"] = aupr
    perf["auc"] = roc_auc
    perf['p_fmax'] = p_fmax
    perf['p_pmax'] = p_pmax
    perf['p_rmax'] = p_rmax
    perf["p_aupr"] = p_aupr
    perf["p_auc"] = p_roc_auc
    return perf
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc
# def evaluate_gene_hpo(labels, preds, threshold):
#     p = 0.0
#     r = 0.0
#     p_total = 0
#     num_pos = sum(labels)
#     num_pos = num_pos.astype(float)
#     preds = np.round(preds, 2)
#     labels = labels.astype(np.int32)
#     predictions = (preds > threshold).astype(np.int32)
#     # p0 = (preds < threshold).astype(np.int32)
#     tpn = np.sum(predictions * labels)
#     fpn = np.sum(predictions) - tpn
#     fnn = np.sum(labels) - tpn
#     r = tpn / (1.0 * (tpn + fnn))
#     p = tpn / (1.0 * (tpn + fpn))
#     f = 0.0
#     if p + r > 0:
#         f = 2 * p * r / (p + r)
#     return f, p, r
# def calculate_Fmax(labels, preds):
#     fmax = 0.0
#     tmax = 0.0
#     pmax = 0.0
#     rmax = 0.0
#     precisions = []
#     recalls = []
#
#     max_preds = None
#     for t in range(0, 101):
#         threshold = t / 100.0
#         fscore, prec, rec = evaluate_gene_hpo(labels, preds, threshold)
#         precisions.append(prec)
#         recalls.append(rec)
#         print(f'Fscore: {fscore}, threshold: {threshold}')
#         if fmax < fscore:
#             fmax = fscore
#             tmax = threshold
#             pmax = prec
#             rmax = rec
#     precisions = np.array(precisions)
#     recalls = np.array(recalls)
#     sorted_index = np.argsort(recalls)
#     recalls = recalls[sorted_index]
#     precisions = precisions[sorted_index]
#     aupr = np.trapz(precisions, recalls)
#     print(f'AUPR: {aupr:0.3f}, Fmax: {fmax:0.3f}, Prec: {pmax:0.3f}, Rec: {rmax:0.3f},  threshold: {tmax}')
#     plt.figure()
#     lw = 2
#     plt.plot(recalls, precisions, color='darkorange',
#              lw=lw, label=f'AUPR curve (area = {aupr:0.2f})')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Area Under the Precision-Recall curve')
#     plt.legend(loc="lower right")
#     df = pd.DataFrame({'precisions': precisions, 'recalls': recalls})
#     df.to_pickle(f'PR.pkl')
#     return fmax, pmax, rmax,aupr
#



def get_result(ontology, Y_test, y_score):
    perf = defaultdict(dict)
    # index_11_30, index_31_100, index_101_300, index_301 = get_label_frequency(ontology)
    #
    # perf['11-30'] = evaluate_performance(Y_test[:, index_11_30], y_score[:, index_11_30])
    # perf['31-100'] = evaluate_performance(Y_test[:, index_31_100], y_score[:, index_31_100])
    # perf['101-300'] = evaluate_performance(Y_test[:, index_101_300], y_score[:, index_101_300])
    # perf['301-'] = evaluate_performance(Y_test[:, index_301], y_score[:, index_301])
    perf['all'] = evaluate_performance(Y_test, y_score)

    # plot_PRCurve( Y_test, y_score)
    return perf




