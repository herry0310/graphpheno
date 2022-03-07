import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    roc_auc = auc(fpr, tpr)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'auc': pd.Series(roc_auc, index=i),
                        'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold']), list(roc_t['auc'])


# Find optimal probability threshold   ###y_hp_data.iloc[:,:].values
def roc_thr(label, predict):
    roc_thr_list = []
    roc_auc_list = []
    for i in range(predict.shape[1]):
        threshold, roc_auc = Find_Optimal_Cutoff(label[:, i], predict[:, i])

        predict[:, i][predict[:, i] >= threshold] = 1
        predict[:, i][predict[:, i] < threshold] = 0
        roc_thr_list.append(threshold)
        roc_auc_list.append(roc_auc)
    return predict, roc_thr_list, roc_auc_list

def get_label_frequency(ontology):
    col_sums = ontology.sum(0)
    index_11_30 = np.where((col_sums>=10) & (col_sums<=30))[0]
    index_31_100 = np.where((col_sums>=31) & (col_sums<=100))[0]
    index_101_300 = np.where((col_sums>=101) & (col_sums<=300))[0]
    index_larger_300 = np.where(col_sums >= 301)[0]
    return index_11_30, index_31_100, index_101_300, index_larger_300
def calculate_fmax(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max,t_max,p_max,r_max

def calculate_f1_score(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    threshold = 0.5
    predictions = (preds > threshold).astype(np.int32)
    p0 = (preds < threshold).astype(np.int32)
    tp = np.sum(predictions * labels)
    fp = np.sum(predictions) - tp
    fn = np.sum(labels) - tp
    tn = np.sum(p0) - fn
    sn = tp / (1.0 * np.sum(labels))
    sp = np.sum((predictions ^ 1) * (labels ^ 1))
    sp /= 1.0 * np.sum(labels ^ 1)
    fpr = 1 - sp
    precision = tp / (1.0 * (tp + fp))
    recall = tp / (1.0 * (tp + fn))
    f = 2 * precision * recall / (precision + recall)
    acc = (tp + tn) / (tp + fp + tn + fn)
    return f,acc,precision,recall



def evaluate_performance(y_test, y_score):
    """Evaluate performance"""
    n_classes = y_test.shape[1]
    perf = dict()

    perf["M-aupr"] = 0.0
    perf["M-auc"] = 0.0
    n = 0
    aupr_list = []
    auc_list = []
    num_pos_list = []
    for i in range(n_classes):
        num_pos = sum(y_test[:, i])
        num_pos = num_pos.astype(float)
        if num_pos > 0:
            ap = average_precision_score(y_test[:, i], y_score[:, i])
            auc = roc_auc_score(y_test[:, i], y_score[:, i])
            n += 1
            perf["M-aupr"] += ap
            perf["M-auc"] += auc
            aupr_list.append(ap)
            auc_list.append(auc)
            num_pos_list.append(num_pos)
    perf["M-aupr"] /= n
    perf['aupr_list'] = aupr_list
    perf['num_pos_list'] = num_pos_list
    perf["M-auc"] /= n
    perf['auc_list'] = auc_list

    # Compute micro-averaged AUPR
    perf['m-aupr'] = average_precision_score(y_test.ravel(), y_score.ravel())
    perf['m-auc'] = roc_auc_score(y_test.ravel(), y_score.ravel())

    perf['F-max'],perf['t_max'],perf['p_max'],perf['r_max'] = calculate_fmax(y_score, y_test)
    perf['F1-score'], perf['accuracy'], perf['precision'], perf['recall'] = calculate_f1_score(y_score, y_test)


    return perf

def get_results(ontology, Y_test, y_score):
    perf = defaultdict(dict) 
    index_11_30, index_31_100,index_101_300, index_301 = get_label_frequency(ontology)

    perf['11-30'] = evaluate_performance(Y_test[:,index_11_30], y_score[:,index_11_30])
    perf['31-100'] = evaluate_performance(Y_test[:,index_31_100], y_score[:,index_31_100])
    perf['101-300'] = evaluate_performance(Y_test[:,index_101_300], y_score[:,index_101_300])
    perf['301-'] = evaluate_performance(Y_test[:,index_301], y_score[:,index_301])
    perf['all'] = evaluate_performance(Y_test, y_score)


    # plot_PRCurve( Y_test, y_score)
    return perf
    
