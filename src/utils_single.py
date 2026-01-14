import test
import numpy as np
import pandas as pd
import random
import torch
import pytz
from datetime import datetime, timezone

# Single parent metrics file.


def accuracy(pred, gt, tr, te):
    preds = np.array(list(pred[:, 0]))
    gts = np.array(list(gt))
    acc = np.sum(preds == gts)/len(gt)
    print("Predictions: ", preds)
    print("GT: ", gts)
    for i in range(len(preds)):
        print(f"Predicted: {tr[preds[i]]}, GT: {te[gts[i]]}")
    return acc


def precision_k(pred, gt, k):
    preds = np.array(list(pred[:, :k]))
    gts = np.array(list(gt))
    val = np.sum(preds == gts[:, np.newaxis])*1.0/(len(gt)*k)
    return val


def recall_k(pred, gt, k):
    num_queries = len(gt)
    if num_queries == 0:
        return 0.0

    preds_k = pred[:, :k]
    gts = np.array(gt)[:, np.newaxis]

    hits_matrix = (preds_k == gts)
    successful_queries = np.any(hits_matrix, axis=1)

    return np.mean(successful_queries)


def rank_scores(pred, gt):
    mrr = 0
    mr = 0
    dcg = 0.0
    idcg = 0.0
    cnt = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j] == gt[i]:
                mr += (j+1)
                mrr += (1/(j+1))
                cnt += 1
                dcg += (1/np.log2((j+1)+1))
                idcg += (1/(np.log2(cnt+1)))
                break
    ndcg = dcg/idcg if idcg != 0 else 0
    ndcg = ndcg/len(gt)
    mrr = mrr/len(gt)
    mr = mr/len(gt)

    return mrr, mr, ndcg


def mrr_score(pred, gt):
    mrr = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j] == gt[i]:
                mrr += 1/(j+1)
    mrr = mrr/len(gt)
    return mrr


def wu_p_score(pred, gt, path2root, compiled):

    pred = np.squeeze(pred[:, 0])
    wu_p = 0
    for i in range(len(pred)):
        path_pred = path2root[pred[i]]
        path_gt = path2root[gt[i]]
        compiled[i].append(len(path_gt)-1)
        shared_nodes = set(path_pred) & set(path_gt)
        lca_depth = 1
        for node in shared_nodes:
            lca_depth = max(len(path2root[node])-1, lca_depth)
        wu_p += 2*lca_depth/(len(path_pred)+len(path_gt))

    wu_p = wu_p/len(gt)

    return wu_p


def metrics_single_parent(indices, gt, train_concept_set, path2root, testid2concept, trainid2concept, testconcepts, sortedscores):
    ind = np.squeeze(indices.detach().cpu().numpy())
    x, y = ind.shape
    pred = np.array([[i for i in range(y)] for _ in range(x)])
    compiled = [[testid2concept[testconcepts[i]], trainid2concept[gt[i]],
                 sortedscores[i][0].item()] for i in range(x)]

    for i in range(len(pred)):
        pred[i] = np.array(list(train_concept_set))[ind[i]]
        compiled[i].append(trainid2concept[pred[i][0]])
        compiled[i].append(True if pred[i][0] == gt[i] else False)

    acc = accuracy(pred, gt, trainid2concept, testid2concept)
    mrr, mr, ndcg = rank_scores(pred, gt)
    wu_p = wu_p_score(pred, gt, path2root, compiled)
    prec5 = precision_k(pred, gt, 5)
    prec10 = precision_k(pred, gt, 10)
    prec1 = precision_k(pred, gt, 1)
    rec1 = recall_k(pred, gt, 1)
    rec10 = recall_k(pred, gt, 10)
    rec5 = recall_k(pred, gt, 5)
    depth = [elem[-2:] for elem in compiled]
    np.savetxt("depth_analysis.csv", depth, fmt="%s,%i", delimiter=",")

    file_path = 'depth_analysis.csv'
    data = pd.read_csv(file_path, header=None, names=['correct', 'depth'])

    report = data.groupby('depth').agg(
        total_attempts=('correct', 'count'),
        correct_answers=('correct', 'sum'),
        accuracy=('correct', lambda x: x.mean() * 100)
    ).reset_index()

    report.columns = ['Depth', 'Total Attempts',
                      'Correct Answers', 'Accuracy (%)']

    print(report)
    return {"Prec@1": prec1, "MRR": mrr, "MR": mr, "Wu": wu_p, "Prec@5": prec5, "Prec@10": prec10, "Recall@1": rec1, "Recall@5": rec5, "Recall@10": rec10}
