import test
import numpy as np
import pandas as pd
import random
import torch
import pytz
from datetime import datetime, timezone


def accuracy(pred, gt, tr, te):
    """
    Computes the exact match accuracy for the top-1 prediction.

    Also prints the predicted term vs ground truth term for debugging purposes.

    Args:
        pred (np.ndarray): Prediction array of shape (batch_size, k). 
                           Column 0 is used for Top-1.
        gt (np.ndarray): Ground truth array of shape (batch_size,).
        tr (dict): Mapping from Train ID to Concept Name (Term).
        te (dict): Mapping from Test ID to Concept Name (Term).

    Returns:
        float: The accuracy score (0.0 to 1.0).
    """
    preds = np.array(list(pred[:, 0]))
    gts = np.array(list(gt))

    acc = np.sum(preds == gts) / len(gt)

    print("Predictions: ", preds)
    print("GT: ", gts)
    for i in range(len(preds)):
        p_term = tr.get(preds[i], f"ID_{preds[i]}")
        gt_term = te.get(gts[i], f"ID_{gts[i]}")
        print(f"Predicted: {p_term}, GT: {gt_term}")

    return acc


def precision_k(pred, gt, k):
    """
    Calculates Precision at K (P@K).

    For a single-label task (one parent), P@K is defined as:
    (Relevant Items in Top K) / K.

    Since there is only 1 relevant item, the maximum P@K is 1/K.

    Args:
        pred (np.ndarray): Prediction matrix of shape (batch_size, >=k).
        gt (np.ndarray): Ground truth vector of shape (batch_size,).
        k (int): The cutoff rank.

    Returns:
        float: Average Precision@K.
    """
    preds = np.array(list(pred[:, :k]))
    gts = np.array(list(gt))

    val = np.sum(preds == gts[:, np.newaxis]) * 1.0 / (len(gt) * k)
    return val


def recall_k(pred, gt, k):
    """
    Calculates Recall at K (R@K), also known as Hit@K for single-label tasks.

    Measures the percentage of test samples where the true parent appears 
    within the top K predictions.

    Args:
        pred (np.ndarray): Prediction matrix.
        gt (np.ndarray): Ground truth vector.
        k (int): The cutoff rank.

    Returns:
        float: Recall score (0.0 to 1.0).
    """
    num_queries = len(gt)
    if num_queries == 0:
        return 0.0

    preds_k = pred[:, :k]
    gts = np.array(gt)[:, np.newaxis]

    hits_matrix = (preds_k == gts)
    successful_queries = np.any(hits_matrix, axis=1)

    return np.mean(successful_queries)


def rank_scores(pred, gt):
    """
    Computes ranking metrics: MRR, MR, and NDCG.

    Args:
        pred (np.ndarray): Prediction matrix.
        gt (np.ndarray): Ground truth vector.

    Returns:
        tuple: 
            - mrr (float): Mean Reciprocal Rank.
            - mr (float): Mean Rank.
            - ndcg (float): Normalized Discounted Cumulative Gain.
    """
    mrr = 0
    mr = 0
    dcg = 0.0
    idcg = 0.0
    cnt = 0

    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j] == gt[i]:
                rank = j + 1
                mr += rank
                mrr += (1 / rank)
                cnt += 1

                dcg += (1 / np.log2(rank + 1))
                idcg += (1 / (np.log2(cnt + 1)))
                break

    ndcg = dcg / idcg if idcg != 0 else 0
    ndcg = ndcg / len(gt)
    mrr = mrr / len(gt)
    mr = mr / len(gt)

    return mrr, mr, ndcg


def mrr_score(pred, gt):
    """
    Calculates Mean Reciprocal Rank (MRR) only.

    Args:
        pred (np.ndarray): Prediction matrix.
        gt (np.ndarray): Ground truth vector.

    Returns:
        float: MRR score.
    """
    mrr = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j] == gt[i]:
                mrr += 1 / (j + 1)
                break
    mrr = mrr / len(gt)
    return mrr


def wu_p_score(pred, gt, path2root, compiled):
    """
    Calculates the Wu-Palmer Similarity score between predicted and true parents.

    Wu-P Similarity is based on the depth of the two concepts in the taxonomy
    and the depth of their Least Common Ancestor (LCA).

    Side Effect:
        Appends the depth of the Ground Truth node to the `compiled` list for analysis.

    Args:
        pred (np.ndarray): Prediction matrix (uses top-1).
        gt (np.ndarray): Ground truth vector.
        path2root (dict): Map of ID -> List of IDs representing path to root.
        compiled (list): List of lists used to store metadata for depth analysis.

    Returns:
        float: Average Wu-Palmer score.
    """
    pred = np.squeeze(pred[:, 0])
    wu_p = 0
    for i in range(len(pred)):
        path_pred = path2root[pred[i]]
        path_gt = path2root[gt[i]]

        compiled[i].append(len(path_gt) - 1)

        shared_nodes = set(path_pred) & set(path_gt)
        lca_depth = 1
        for node in shared_nodes:
            lca_depth = max(len(path2root[node]) - 1, lca_depth)

        wu_p += 2 * lca_depth / (len(path_pred) + len(path_gt))

    wu_p = wu_p / len(gt)

    return wu_p


def metrics_single_parent(indices, gt, train_concept_set, path2root, testid2concept, trainid2concept, testconcepts, sortedscores):
    """
    Master function to compute evaluation metrics for Single-Parent (Tree) Taxonomies.

    Orchestrates the conversion of tensor indices to concept IDs, calculates standard metrics
    (Accuracy, MRR, P@K, R@K, Wu-P), and performs a depth-wise accuracy analysis using Pandas.

    Args:
        indices (torch.Tensor): Indices of candidate parents sorted by score (descending).
        gt (list/np.array): Ground Truth parent IDs.
        train_concept_set (set): Set of all potential parent IDs available in training.
        path2root (dict): Taxonomy structure mapping IDs to paths.
        testid2concept (dict): Map of Test IDs to names.
        trainid2concept (dict): Map of Train IDs to names.
        testconcepts (list): IDs of the child concepts being tested.
        sortedscores (torch.Tensor): The actual similarity scores corresponding to `indices`.

    Returns:
        dict: A dictionary containing all computed metrics.
    """
    ind = np.squeeze(indices.detach().cpu().numpy())
    x, y = ind.shape

    candidate_list = np.array(list(train_concept_set))
    pred = np.zeros((x, y), dtype=int)

    compiled = [[testid2concept[testconcepts[i]], trainid2concept[gt[i]],
                 sortedscores[i][0].item()] for i in range(x)]

    for i in range(x):
        pred[i] = candidate_list[ind[i]]

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

    depth_data = [elem[-2:] for elem in compiled]

    np.savetxt("depth_analysis.csv", depth_data, fmt="%s,%i", delimiter=",")

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

    return {
        "Prec@1": prec1,
        "MRR": mrr,
        "MR": mr,
        "Wu": wu_p,
        "Prec@5": prec5,
        "Prec@10": prec10,
        "Recall@1": rec1,
        "Recall@5": rec5,
        "Recall@10": rec10,
        "WuP": wu_p
    }
