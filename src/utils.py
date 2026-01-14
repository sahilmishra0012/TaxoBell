import test
import numpy as np
import pandas as pd
import random
import torch
import pytz
from datetime import datetime, timezone
import json

# Logging and multi prediction utils file


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_local_time():
    utc_dt = datetime.now(timezone.utc)
    PST = pytz.timezone('Asia/Kolkata')
    print("Pacific time {}".format(utc_dt.astimezone(PST).isoformat()))
    return


def log_predictions_to_file(args, pred_ids, gt, id_concept, test_query_ids, output_file_path="../outputs/wordnet_verb/wordnet_verb.out", top_k=10):

    output_file_path = f"../outputs/{args.dataset}/{args.dataset}_0_kl_case_study.out"
    with open(output_file_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*20} EPOCH {'='*20}\n\n")

        for i in range(len(test_query_ids)):
            query_id = test_query_ids[i]
            query_name = id_concept.get(
                query_id, f"ID_{query_id}_(Name not found)")

            top_k_pred_ids = pred_ids[i][:top_k]
            top_k_pred_names = [id_concept.get(
                pid, f"ID_{pid}") for pid in top_k_pred_ids]

            gt_ids = gt[i]
            gt_names = [id_concept[gt_id] for gt_id in gt_ids]

            f.write(f"Query (Child): {query_name} [ID: {query_id}]\n")
            f.write(f"  > Ground Truth Parents: {gt_names}\n")
            f.write(f"  > Top {top_k} Predictions:\n")
            for rank, pred_name in enumerate(top_k_pred_names, 1):
                is_correct = pred_name in gt_names
                marker = " (*)" if is_correct else ""
                f.write(f"      {rank}. {pred_name}{marker}\n")
            f.write("-" * 50 + "\n")

    print(
        f"Predictions for epoch have been appended to {output_file_path}")


def accuracy(pred, gt, tr, te, id_term):
    preds = np.array(list(pred[:, 0]))
    gts = np.array(list(gt))

    preds_terms = list()
    gts_terms = list()

    for i in range(len(preds)):
        preds_terms.append(id_term[tr[preds[i]]])
        gts_terms.append(id_term[te[gts[i]]])
    acc = np.sum(preds_terms == gts_terms)/len(gts_terms)
    for i in range(len(preds_terms)):
        print(f"Predicted: {preds_terms[i]}, GT: {gts_terms[i]}")
    return acc


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


def hits_at_k(pred, gt, k):
    num_queries = len(gt)
    if num_queries == 0:
        return 0.0

    num_hits = 0
    for i in range(num_queries):

        preds_k = set(pred[i][:k])
        true_parents = set(gt[i])

        if not true_parents:

            continue

        if not preds_k.isdisjoint(true_parents):
            num_hits += 1

    return num_hits / num_queries


def recall_k(pred, gt, k):
    """
    Calculates the fraction of triplets where the true parent is in the top-k predictions.
    This is equivalent to Hits@k or Recall@k evaluated over all triplets.

    Args:
        pred (list of lists): The ranked predictions for each sample.
        gt (list of lists): The ground truth labels for each sample.
        k (int): The "k" in Precision@k.
    """
    total_hits = 0
    num_triplets = 0

    for i in range(len(gt)):
        preds_k = set(pred[i][:k])

        true_parents = gt[i]
        num_triplets += len(true_parents)

        if not true_parents:
            continue

        for true_parent in true_parents:
            if true_parent in preds_k:
                total_hits += 1

    if num_triplets == 0:
        return 0.0

    return total_hits / num_triplets


def rank_scores(pred, gt):
    total_mrr = 0.0
    total_mr = 0.0
    num_triplets = 0

    for i in range(len(gt)):
        ranked_list = pred[i]

        true_parents = gt[i]

        num_triplets += len(true_parents)

        rank_map = {item: rank + 1 for rank, item in enumerate(ranked_list)}

        for true_parent in true_parents:
            if true_parent in rank_map:
                rank = rank_map[true_parent]

                total_mr += rank
                total_mrr += 1 / rank
            else:
                pass

    if num_triplets == 0:
        return 0.0, 0.0

    mrr = total_mrr / num_triplets
    mr = total_mr / num_triplets

    return mrr, mr

# def hits_at_k(pred, gt, k):
#     preds = np.array(list(pred[:, :k]))
#     gts = np.array(list(gt))
#     val = np.sum(preds == gts[:, np.newaxis])*1.0/(len(gt)*k)
#     return val


# def rank_scores(pred, gt):
#     mrr = 0
#     mr = 0
#     dcg = 0.0
#     idcg = 0.0
#     cnt = 0
#     for i in range(len(pred)):
#         for j in range(len(pred[i])):
#             if pred[i][j] == gt[i]:
#                 # print(f"For {i}th element: We've found it at {j}th position")
#                 mr += (j+1)
#                 mrr += (1/(j+1))
#                 cnt += 1
#                 dcg += (1/np.log2((j+1)+1))
#                 idcg += (1/(np.log2(cnt+1)))
#                 break
#     ndcg = dcg/idcg if idcg != 0 else 0
#     ndcg = ndcg/len(gt)
#     mrr = mrr/len(gt)
#     mr = mr/len(gt)

#     return mrr, mr


def metrics(args, indices, gt, candidate_list, id_concept, test_concepts_id, out=False):
    ind = np.squeeze(indices.detach().cpu().numpy())
    x, y = ind.shape

    pred = np.zeros_like(ind)

    for i in range(x):

        pred[i] = candidate_list[ind[i]]

    mrr, mr = rank_scores(
        pred, gt)
    prec5 = hits_at_k(pred, gt, 5)
    prec10 = hits_at_k(pred, gt, 10)
    prec1 = hits_at_k(pred, gt, 1)
    rec1 = recall_k(pred, gt, 1)
    rec5 = recall_k(pred, gt, 5)
    rec10 = recall_k(pred, gt, 10)
    if out:
        log_predictions_to_file(
            args, pred, gt, id_concept, list(test_concepts_id))

    return {"Prec@1": prec1, "MRR": mrr, "MR": mr, "Recall@1": rec1, "Prec@5": prec5, "Prec@10": prec10, "Recall@5": rec5, "Recall@10": rec10}
