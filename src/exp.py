import os
import time
import numpy as np
import pickle as pkl
import torch
import sys
import json
from tqdm import tqdm
from torch import optim
import csv
import matplotlib.patches as patches
from transformers import BertTokenizer, AutoTokenizer
from utils import *
from utils_single import *
from data import *
from model import GaussianBox
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import gc
import wandb

os.environ["WANDB_MODE"] = "offline"


class Experiments(object):
    """
    Manages the end-to-end lifecycle of the Gaussian Embedding experiments.

    This class handles:
    1. Data Loading: preparing train and test dataloaders.
    2. Model Initialization: setting up the GaussianBox model and optimizer.
    3. Training: The main training loop with gradient accumulation and checkpointing.
    4. Evaluation: Computing metrics (MRR, Hit@K, etc.) using Bhattacharyya Coefficient and KL Divergence.
    5. Visualization: Generating 2D plots of Gaussian ellipses for case studies.

    Attributes:
        args (Namespace): Configuration arguments.
        tokenizer: HuggingFace tokenizer.
        train_loader (DataLoader): Loader for training triplets.
        test_loader (DataLoader): Loader for candidate parents during evaluation.
        train_set (Dataset): Training dataset object.
        test_set (Dataset): Testing dataset object.
        model (GaussianBox): The taxonomy induction model.
        optimizer (torch.optim.Optimizer): The optimizer (Adam/AdamW).
    """

    def __init__(self, args):
        """
        Initializes the experiment environment.

        Args:
            args (Namespace): Parsed command-line arguments containing hyperparameters,
                              dataset paths, and model configuration.
        """
        super(Experiments, self).__init__()

        self.args = args
        self.tokenizer = self.__load_tokenizer__()
        self.train_loader, self.train_set = load_data(
            self.args, self.tokenizer, "train")
        self.test_loader, self.test_set = load_data(
            self.args, self.tokenizer, "test")

        self.model = GaussianBox(args)
        self.accumulation_steps = self.args.accumulation_steps
        self.optimizer = self._select_optimizer()
        self._set_device()
        self.exp_setting = "_".join([str(elem) for elem in [self.args.pre_train, self.args.dataset, self.args.expID, self.args.epochs,
                                    self.args.batch_size, self.args.mixture if self.args.mixture else "superposn", self.args.lr, "complex" if self.args.complex else "real"]])

        setting = {
            "pre_train": self.args.pre_train,
            "dataset": self.args.dataset,
            "expID": self.args.expID,
            "epochs": self.args.epochs,
            "batch_size": self.args.batch_size,
            "lr": self.args.lr,
            "hidden": self.args.hidden,
            "mixture": self.args.mixture if self.args.mixture else "superposn",
            "complex": self.args.complex,
            "matrixsize": self.args.matrixsize
        }
        print(setting)

        if self.args.wandb:
            wandb.init(project='gaussian', config=setting, entity='taxo_iitd')
            wandb.run.log_code(".")

    def __load_tokenizer__(self):
        """Loads the specific tokenizer based on the selected backbone model (BERT, Snowflake, E5)."""
        if self.args.model == 'bert':
            tokenizer = BertTokenizer.from_pretrained(
                '/home/models/bert-base-uncased')
        elif self.args.model == 'snowflake':
            tokenizer = AutoTokenizer.from_pretrained(
                'Snowflake/snowflake-arctic-embed-m')
        elif self.args.model == 'e5':
            tokenizer = AutoTokenizer.from_pretrained(
                '/home/models/e5-large-v2')
        print("Tokenizer Loaded!")
        return tokenizer

    def _select_optimizer(self):
        """Configures the optimizer with weight decay settings."""
        parameters = [{"params": [p for n, p in self.model.named_parameters()],
                       "weight_decay": 0.0},]

        if self.args.optim == "adam":
            optimizer = optim.Adam(parameters, lr=self.args.lr)
        elif self.args.optim == "adamw":
            optimizer = optim.AdamW(
                parameters, lr=self.args.lr, eps=self.args.eps)

        return optimizer

    def _set_device(self):
        """Moves the model to GPU if CUDA is enabled."""
        if self.args.cuda:
            self.model = self.model.cuda()

    def train_one_step(self, it, encode_parent, encode_child, encode_negative_parents):
        """
        Performs a single training step (forward pass, loss computation, backward pass).
        Handles gradient accumulation.

        Args:
            it (int): Current iteration index.
            encode_parent (dict): Tokenized input for parent concepts.
            encode_child (dict): Tokenized input for child concepts.
            encode_negative_parents (dict): Tokenized input for negative samples.

        Returns:
            torch.Tensor: The calculated loss for this step.
        """

        self.model.train()

        loss = self.model(encode_parent, encode_child, encode_negative_parents)
        loss = loss/self.accumulation_steps
        loss.backward()
        if not it % self.accumulation_steps:
            self.optimizer.step()
            self.optimizer.zero_grad()

        del encode_parent, encode_child, encode_negative_parents
        torch.cuda.empty_cache()
        gc.collect()

        return loss

    def train(self, checkpoint=None, save_path=None):
        """
        Main training loop.

        Iterates through epochs, performs training steps, saves checkpoints, 
        and triggers evaluation on the test set.

        Args:
            checkpoint (str, optional): Path to a checkpoint to resume training from.
            save_path (str, optional): Custom path to save the best model.
        """
        time_tracker = []
        save_path = f"../final_result/{self.args.dataset}/BC_volume_containment_{self.args.expID}_{self.args.method}_{self.args.model}_{self.args.negsamples}.pt"
        test_acc = test_mrr = test_wu_p = 0
        old_test_acc = old_test_mrr = old_test_wu_p = 0

        if checkpoint:
            self.model.load_state_dict(torch.load(f"{checkpoint}"))

        if save_path is None:
            savedir = os.path.join("../result", self.args.dataset, "model")
            traindir = os.path.join("../result", self.args.dataset, "train")
            if not os.path.exists(savedir):
                os.makedirs(savedir, exist_ok=True)
            if not os.path.exists(traindir):
                os.makedirs(traindir, exist_ok=True)
            save_path = os.path.join(
                "../result", self.args.dataset, "model", f"exp_model_{self.exp_setting}.checkpoint")

        self.model.compile()
        for epoch in tqdm(range(self.args.epochs)):
            epoch_time = time.time()
            train_loss = []

            self.optimizer.zero_grad()
            for i, (encode_parent, encode_child, encode_negative_parents) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                loss = self.train_one_step(
                    it=i, encode_parent=encode_parent, encode_child=encode_child, encode_negative_parents=encode_negative_parents)
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            print(train_loss)

            torch.save(self.model.state_dict(), os.path.join(
                "../result", self.args.dataset, "train", f"BC_volume_containment_{self.args.expID}_{self.args.method}_{self.args.model}_{self.args.negsamples}_{epoch}.checkpoint"))
            if epoch >= 1:
                os.remove(os.path.join(
                    "../result", self.args.dataset, "train", f"BC_volume_containment_{self.args.expID}_{self.args.method}_{self.args.model}_{self.args.negsamples}_{epoch-1}.checkpoint"))

            test_metrics, test_metrics_kl = self.predict()
            test_acc = test_metrics["Prec@1"]
            test_acc_kl = test_metrics_kl["Prec@1"]
            test_mrr = test_metrics['MRR']
            test_mrr_kl = test_metrics_kl['MRR']
            if test_mrr >= old_test_mrr:
                torch.save(self.model.state_dict(
                ), f"../final_result/{self.args.dataset}/BC_volume_containment_{self.args.expID}_{self.args.method}_{self.args.model}_{self.args.negsamples}_{epoch}.pt")
                old_test_mrr = test_mrr
            time_tracker.append(time.time()-epoch_time)

            print('\nEpoch: {:04d}'.format(epoch + 1),
                  'train_loss:{:.05f}'.format(train_loss),
                  'BC_hit@1:{:.05f}'.format(test_acc),
                  'BC_mrr:{:.05f}'.format(test_mrr),
                  'BC_Recall@1:{:.05f}'.format(test_metrics["Recall@1"]),
                  'BC_Recall@5:{:.05f}'.format(test_metrics["Recall@5"]),
                  'BC_Recall@10:{:0.5f}'.format(test_metrics["Recall@10"]),
                  'BC_mr:{:.05f}'.format(test_metrics["MR"]),
                  'BC_hit5:{:.05f}'.format(test_metrics["Prec@5"]),
                  'BC_hit10:{:.05f}'.format(test_metrics["Prec@10"]),
                  'KL_Hit@1:{:.05f}'.format(test_metrics_kl["Prec@1"]),
                  'KL_mrr:{:.05f}'.format(test_metrics_kl["MRR"]),
                  'KL_Recall@1:{:.05f}'.format(test_metrics_kl["Recall@1"]),
                  'KL_mr:{:.05f}'.format(test_metrics_kl["MR"]),
                  'KL_Hit@5:{:.05f}'.format(test_metrics_kl["Prec@5"]),
                  'KL_Hit@10:{:.05f}'.format(test_metrics_kl["Prec@10"]),
                  'KL_Recall@5:{:.05f}'.format(test_metrics_kl["Recall@5"]),
                  'KL_Recall@10: {:.05f}'.format(test_metrics_kl["Recall@10"]),
                  'epoch_time:{:.01f}s'.format(time.time()-epoch_time),
                  'remain_time:{:.01f}s'.format(
                      np.mean(time_tracker)*(self.args.epochs-(1+epoch))),
                  )

            wandb.log({
                'train_loss': (train_loss),
                'BC_hit@1': (test_acc),
                'BC_mrr': (test_mrr),
                'BC_Recall@1': (test_metrics['Recall@1']),
                'BC_Recall@5': (test_metrics['Recall@5']),
                'BC_Recall@10': (test_metrics["Recall@10"]),
                'BC_mr': (test_metrics["MR"]),
                'BC_hit@5': (test_metrics["Prec@5"]),
                'BC_hit@10': (test_metrics["Prec@10"]),
                'KL_hit@1': (test_acc_kl),
                'KL_mrr': (test_mrr_kl),
                'KL_Recall@1': (test_metrics_kl['Recall@1']),
                'KL_Recall@5': (test_metrics_kl['Recall@5']),
                'KL_Recall@10': (test_metrics_kl["Recall@10"]),
                'KL_mr': (test_metrics_kl["MR"]),
                'KL_hit@5': (test_metrics_kl["Prec@5"]),
                'KL_hit@10': (test_metrics_kl["Prec@10"]),
            })

            torch.save(self.model.state_dict(), os.path.join("../result", self.args.dataset,
                       "train", "exp_model_"+self.exp_setting+"_"+str(epoch)+".checkpoint"))
            if epoch:
                os.remove(os.path.join("../result", self.args.dataset, "train",
                          "exp_model_"+self.exp_setting+"_"+str((epoch-1))+".checkpoint"))

    def plot_gaussians(self, q_mu, q_sigma, top_mus, top_sigmas, query_name, top_names, save_path,
                       extra_gts=None):
        """
        Visualizes the Gaussian embeddings of a query and its predicted parents/ground truths.

        Uses PCA to project high-dimensional Gaussian parameters (Mean and Covariance) 
        onto a 2D plane for plotting ellipses.

        Args:
            q_mu (Tensor): Query mean vector.
            q_sigma (Tensor): Query covariance matrix/diagonal.
            top_mus (list[Tensor]): List of means for top predicted parents.
            top_sigmas (list[Tensor]): List of covariances for top predicted parents.
            query_name (str): Name of the child concept.
            top_names (list[str]): Names of predicted parents.
            save_path (str): File path to save the generated plot.
            extra_gts (list, optional): List of ground truth parameters if they weren't in the top predictions.
        """
        fig, ax = plt.subplots(figsize=(16, 12))

        colors = {
            'query': '#d62728',
            'preds': ['#1f77b4', '#2ca02c', '#9467bd'],
            'gt': '#ff7f0e'
        }

        all_mus_for_pca = [q_mu] + top_mus
        if extra_gts:
            all_mus_for_pca.extend([gt[0] for gt in extra_gts])

        all_mus_np = [m.cpu().numpy() for m in all_mus_for_pca]

        pca = PCA(n_components=2)
        pca.fit(all_mus_np)

        P = torch.tensor(pca.components_, dtype=torch.float32)

        print(
            f"Plotting along the top 2 Principal Components for query '{query_name}'.")

        def plot_single_gaussian(ax, mu, sigma, color, label, zorder, linestyle, marker,
                                 is_query=False, is_predicted_parent=False):
            mu_2d = P @ mu.cpu().float()
            sigma_2d = P @ sigma.cpu().float() @ P.T
            vals, vecs = torch.linalg.eigh(sigma_2d)
            angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
            stds_2d = torch.sqrt(vals)
            std_major, std_minor = stds_2d[1], stds_2d[0]

            stats_string = f"{label}:\n  μ=[{mu_2d[0]:.3f}, {mu_2d[1]:.3f}], σ=[{std_major:.3f}, {std_minor:.3f}]"

            scale_factor = 1.0
            width, height = 1.5 * scale_factor * std_major, 1.5 * scale_factor * std_minor
            face_color = color if is_predicted_parent else 'none'
            alpha = 0.1 if is_predicted_parent else 1.0

            ellipse = patches.Ellipse(
                xy=mu_2d, width=width, height=height, angle=angle, facecolor=face_color,
                alpha=alpha, edgecolor=color,
                linewidth=4.0 if is_query else (
                    3.5 if is_predicted_parent else 2.5),
                linestyle=linestyle, label=label, zorder=zorder
            )
            ax.add_patch(ellipse)
            ax.scatter(mu_2d[0], mu_2d[1], c=color, marker=marker, s=250, zorder=zorder + 5,
                       linewidths=2, edgecolor='black')

            return (mu_2d[0] - width, mu_2d[0] + width, mu_2d[1] - height, mu_2d[1] + height), stats_string

        all_bounds, all_stats_info = [], []

        bounds, stats = plot_single_gaussian(
            ax, q_mu, q_sigma, colors['query'], f'Query: {query_name}', 1, '--', '*', True)
        all_bounds.append(bounds)
        all_stats_info.append(stats)

        for i, (p_mu, p_sigma, p_name) in enumerate(zip(top_mus, top_sigmas, top_names)):
            is_main_parent = (i == 0)
            bounds, stats = plot_single_gaussian(ax, p_mu, p_sigma, colors['preds'][i], f'Pred {i+1}: {p_name}',
                                                 i + 2, '-', 'x', is_predicted_parent=is_main_parent)
            all_bounds.append(bounds)
            all_stats_info.append(stats)

        if extra_gts:
            for gt_mu, gt_sigma, gt_name in extra_gts:
                bounds, stats = plot_single_gaussian(
                    ax, gt_mu, gt_sigma, colors['gt'], f'Ground Truth: {gt_name}', 9, ':', '+')
                all_bounds.append(bounds)
                all_stats_info.append(stats)

            ax.text(0.97, 0.97, 'A Ground Truth was not in Top 3', transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor=colors['gt'], alpha=0.5))

        min_x, max_x = min(b[0] for b in all_bounds), max(b[1]
                                                          for b in all_bounds)
        min_y, max_y = min(b[2] for b in all_bounds), max(b[3]
                                                          for b in all_bounds)
        pad_x, pad_y = (max_x - min_x) * 0.3, (max_y - min_y) * 0.3
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

        ax.legend(fontsize='large', title='Concepts', title_fontsize='x-large')
        ax.set_title(
            f'Containment Visualization for Query: "{query_name}"', fontsize=20, pad=20)
        ax.set_xlabel('Principal Component 1', fontsize=14)
        ax.set_ylabel('Principal Component 2', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)

        stats_text = "\n\n".join(all_stats_info)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        fig.tight_layout(rect=[0, 0, 0.85, 1])

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def case_study(self, tag=None, path=None):
        """
        Performs a detailed qualitative analysis (Case Study).

        1. Loads a trained model.
        2. Computes embeddings for queries and all candidates.
        3. Ranks candidates based on Bhattacharyya Coefficient and KL Divergence.
        4. Writes detailed results to a CSV file (scores, volumes, containment checks).
        5. Generates visualization plots for the top queries using `plot_gaussians`.

        Args:
            tag (str, optional): If 'test', loads weights from `path`.
            path (str, optional): Path to the checkpoint file.

        Returns:
            tuple: (test_metrics, test_metrics_kl) dictionaries containing quantitative results.
        """
        print("Prediction starting.....")
        if (tag == "test"):
            if path:
                self.model.load_state_dict(
                    torch.load(path))
        self.model.eval()
        with torch.no_grad():
            score_list = []
            score_list_kl = []
            gt_label = self.test_set.test_gt_id

            query_center, query_delta = self.model.child_projection_box(
                self.test_set.encode_query)
            query_mu, query_sigma = self.model.compute_gaussian(
                query_center, query_delta)

            candidates_mu = []
            candidates_sigma = []
            for j, (encode_candidate) in enumerate(self.test_loader):
                par_center, par_delta = self.model.par_projection_box(
                    encode_candidate)
                candidate_mu, candidate_sigma = self.model.compute_gaussian(
                    par_center, par_delta)
                candidates_mu.append(candidate_mu)
                candidates_sigma.append(candidate_sigma)
            candidates_mu = torch.cat(candidates_mu, 0)
            candidates_sigma = torch.cat(candidates_sigma, 0)
            num_query = query_mu.size(0)

            for i in tqdm(range(num_query), desc="Evaluating queries"):
                q_mu = query_mu[i].unsqueeze(0).expand(
                    candidates_mu.size(0), -1)
                q_sigma = query_sigma[i].unsqueeze(
                    0).expand(candidates_mu.size(0), -1, -1)
                score = self.model.bhattacharyya_coefficient(
                    q_mu, q_sigma, candidates_mu, candidates_sigma)
                score_kl = - \
                    self.model.kl_divergence_gaussians(
                        q_mu, q_sigma, candidates_mu, candidates_sigma)
                score_list.append(score)
                score_list_kl.append(score_kl)

            score_matrix = torch.stack(score_list, dim=0)
            sorted_scores, indices = score_matrix.sort(dim=1, descending=True)
            score_matrix_kl = torch.stack(score_list_kl, dim=0)
            sorted_scores_kl, indices_kl = score_matrix_kl.sort(
                dim=1, descending=True)

            print(sorted_scores[:, :5])
            print(sorted_scores_kl[:, :5])

            print("\n--- Generating Plots and Detailed CSV for Top 10 Predictions ---")
            top_10_indices_bc = indices[:, :10]
            top_10_scores_bc = sorted_scores[:, :10]

            results_dir = f'../results/{self.args.dataset}'
            plot_dir = os.path.join(results_dir, 'query_plots')
            csv_path = os.path.join(results_dir, 'case_study_predictions.csv')
            os.makedirs(plot_dir, exist_ok=True)

            candidate_id_list = np.array(
                sorted(list(self.test_set.true_concept_set)))

            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                header = ['Query', 'GroundTruths']
                header.extend(['query_volume', 'gt_volumes',
                              'pred_1_volume', 'GT_Vol_Check', 'Pred_Vol_Check'])
                for k in range(1, 11):
                    header.extend([f'Prediction_{k}', f'Score_{k}'])
                    if k <= 3:
                        header.extend(
                            [f'pred{k}_mu_0', f'pred{k}_mu_1', f'pred{k}_std_0', f'pred{k}_std_1'])
                csv_writer.writerow(header)

                for i in range(num_query):
                    query_concept_id = self.test_set.test_concepts_id[i]
                    query_name = self.test_set.id_concept.get(
                        query_concept_id, f"ID_{query_concept_id}")
                    q_mu, q_sigma = query_mu[i], query_sigma[i]

                    device = q_sigma.device
                    epsilon = 1e-9

                    q_sigma_stable = q_sigma + \
                        torch.eye(q_sigma.shape[0], device=device) * epsilon
                    query_volume = torch.sqrt(torch.det(q_sigma_stable)).item()

                    gt_ids_for_query = gt_label[i]
                    gt_names = [self.test_set.id_concept.get(
                        gid, f"ID_{gid}") for gid in gt_ids_for_query]
                    gt_names_str = "; ".join(gt_names)

                    gt_volumes, gt_containment_checks = [], []
                    for gt_id in gt_ids_for_query:
                        gt_idx_tuple = np.where(candidate_id_list == gt_id)
                        if gt_idx_tuple[0].size > 0:
                            gt_sigma_i = candidates_sigma[gt_idx_tuple[0][0]]
                            gt_sigma_stable = gt_sigma_i + \
                                torch.eye(
                                    gt_sigma_i.shape[0], device=device) * epsilon
                            gt_vol = torch.sqrt(
                                torch.det(gt_sigma_stable)).item()
                            gt_volumes.append(f"{gt_vol:.4e}")
                            gt_containment_checks.append(
                                '✓' if gt_vol > query_volume else '✗')
                        else:
                            gt_volumes.append('N/A')
                            gt_containment_checks.append('?')

                    pred_1_index = top_10_indices_bc[i, 0].item()
                    pred_1_sigma = candidates_sigma[pred_1_index]
                    pred_1_sigma_stable = pred_1_sigma + \
                        torch.eye(
                            pred_1_sigma.shape[0], device=device) * epsilon
                    pred_1_volume = torch.sqrt(
                        torch.det(pred_1_sigma_stable)).item()
                    pred_containment_check = '✓' if pred_1_volume > query_volume else '✗'

                    csv_row = [
                        query_name, gt_names_str,
                        f"{query_volume:.4e}", "; ".join(
                            gt_volumes), f"{pred_1_volume:.4e}",
                        " ".join(gt_containment_checks), pred_containment_check
                    ]

                    top_10_indices = top_10_indices_bc[i, :].cpu().numpy()
                    top_10_ids = candidate_id_list[top_10_indices]

                    for j in range(len(top_10_indices)):
                        pred_index = top_10_indices[j]
                        pred_name = self.train_set.id_concept.get(
                            top_10_ids[j], f"ID_{top_10_ids[j]}")
                        pred_score = top_10_scores_bc[i, j].item()
                        csv_row.extend([pred_name, f"{pred_score:.5f}"])

                        if j < 3:
                            p_mu, p_sigma = candidates_mu[pred_index], candidates_sigma[pred_index]
                            mu_2d = p_mu.cpu().numpy()[:2]
                            stds_2d = np.sqrt(
                                np.diag(p_sigma.cpu().numpy())[:2])
                            csv_row.extend(
                                [f"{mu_2d[0]:.3f}", f"{mu_2d[1]:.3f}", f"{stds_2d[0]:.3f}", f"{stds_2d[1]:.3f}"])

                    csv_writer.writerow(csv_row)

                    top_3_mus = [candidates_mu[idx]
                                 for idx in top_10_indices[:3]]
                    top_3_sigmas = [candidates_sigma[idx]
                                    for idx in top_10_indices[:3]]
                    top_3_names = [self.train_set.id_concept.get(
                        pid, f"ID_{pid}") for pid in top_10_ids[:3]]

                    extra_gts = []
                    for gt_id in gt_ids_for_query:
                        if gt_id not in top_10_ids[:3]:
                            gt_idx_tuple = np.where(candidate_id_list == gt_id)
                            if gt_idx_tuple[0].size > 0:
                                gt_idx = gt_idx_tuple[0][0]
                                extra_gts.append(
                                    (candidates_mu[gt_idx], candidates_sigma[gt_idx], self.test_set.id_concept.get(gt_id)))

                    save_path = os.path.join(
                        plot_dir, f'query_{query_name.replace(" ", "_")}.png')
                    self.plot_gaussians(q_mu, q_sigma, top_3_mus, top_3_sigmas,
                                        query_name, top_3_names, save_path,
                                        extra_gts=extra_gts)

            print(f"Saved {num_query} plots to {plot_dir}")
            print(
                f"Saved detailed prediction results with volume analysis to {csv_path}\n")

            candidate_list = np.array(
                list(self.test_set.true_concept_set))
            test_metrics = metrics(
                self.args,
                indices,
                gt_label,
                candidate_list,
                self.test_set.id_concept,
                self.test_set.test_concepts_id,
            )

            test_metrics_kl = metrics(
                self.args,
                indices_kl,
                gt_label,
                candidate_list,
                self.train_set.id_concept,
                self.test_set.test_concepts_id

            )

            print('BC_Hit@1:{:.05f}'.format(test_metrics["Prec@1"]),
                  'BC_mrr:{:.05f}'.format(test_metrics["MRR"]),
                  'BC_Recall@1:{:.05f}'.format(test_metrics["Recall@1"]),
                  'BC_mr:{:.05f}'.format(test_metrics["MR"]),
                  'BC_@5:{:.05f}'.format(test_metrics["Prec@5"]),
                  'BC_@10:{:.05f}'.format(test_metrics["Prec@10"]),
                  'BC_Recall@5:{:.05f}'.format(test_metrics["Recall@5"]),
                  'BC_Recall@10: {:.05f}'.format(test_metrics["Recall@10"]))

            print('KL_Hit@1:{:.05f}'.format(test_metrics_kl["Prec@1"]),
                  'KL_mrr:{:.05f}'.format(test_metrics_kl["MRR"]),
                  'KL_Recall@1:{:.05f}'.format(test_metrics_kl["Recall@1"]),
                  'KL_mr:{:.05f}'.format(test_metrics_kl["MR"]),
                  'KL_Hit@5:{:.05f}'.format(test_metrics_kl["Prec@5"]),
                  'KL_Hit@10:{:.05f}'.format(test_metrics_kl["Prec@10"]),
                  'KL_Recall@5:{:.05f}'.format(test_metrics_kl["Recall@5"]),
                  'KL_Recall@10: {:.05f}'.format(test_metrics_kl["Recall@10"]))

        with open(f'../results/{self.args.dataset}/res_{self.exp_setting}.json', 'a+') as f:
            d = vars(self.args)
            expt_details = {
                "Arguments": d,
                "Test Metrics BC": test_metrics,
                "Test Metrics KL": test_metrics_kl
            }
            json.dump(expt_details, f, indent=4)

        return test_metrics, test_metrics_kl

    def predict(self, tag=None, path=None):
        """
        Evaluates the model on the test dataset.

        Computes the similarity scores (Bhattacharyya and KL) between test queries 
        and all candidate parents. Calculates standard ranking metrics.

        Args:
            tag (str, optional): If 'test', forces loading weights from `path`.
            path (str, optional): Path to a checkpoint file.

        Returns:
            tuple: (test_metrics, test_metrics_kl) containing Precision, Recall, MRR, Wu-Palmer scores.
        """
        print("Prediction starting.....")
        if (tag == "test"):
            if path:
                self.model.load_state_dict(
                    torch.load(path))
        self.model.eval()
        with torch.no_grad():
            score_list = []
            score_list_kl = []
            gt_label = self.test_set.test_gt_id

            query_center, query_delta = self.model.child_projection_box(
                self.test_set.encode_query)
            query_mu, query_sigma = self.model.compute_gaussian(
                query_center, query_delta)
            candidates_mu = []
            candidates_sigma = []

            for j, (encode_candidate) in enumerate(self.test_loader):
                par_center, par_delta = self.model.par_projection_box(
                    encode_candidate)
                candidate_mu, candidate_sigma = self.model.compute_gaussian(
                    par_center, par_delta)
                candidates_mu.append(candidate_mu)
                candidates_sigma.append(candidate_sigma)
            candidates_mu = torch.cat(candidates_mu, 0)
            candidates_sigma = torch.cat(candidates_sigma, 0)
            num_query = query_mu.size(0)
            num_candidate = candidates_mu.size(0)

            for i in tqdm(range(num_query), desc="Evaluating queries"):
                q_mu = query_mu[i].unsqueeze(0).expand(num_candidate, -1)
                q_sigma = query_sigma[i].unsqueeze(
                    0).expand(num_candidate, -1, -1)

                score = self.model.bhattacharyya_coefficient(
                    q_mu, q_sigma, candidates_mu, candidates_sigma)
                score_kl = -self.model.kl_divergence_gaussians(
                    q_mu, q_sigma, candidates_mu, candidates_sigma)

                score_list.append(score)
                score_list_kl.append(score_kl)

            score_matrix = torch.stack(score_list, dim=0)
            sorted_scores, indices = score_matrix.sort(dim=1, descending=True)

            print(sorted_scores[:, :5])
            score_matrix_kl = torch.stack(score_list_kl, dim=0)
            sorted_scores_kl, indices_kl = score_matrix_kl.sort(
                dim=1, descending=True)

            print(sorted_scores_kl[:, :5])

            if self.args.is_multiparent is True:
                candidate_list = np.array(
                    list(self.test_set.true_concept_set))
                test_metrics = metrics(
                    self.args,
                    indices,
                    gt_label,
                    candidate_list,
                    self.test_set.id_concept,
                    self.test_set.test_concepts_id,
                )

                test_metrics_kl = metrics(
                    self.args,
                    indices_kl,
                    gt_label,
                    candidate_list,
                    self.train_set.id_concept,
                    self.test_set.test_concepts_id

                )

                print('BC_Hit@1:{:.05f}'.format(test_metrics["Prec@1"]),
                      'BC_mrr:{:.05f}'.format(test_metrics["MRR"]),
                      'BC_Recall@1:{:.05f}'.format(test_metrics["Recall@1"]),
                      'BC_mr:{:.05f}'.format(test_metrics["MR"]),
                      'BC_@5:{:.05f}'.format(test_metrics["Prec@5"]),
                      'BC_@10:{:.05f}'.format(test_metrics["Prec@10"]),
                      'BC_Recall@5:{:.05f}'.format(test_metrics["Recall@5"]),
                      'BC_Recall@10: {:.05f}'.format(test_metrics["Recall@10"]))

                print('KL_Hit@1:{:.05f}'.format(test_metrics_kl["Prec@1"]),
                      'KL_mrr:{:.05f}'.format(test_metrics_kl["MRR"]),
                      'KL_Recall@1:{:.05f}'.format(
                          test_metrics_kl["Recall@1"]),
                      'KL_mr:{:.05f}'.format(test_metrics_kl["MR"]),
                      'KL_Hit@5:{:.05f}'.format(test_metrics_kl["Prec@5"]),
                      'KL_Hit@10:{:.05f}'.format(test_metrics_kl["Prec@10"]),
                      'KL_Recall@5:{:.05f}'.format(
                          test_metrics_kl["Recall@5"]),
                      'KL_Recall@10: {:.05f}'.format(test_metrics_kl["Recall@10"]))
            else:
                test_metrics = metrics_single_parent(indices, gt_label, self.train_set.train_concept_set, self.test_set.path2root,
                                                     self.test_set.id_concept, self.train_set.id_concept, self.test_set.test_concepts_id, sorted_scores)
                test_metrics_kl = metrics_single_parent(
                    indices_kl,
                    gt_label,
                    self.train_set.train_concept_set,
                    self.test_set.path2root,
                    self.test_set.id_concept,
                    self.train_set.id_concept,
                    self.test_set.test_concepts_id,
                    sorted_scores
                )

                print('BC_Hit@1:{:.05f}'.format(test_metrics["Prec@1"]),
                      'BC_mrr:{:.05f}'.format(test_metrics["MRR"]),
                      'BC_Recall@1:{:.05f}'.format(test_metrics["Recall@1"]),
                      'BC_mr:{:.05f}'.format(test_metrics["MR"]),
                      'BC_@5:{:.05f}'.format(test_metrics["Prec@5"]),
                      'BC_@10:{:.05f}'.format(test_metrics["Prec@10"]),
                      'BC_Recall@5:{:.05f}'.format(test_metrics["Recall@5"]),
                      'BC_Recall@10: {:.05f}'.format(
                          test_metrics["Recall@10"]),
                      'WU_Palmer: {:.05f}'.format(test_metrics['WuP']))

                print('KL_Hit@1:{:.05f}'.format(test_metrics_kl["Prec@1"]),
                      'KL_mrr:{:.05f}'.format(test_metrics_kl["MRR"]),
                      'KL_Recall@1:{:.05f}'.format(
                          test_metrics_kl["Recall@1"]),
                      'KL_mr:{:.05f}'.format(test_metrics_kl["MR"]),
                      'KL_Hit@5:{:.05f}'.format(test_metrics_kl["Prec@5"]),
                      'KL_Hit@10:{:.05f}'.format(test_metrics_kl["Prec@10"]),
                      'KL_Recall@5:{:.05f}'.format(
                          test_metrics_kl["Recall@5"]),
                      'KL_Recall@10: {:.05f}'.format(
                          test_metrics_kl["Recall@10"]),
                      'WU_Palmer: {:.05f}'.format(test_metrics['WuP']))

        with open(f'../results/{self.args.dataset}/res_{self.exp_setting}.json', 'a+') as f:
            d = vars(self.args)
            expt_details = {
                "Arguments": d,
                "Test Metrics BCE": test_metrics,
                "Test Metrics Kl": test_metrics_kl
            }
            json.dump(expt_details, f, indent=4)

        return test_metrics, test_metrics_kl
