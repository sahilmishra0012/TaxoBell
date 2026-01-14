import os
import pickle as pkl
import torch
import numpy as np
import sys
import torch.nn as nn
from utils import *
from layers import MLP, MLP_Large
from transformers import BertModel, AutoModel
import torch.nn.functional as F


class GaussianBox(nn.Module):
    def __init__(self, args):
        super(GaussianBox, self).__init__()

        self.args = args
        # self.data = self.__load_data__(self.args.dataset)
        # self.FloatTensor = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
        # self.concept_set = self.data["concept_set"]
        # self.concept_id = self.data["concept2id"]
        # self.id_concept = self.data["id2concept"]
        # self.id_context = self.data["id2context"]

        # self.train_concept_set = list(self.data["train_concept_set"])
        # self.train_taxo_dict = self.data["train_taxo_dict"]
        # self.train_child_parent_negative_parent_triple = self.data[
        #     "train_child_parent_negative_parent_triple"]
        # self.path2root = self.data["path2root"]
        # self.test_concepts_id = self.data["test_concepts_id"]
        # self.test_gt_id = self.data["test_gt_id"]

        self.pre_train_model = self.__load_pre_trained__()

        self.par_projection_center = MLP(
            input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size)
        self.par_projection_delta = MLP(
            input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size)

        self.child_projection_center = MLP(
            input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size)
        self.child_projection_delta = MLP(
            input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size)

        self.gate_par_delta = MLP(
            input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size)
        self.gate_child_delta = MLP(
            input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size)

        self.parent_small_offset = MLP(
            input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size)
        self.parent_large_offset = MLP_Large(
            input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size)

        self.child_small_offset = MLP(
            input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size)
        self.child_large_offset = MLP_Large(
            input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size)

        # self.parent_gated_flow = GatedTangentFlowBoxMap(
        #     hidden_dim=self.args.embed_size, box_dim=self.args.embed_size*2)
        # self.child_gated_flow = GatedTangentFlowBoxMap(
        #     hidden_dim=self.args.embed_size, box_dim=self.args.embed_size*2)
        self.parent_mlp = MLP(
            input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size)
        self.child_mlp = MLP(
            input_dim=768, output_dim=self.args.embed_size, hidden=self.args.hidden)

        self.dropout = nn.Dropout(self.args.dropout)

    # def __load_data__(self, dataset):
    #     with open(os.path.join("../data/", dataset, "processed", "taxonomy_data_"+str(self.args.expID)+"_.pkl"), "rb") as f:
    #         data = pkl.load(f)

    #     return data

    def __load_pre_trained__(self):
        if self.args.model == 'bert':
            model = BertModel.from_pretrained(
                '/home/models/bert-base-uncased')
        elif self.args.model == 'snowflake':
            model = AutoModel.from_pretrained(
                'Snowflake/snowflake-arctic-embed-m', add_pooling_layer=False)
        elif self.args.model == 'e5':
            model = AutoModel.from_pretrained('/home/models/e5-large-v2')
        print("Model Loaded!")
        return model

    def get_cls(self, encode_inputs):
        if self.args.model == 'snowflake':
            cls_embed = self.pre_train_model(
                **encode_inputs).last_hidden_state[:, 0]
        elif self.args.model == 'bert':
            outputs = self.pre_train_model(**encode_inputs)
            last_hidden_state = outputs.last_hidden_state

            cls_embed = last_hidden_state[:, 0, :]
        elif self.args.model == 'e5':
            outputs = self.pre_train_model(**encode_inputs)
            # last_hidden=outputs.last_hidden_states.masked_fill(encode_inputs['attention_mask'])
            last_hidden_states = outputs.last_hidden_state
            attention_mask = encode_inputs['attention_mask']
            last_hidden = last_hidden_states.masked_fill(
                ~attention_mask[..., None].bool(), 0.0)
            cls_embed = last_hidden.sum(
                dim=1) / attention_mask.sum(dim=1)[..., None]
        # attention_mask = encode_inputs['attention_mask']
        # last_hidden_state = outputs.last_hidden_state
        # # mask_expanded = attention_mask.squeeze(
        # #     -1).expand(last_hidden_state.size())
        # # sum_embeddings = torch.sum(last_hidden_state, -1)
        # # sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        # # cls_embed = sum_embeddings/sum_mask
        # last_hidden = last_hidden_state.masked_fill(
        #     ~attention_mask[..., None].bool(), 0.0)
        # cls_embed = last_hidden.sum(dim=1)/attention_mask.sum(dim=1)[..., None]

        return cls_embed

    def par_projection_box(self, cls_embed):
        if self.args.method == 'normal':
            center = self.child_projection_center(self.get_cls(cls_embed))
            delta = self.child_projection_delta(self.get_cls(cls_embed))
            delta = torch.exp(delta).clamp_min(1e-38)
        elif self.args.method == 'gated':
            center = self.child_projection_center(self.get_cls(cls_embed))
            gate = F.sigmoid(self.gate_child_delta(self.get_cls(cls_embed)))

            w_small = self.child_small_offset(self.get_cls(cls_embed))
            w_large = self.child_large_offset(self.get_cls(cls_embed))

            delta = torch.exp(gate*w_large+(1-gate)*w_small).clamp_min(1e-38)

        return center, delta

    def child_projection_box(self, cls_embed):

        if self.args.method == 'normal':
            center = self.child_projection_center(self.get_cls(cls_embed))
            delta = self.child_projection_delta(self.get_cls(cls_embed))
            delta = torch.exp(delta).clamp_min(1e-38)
        elif self.args.method == 'gated':
            center = self.child_projection_center(self.get_cls(cls_embed))
            gate = F.sigmoid(self.gate_child_delta(self.get_cls(cls_embed)))

            w_small = self.child_small_offset(self.get_cls(cls_embed))
            w_large = self.child_large_offset(self.get_cls(cls_embed))

            delta = torch.exp(gate*w_large+(1-gate)*w_small).clamp_min(1e-38)

        return center, delta

    def mean_regularization(self, mu):
        return torch.norm(mu, dim=1).mean()

    def clip_covariance(self, sigma, cmin=1.0, cmax=10.0):
        diag = torch.diagonal(sigma, dim1=-2, dim2=-1)
        below = F.relu(cmin - diag)
        above = F.relu(diag - cmax)
        return (below**2 + above**2).mean()

    def minimum_volume_regularization(self, sigma, threshold=0.5):
        std = torch.sqrt(torch.diagonal(sigma, dim1=-2, dim2=-1))
        penalty = F.relu(threshold - std)
        return penalty.pow(2).mean()

    def compute_gaussian(self, center, offset):
        """
        Compute mean (mu) and covariance (sigma) from center and offset.
        """
        mu = center
        sigma = torch.diag_embed(offset ** 2)

        return mu, sigma

    def bhattacharyya_distance(self, mu1, sigma1, mu2, sigma2):
        d = mu1.size(-1)
        sigma = 0.5 * (sigma1 + sigma2)
        sigma_inv = torch.linalg.inv(sigma)
        diff = mu1 - mu2
        term1 = 0.125 * torch.einsum('bi,bij,bj->b', diff, sigma_inv, diff)
        det_sigma = torch.logdet(sigma + 1e-6 * torch.eye(d).to(mu1.device))
        det_1 = torch.logdet(sigma1 + 1e-6 * torch.eye(d).to(mu1.device))
        det_2 = torch.logdet(sigma2 + 1e-6 * torch.eye(d).to(mu1.device))
        term2 = 0.5 * (det_sigma - 0.5 * (det_1 + det_2))
        return term1 + term2

    def bhattacharyya_coefficient(self, mu1, sigma1, mu2, sigma2):
        return torch.exp(-self.bhattacharyya_distance(mu1, sigma1, mu2, sigma2))

    def stochastic_containment_loss(
        self,
        mu_c: torch.Tensor,
        cov_c: torch.Tensor,
        mu_p: torch.Tensor,
        cov_p: torch.Tensor,
    ):

        sigma_p_inv = torch.linalg.inv(cov_p)
        trace_term = torch.einsum(
            'bij,bij->b', sigma_p_inv, cov_c)
        mean_term = (mu_p-mu_c).unsqueeze(-1)

        mahalanbois = torch.matmul(torch.matmul(
            mean_term.transpose(1, 2), sigma_p_inv), mean_term).squeeze()

        return trace_term+mahalanbois

    def kl_divergence_gaussians(self, mu_c, sigma_c, mu_p, sigma_p):
        """
        Computes KL divergence D_KL(N_c || N_p) for batched Gaussians
        mu_c, sigma_c: child (query)
        mu_p, sigma_p: parent (anchor)
        """
        d = mu_c.size(-1)

        sigma_p_inv = torch.linalg.inv(sigma_p)

        trace_term = torch.einsum('bij,bij->b', sigma_p_inv, sigma_c)

        diff = (mu_p - mu_c).unsqueeze(-1)
        mahalanobis = torch.matmul(torch.matmul(
            diff.transpose(1, 2), sigma_p_inv), diff).squeeze()

        log_det_p = torch.logdet(sigma_p + 1e-8 * torch.eye(d).to(mu_c.device))
        log_det_c = torch.logdet(sigma_c + 1e-8 * torch.eye(d).to(mu_c.device))
        log_det_ratio = log_det_p - log_det_c

        kl = 0.5 * (trace_term + mahalanobis - d + log_det_ratio)
        return kl

    def child_volume(self, sigma_c, eps=1e-6):

        d = sigma_c.size(-1)
        eye = torch.eye(d, device=sigma_c.dev̦ice).unsqueeze(0)
        sigma_stable = sigma_c + eps * eye

        log_det = torch.linalg.slogdet(sigma_stable)[1]
        volume = torch.exp(0.5 * log_det)
        return volume

    def log_det_term(self, sigma, eps=1e-6):
        d = sigma.size(-1)
        eye = torch.eye(d, device=sigma.device).unsqueeze(0)

        sigma_stable = sigma+eps*eye
        log_det = torch.linalg.slogdet(sigma_stable)[1]

        return log_det

    def intersection_distribution(self, sigma_c, sigma_p, mu_c, mu_p):
        sigma_c_inv = torch.linalg.inv(sigma_c)
        sigma_p_inv = torch.linalg.inv(sigma_p)
        mu_c = mu_c.view(mu_c.size(0), mu_c.size(1), 1)
        mu_p = mu_p.view(mu_p.size(0), mu_p.size(1), 1)

        intersection_sigma = torch.linalg.inv(
            torch.add(sigma_c_inv, sigma_p_inv))

        intersection_mean = intersection_sigma @ (
            (sigma_c_inv@mu_c)+(sigma_p_inv@mu_p))

        intersection_mean = intersection_mean.view(
            intersection_mean.size(0), intersection_mean.size(1))

        return intersection_mean, intersection_sigma

    def forward(self, encode_parent, encode_child, encode_negative_parents, flag="trace"):

        par_center, par_delta = self.par_projection_box(encode_parent)
        par_mu, par_sigma = self.compute_gaussian(par_center, par_delta)

        child_center, child_delta = self.child_projection_box(encode_child)
        child_mu, child_sigma = self.compute_gaussian(
            child_center, child_delta)

        neg_par_center, neg_par_delta = self.par_projection_box(
            encode_negative_parents)
        neg_par_mu, neg_par_sigma = self.compute_gaussian(
            neg_par_center, neg_par_delta)

        log_det_parent = self.log_det_term(par_sigma)
        log_det_child = self.log_det_term(child_sigma)
        C = self.args.C

        dynamic = C*(log_det_parent-log_det_child)
        pos_kld = self.kl_divergence_gaussians(
            child_mu, child_sigma, par_mu, par_sigma)
        pos_kld_other = self.kl_divergence_gaussians(
            par_mu, par_sigma, child_mu, child_sigma)
        neg_kld = self.kl_divergence_gaussians(
            child_mu, child_sigma, neg_par_mu, neg_par_sigma)

        kl_containment = dynamic-pos_kld_other
        margin = 0.5
        lam = self.args.lam

        kl_loss = F.relu(pos_kld+margin-neg_kld+lam*kl_containment).mean()

        pos_bc = self.bhattacharyya_coefficient(
            par_mu, par_sigma, child_mu, child_sigma)
        neg_bc = self.bhattacharyya_coefficient(
            neg_par_mu, neg_par_sigma, child_mu, child_sigma)

        bc_loss = (-torch.log(pos_bc) - torch.log(1 - neg_bc)).mean()

        l1 = self.minimum_volume_regularization(
            par_sigma) + self.clip_covariance(par_sigma)
        l2 = self.minimum_volume_regularization(
            child_sigma) + self.clip_covariance(child_sigma)
        l3 = self.minimum_volume_regularization(
            neg_par_sigma) + self.clip_covariance(neg_par_sigma)

        return self.args.wtbce*bc_loss+self.args.wtkl*kl_loss + self.args.wtreg*(l1+l2+l3)
