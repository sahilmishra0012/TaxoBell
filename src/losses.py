import torch
import torch.nn.functional as F


def kl_divergence_gaussians(mu1, sigma1, mu2, sigma2):
    d = mu1.size(-1)
    sigma2_inv = torch.linalg.inv(sigma2)
    trace_term = torch.einsum('bij->b', torch.matmul(sigma2_inv, sigma1))
    diff = (mu2 - mu1).unsqueeze(-1)
    mahal = torch.matmul(torch.matmul(
        diff.transpose(1, 2), sigma2_inv), diff).squeeze()
    log_det_ratio = torch.logdet(sigma2 + 1e-6 * torch.eye(d).to(mu1.device)) - \
        torch.logdet(sigma1 + 1e-6 * torch.eye(d).to(mu1.device))
    return 0.5 * (trace_term + mahal - d + log_det_ratio)


def kl_containment_exclusion_loss(mu_c, sigma_c, mu_p, sigma_p, mu_n, sigma_n, margin=5.0, alpha=1.0):
    pos_kl = kl_divergence_gaussians(mu_c, sigma_c, mu_p, sigma_p)
    neg_kl = kl_divergence_gaussians(mu_c, sigma_c, mu_n, sigma_n)
    neg_margin = torch.relu(margin - neg_kl)
    return alpha * pos_kl.mean() + neg_margin.mean()


def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
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


def bhattacharyya_coefficient(mu1, sigma1, mu2, sigma2):
    return torch.exp(-bhattacharyya_distance(mu1, sigma1, mu2, sigma2))


def gaussian_volume(sigma):
    d = sigma.size(-1)
    return torch.sqrt(torch.det(sigma + 1e-6 * torch.eye(d).to(sigma.device)))


def gaussian_asymmetry_loss(mu_c, sigma_c, mu_p, sigma_p, mu_n, sigma_n):
    vol_c = gaussian_volume(sigma_c)
    bc_pos = bhattacharyya_coefficient(mu_c, sigma_c, mu_p, sigma_p)
    p_pos = bc_pos / (vol_c + 1e-8)
    loss_pos = (p_pos - 1.0) ** 2
    bc_neg = bhattacharyya_coefficient(mu_c, sigma_c, mu_n, sigma_n)
    p_neg = bc_neg / (vol_c + 1e-8)
    loss_neg = (p_neg - 0.0) ** 2
    return loss_pos.mean() + loss_neg.mean()


def combined_gaussian_loss(mu_c, sigma_c, mu_p, sigma_p, mu_n, sigma_n,
                           margin=5.0, alpha=1.0, beta=1.0):
    loss_kl = kl_containment_exclusion_loss(
        mu_c, sigma_c, mu_p, sigma_p, mu_n, sigma_n, margin, alpha)
    loss_asym = gaussian_asymmetry_loss(
        mu_c, sigma_c, mu_p, sigma_p, mu_n, sigma_n)
    return loss_kl + beta * loss_asym


def kl_divergence_gaussians(mu1, sigma1, mu2, sigma2):
    d = mu1.size(-1)
    sigma2_inv = torch.linalg.inv(sigma2)
    trace_term = torch.einsum('bij->b', torch.matmul(sigma2_inv, sigma1))
    diff = (mu2 - mu1).unsqueeze(-1)
    mahal = torch.matmul(torch.matmul(
        diff.transpose(1, 2), sigma2_inv), diff).squeeze()
    log_det_ratio = torch.logdet(sigma2 + 1e-6 * torch.eye(d).to(mu1.device)) - \
        torch.logdet(sigma1 + 1e-6 * torch.eye(d).to(mu1.device))
    return 0.5 * (trace_term + mahal - d + log_det_ratio)


def kl_containment_exclusion_loss(mu_c, sigma_c, mu_p, sigma_p, mu_n, sigma_n, margin=5.0, alpha=1.0):
    pos_kl = kl_divergence_gaussians(mu_c, sigma_c, mu_p, sigma_p)
    neg_kl = kl_divergence_gaussians(mu_c, sigma_c, mu_n, sigma_n)
    neg_margin = torch.relu(margin - neg_kl)
    return alpha * pos_kl.mean() + neg_margin.mean()


def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
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


def bhattacharyya_coefficient(mu1, sigma1, mu2, sigma2):
    return torch.exp(-bhattacharyya_distance(mu1, sigma1, mu2, sigma2))


def gaussian_volume(sigma):
    d = sigma.size(-1)
    return torch.sqrt(torch.det(sigma + 1e-6 * torch.eye(d).to(sigma.device)))


def gaussian_asymmetry_loss(mu_c, sigma_c, mu_p, sigma_p, mu_n, sigma_n):
    vol_c = gaussian_volume(sigma_c)
    bc_pos = bhattacharyya_coefficient(mu_c, sigma_c, mu_p, sigma_p)
    p_pos = bc_pos / (vol_c + 1e-8)
    loss_pos = (p_pos - 1.0) ** 2
    bc_neg = bhattacharyya_coefficient(mu_c, sigma_c, mu_n, sigma_n)
    p_neg = bc_neg / (vol_c + 1e-8)
    loss_neg = (p_neg - 0.0) ** 2
    return loss_pos.mean() + loss_neg.mean()


def variance_regularization_loss(sigma, min_std=0.1):
    std = torch.sqrt(torch.diagonal(sigma, dim1=-2, dim2=-1))
    penalty = F.relu(min_std - std)
    return penalty.mean()


def child_variance_below_parent_loss(sigma_c, sigma_p):
    var_c = torch.diagonal(sigma_c, dim1=-2, dim2=-1)
    var_p = torch.diagonal(sigma_p, dim1=-2, dim2=-1)
    penalty = F.relu(var_c - var_p)
    return penalty.mean()


def combined_gaussian_loss(mu_c, sigma_c, mu_p, sigma_p, mu_n, sigma_n,
                           margin=5.0, alpha=1.0, beta=1.0, reg_weight=0.1):
    loss_kl = kl_containment_exclusion_loss(
        mu_c, sigma_c, mu_p, sigma_p, mu_n, sigma_n, margin, alpha)
    loss_asym = gaussian_asymmetry_loss(
        mu_c, sigma_c, mu_p, sigma_p, mu_n, sigma_n)
    var_reg = (
        variance_regularization_loss(sigma_c) +
        variance_regularization_loss(sigma_p) +
        variance_regularization_loss(sigma_n)
    )
    var_hier = child_variance_below_parent_loss(sigma_c, sigma_p)
    return loss_kl + beta * loss_asym + reg_weight * (var_reg + var_hier)
