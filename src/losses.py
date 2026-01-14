import torch
import torch.nn.functional as F


def kl_divergence_gaussians(mu1, sigma1, mu2, sigma2):
    """
    Computes the Kullback-Leibler (KL) divergence between two multivariate Gaussian distributions.

    The KL divergence D_KL(N1 || N2) measures the information lost when N2 is used to
    approximate N1. It is asymmetric and non-negative.

    Formula:
        0.5 * (tr(Σ2^-1 Σ1) + (μ2 - μ1)^T Σ2^-1 (μ2 - μ1) - d + ln(|Σ2|/|Σ1|))

    Args:
        mu1 (torch.Tensor): Mean vector of the first Gaussian (N1) of shape (batch_size, d).
        sigma1 (torch.Tensor): Covariance matrix of N1 of shape (batch_size, d, d).
        mu2 (torch.Tensor): Mean vector of the second Gaussian (N2) of shape (batch_size, d).
        sigma2 (torch.Tensor): Covariance matrix of N2 of shape (batch_size, d, d).

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) containing the KL divergence for each pair.
    """
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
    """
    Computes a contrastive loss based on KL divergence to enforce containment relationships.

    Objectives:
    1. Minimize D_KL(Child || Parent) (Child should be 'inside' Parent).
    2. Maximize D_KL(Child || Negative Parent) up to a margin (Child should be 'far' from False Parent).

    Args:
        mu_c, sigma_c: Parameters for the Child concept.
        mu_p, sigma_p: Parameters for the Positive Parent.
        mu_n, sigma_n: Parameters for the Negative Parent.
        margin (float): The minimum desired distance (divergence) from negative samples.
        alpha (float): Scaling factor for the positive containment term.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    pos_kl = kl_divergence_gaussians(mu_c, sigma_c, mu_p, sigma_p)
    neg_kl = kl_divergence_gaussians(mu_c, sigma_c, mu_n, sigma_n)

    neg_margin = torch.relu(margin - neg_kl)

    return alpha * pos_kl.mean() + neg_margin.mean()


def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
    """
    Computes the Bhattacharyya distance between two multivariate Gaussians.

    This metric measures the separability of distributions. Unlike KL divergence,
    it is symmetric and satisfies the triangle inequality.

    Args:
        mu1, sigma1: Parameters of the first Gaussian.
        mu2, sigma2: Parameters of the second Gaussian.

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) containing the distance.
    """
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
    """
    Computes the Bhattacharyya Coefficient (BC).

    BC is a measure of the amount of overlap between two statistical samples.
    Range: [0, 1], where 1 indicates complete overlap and 0 indicates no overlap.

    Formula: BC = exp(-D_B)

    Returns:
        torch.Tensor: Tensor of shape (batch_size,).
    """
    return torch.exp(-bhattacharyya_distance(mu1, sigma1, mu2, sigma2))


def gaussian_volume(sigma):
    """
    Estimates the volume of the Gaussian distribution.

    Calculated as the square root of the determinant of the covariance matrix.
    This is proportional to the geometric volume of the confidence ellipsoid.

    Args:
        sigma (torch.Tensor): Covariance matrix of shape (batch_size, d, d).

    Returns:
        torch.Tensor: Tensor of shape (batch_size,).
    """
    d = sigma.size(-1)
    return torch.sqrt(torch.det(sigma + 1e-6 * torch.eye(d).to(sigma.device)))


def gaussian_asymmetry_loss(mu_c, sigma_c, mu_p, sigma_p, mu_n, sigma_n):
    """
    Computes an asymmetry loss to model directional entailment (Hypernymy).

    This loss uses a ratio of Overlap (BC) to Child Volume.
    Ideally:
    - Child should be fully contained in Parent -> BC(c, p) / Vol(c) approx 1.
    - Child should not overlap Negative Parent -> BC(c, n) / Vol(c) approx 0.

    Args:
        mu_c, sigma_c: Child parameters.
        mu_p, sigma_p: Positive Parent parameters.
        mu_n, sigma_n: Negative Parent parameters.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    vol_c = gaussian_volume(sigma_c)

    bc_pos = bhattacharyya_coefficient(mu_c, sigma_c, mu_p, sigma_p)
    p_pos = bc_pos / (vol_c + 1e-8)
    loss_pos = (p_pos - 1.0) ** 2

    bc_neg = bhattacharyya_coefficient(mu_c, sigma_c, mu_n, sigma_n)
    p_neg = bc_neg / (vol_c + 1e-8)
    loss_neg = (p_neg - 0.0) ** 2

    return loss_pos.mean() + loss_neg.mean()


def variance_regularization_loss(sigma, min_std=0.1):
    """
    Regularization to prevent Gaussian collapse (variance becoming zero).

    Penalizes standard deviations that fall below `min_std`.

    Args:
        sigma (torch.Tensor): Covariance matrix.
        min_std (float): Minimum threshold for standard deviation.

    Returns:
        torch.Tensor: Scalar penalty.
    """
    std = torch.sqrt(torch.diagonal(sigma, dim1=-2, dim2=-1))
    penalty = F.relu(min_std - std)
    return penalty.mean()


def child_variance_below_parent_loss(sigma_c, sigma_p):
    """
    Hierarchical regularization enforcing that Child variance <= Parent variance.

    In a taxonomy, specific concepts (children) generally have narrower
    meanings (lower variance) than general concepts (parents).

    Args:
        sigma_c (torch.Tensor): Child covariance.
        sigma_p (torch.Tensor): Parent covariance.

    Returns:
        torch.Tensor: Scalar penalty.
    """
    var_c = torch.diagonal(sigma_c, dim1=-2, dim2=-1)
    var_p = torch.diagonal(sigma_p, dim1=-2, dim2=-1)
    penalty = F.relu(var_c - var_p)
    return penalty.mean()


def combined_gaussian_loss(mu_c, sigma_c, mu_p, sigma_p, mu_n, sigma_n,
                           margin=5.0, alpha=1.0, beta=1.0, reg_weight=0.1):
    """
    Composite loss function combining KL divergence, Asymmetry (overlap),
    and Geometric Regularization.

    Args:
        mu_c, sigma_c: Child parameters.
        mu_p, sigma_p: Positive Parent parameters.
        mu_n, sigma_n: Negative Parent parameters.
        margin (float): Margin for KL hinge loss.
        alpha (float): Weight for positive KL term.
        beta (float): Weight for asymmetry loss.
        reg_weight (float): Weight for variance regularization terms.

    Returns:
        torch.Tensor: The total combined loss.
    """
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
