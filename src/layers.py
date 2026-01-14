import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MLP_Large(nn.Module):
    """
    A 3-layer Multi-Layer Perceptron (Deep Neural Network) with ReLU activations.

    Structure:
        Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
    """

    def __init__(self, input_dim, hidden, output_dim):
        """
        Args:
            input_dim (int): Size of the input feature vector.
            hidden (int): Size of the hidden layers.
            output_dim (int): Size of the output feature vector.
        """
        super(MLP_Large, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x


class MLP(nn.Module):
    """
    A 2-layer Multi-Layer Perceptron.

    Note: Although `fc2` is defined in __init__, it is skipped in the forward pass 
    in this implementation. The structure is effectively:
    Input -> Linear -> ReLU -> Linear -> Output.
    """

    def __init__(self, input_dim, hidden, output_dim):
        """
        Args:
            input_dim (int): Size of the input feature vector.
            hidden (int): Size of the hidden layer.
            output_dim (int): Size of the output feature vector.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        # Defined but not used in forward
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = self.fc1(x)
        x = F.relu(x)
        # self.fc2 is skipped here
        x = self.fc3(x)
        return x


class MLP_VEC(nn.Module):
    """
    A Deep Multi-Layer Perceptron (4 layers) ending with a Sigmoid activation.

    This is typically used for generating gating mechanisms, probabilities, 
    or normalized coefficients (output range [0, 1]).
    """

    def __init__(self, input_dim, hidden, output_dim):
        """
        Args:
            input_dim (int): Size of the input feature vector.
            hidden (int): Size of the hidden layers.
            output_dim (int): Size of the output feature vector.
        """
        super(MLP_VEC, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.nlnr = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc25 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x, printit=False):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            printit (bool, optional): Unused argument kept for API compatibility.

        Returns:
            torch.Tensor: Output tensor with values in range [0, 1].
        """
        x = self.fc1(x)
        x = self.nlnr(x)
        x = self.fc2(x)
        x = self.nlnr(x)
        x = self.fc25(x)
        x = self.nlnr(x)
        x = self.fc3(x)
        x = self.sig(x)
        return x


class NormalisedWeights(nn.Module):
    """
    Module to generate normalized mixing weights for mixture models.

    Supports various initialization strategies (Random, Constant, Uniform, Self-Attention)
    and ensures output weights sum to 1 via Softmax.
    """

    def __init__(self, input_dim, mixturetype=None, modeltype=None):
        """
        Args:
            input_dim (int): The number of components to generate weights for.
            mixturetype (str, optional): Strategy for weight initialization 
                                         ('random', 'constant', 'uniform', 'self_attention').
            modeltype (str, optional): If 'complex', applies sqrt to weights for complex space projections.
        """
        super(NormalisedWeights, self).__init__()
        self.type = mixturetype
        self.input_dim = input_dim
        self.valspace = "complex" if modeltype else "real"

        if mixturetype == "random":
            self.weights = nn.Parameter(torch.randn(input_dim))
        elif mixturetype == "constant":
            self.weights = torch.ones(input_dim).cuda()
        elif mixturetype == "uniform":
            self.weights = nn.Parameter(torch.empty(input_dim).uniform_(-1, 1))
        elif mixturetype == "self_attention":
            # Note: self.args is expected to be injected or defined globally for this branch
            self.weights = nn.Parameter(torch.empty(
                (input_dim//2, self.args.matrixsize)))
            self.a = nn.Parameter(torch.empty(input_dim))

        self.fc1 = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Generates normalized weights.

        Args:
            x (torch.Tensor): Input tensor. Used primarily in the 'self_attention' mode 
                              to compute data-dependent weights.

        Returns:
            torch.Tensor: Normalized weights summing to 1.
        """
        if self.type == "self_attention":
            # Experimental block for attention-based dynamic weighting
            weight_mat = self.weights.T
            print("Wt mat:", weight_mat.shape)
            a = self.a
            print("a param: ", a.shape)

            # Compute attention scores
            weighted_states = torch.matmul(x, weight_mat)
            print("Wt st:", weighted_states.shape)

            # Pairwise combinations
            mat1 = weighted_states.unsqueeze(-2).repeat(1,
                                                        1, self.input_dim, 1)
            print("M1", mat1.shape)
            mat2 = weighted_states.unsqueeze(-3).repeat(1,
                                                        self.input_dim, 1, 1)
            print("M2: ", mat2.shape)
            w_concat = torch.cat([mat1, mat2], dim=-1)
            print("W cat: ", w_concat.shape)

            # Project and Activate
            w_concat = torch.matmul(w_concat, a)
            print("W cat2: ", w_concat.shape)
            # Note: lrelu must be defined in parent context or init
            w_concat = self.lrelu(w_concat)
            print("W cat3: ", w_concat.shape)

            weights = self.fc1(w_concat)
            print("Weight: ", weights.shape)
            weighted_sum_batch = torch.matmul(weights, weighted_states)
            print(weighted_sum_batch.shape)
            # This branch seems to return intermediate prints currently;
            # usually returns weights or weighted_sum_batch.
        elif self.type == "nn":
            weights = self.ann(self.weights)  # Note: ann must be defined
            weights = self.fc1(weights)
        else:
            # Standard static learned weights
            weights = self.fc1(self.weights)
            # Ensure dimension matching if x is involved in validation
            assert (x.shape[1] == weights.shape[0])

            if self.valspace == "complex":
                weights = torch.sqrt(weights)

        return weights


class LINEAR_ONE(nn.Module):
    """
    Applies a linear transformation derived from the input onto the outer product of the input.

    Effectively calculates: Linear(OuterProduct(x, x), Softmax(Linear(x)))
    """

    def __init__(self, input_dim):
        super(LINEAR_ONE, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc2 = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Result of the linear mapping on the outer product of x.
        """
        y = self.fc1(x)
        y = self.fc2(y)      # Generate weights/coefficients
        x = torch.outer(x, x)  # Compute outer product (relationship map)
        x = F.linear(x, y)    # Apply linear map using y as weights
        return x


class Observation(nn.Module):
    """
    Computes a Quadratic Form observation.

    Given a set of learnable observation vectors (weights) and an input matrix x,
    computes the quadratic projection: w^T * x * w.
    """

    def __init__(self, num_obs=64):
        """
        Args:
            num_obs (int): The number of observation vectors (channels) to learn.
                           Defaults to 64.
        """
        super(Observation, self).__init__()
        # Learnable projection vectors (num_obs, embedding_dim=768)
        self.weights = nn.Parameter(torch.empty(num_obs, 768).uniform_(-1, 1))

    def forward(self, x):
        """
        Applies Einstein Summation to compute quadratic observations.

        Equation: result[b, k] = sum_{i, j} (weights[k, i] * x[b, i, j] * weights[k, j])

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 768, 768).
                              Usually represents a covariance matrix or interaction map.

        Returns:
            torch.Tensor: Observation vector of shape (batch_size, num_obs).
        """
        output = torch.einsum('ki,bij,kj->bk', self.weights, x, self.weights)
        return output

# class GatedTangentFlowBoxMap(nn.Module):
#     def __init__(self, hidden_dim: int, box_dim: int):
#         super().__init__()
#         self.flow_control_projection = nn.Linear(hidden_dim, box_dim)
#         self.scale_projection = nn.Linear(hidden_dim, box_dim)

#         self.center_projection = nn.Linear(hidden_dim, box_dim)

#     def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         center = self.center_projection(h)

#         flow_control_vector = self.flow_control_projection(h)

#         base_offset = torch.sigmoid(torch.tanh(flow_control_vector))

#         scale = torch.exp(base_offset).clamp_min(1e-38)

#         jitter = torch.finfo(h.dtype).eps
#         final_offset = (scale * base_offset) + jitter

#         return center, final_offset
