import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MLP_Large(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(MLP_Large, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class MLP_VEC(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(MLP_VEC, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.nlnr = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc25 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x, printit=False):
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
    def __init__(self, input_dim, mixturetype=None, modeltype=None):
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
            self.weights = nn.Parameter(torch.empty(
                (input_dim//2, self.args.matrixsize)))
            self.a = nn.Parameter(torch.empty(input_dim))
        self.fc1 = nn.Softmax(dim=-1)

    def forward(self, x):
        if self.type == "self_attention":
            weight_mat = self.weights.T
            print("Wt mat:", weight_mat.shape)
            a = self.a
            print("a param: ", a.shape)
            weighted_states = torch.matmul(x, weight_mat)
            print("Wt st:", weighted_states.shape)
            mat1 = weighted_states.unsqueeze(-2).repeat(1,
                                                        1, self.input_dim, 1)
            print("M1", mat1.shape)
            mat2 = weighted_states.unsqueeze(-3).repeat(1,
                                                        self.input_dim, 1, 1)
            print("M2: ", mat2.shape)
            w_concat = torch.cat([mat1, mat2], dim=-1)
            print("W cat: ", w_concat.shape)
            w_concat = torch.matmul(w_concat, a)
            print("W cat2: ", w_concat.shape)
            w_concat = self.lrelu(w_concat)
            print("W cat3: ", w_concat.shape)
            weights = self.fc1(w_concat)
            print("Weight: ", weights.shape)
            weighted_sum_batch = torch.matmul(weights, weighted_states)
            print(weighted_sum_batch.shape)
        elif self.type == "nn":
            weights = self.ann(self.weights)
            weights = self.fc1(weights)
        else:
            weights = self.fc1(self.weights)
            assert (x.shape[1] == weights.shape[0])
            if self.valspace == "complex":
                weights = torch.sqrt(weights)
        return weights


class LINEAR_ONE(nn.Module):
    def __init__(self, input_dim):
        super(LINEAR_ONE, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc2 = nn.Softmax(dim=-1)

    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        x = torch.outer(x, x)
        x = F.linear(x, y)
        return x


class Observation(nn.Module):
    def __init__(self, num_obs=64):
        super(Observation, self).__init__()
        self.weights = nn.Parameter(torch.empty(num_obs, 768).uniform_(-1, 1))

    def forward(self, x):
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
