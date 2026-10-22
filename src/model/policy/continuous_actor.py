import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, MultivariateNormal

from model.mixed_distribution import MixedDistribution
from utils import torch_utils


class ContinuousActor(nn.Module):
    def __init__(self,
                 action_dim: int,
                 layers: list[int],
                 activation: nn.Module | dict[str, any],
                 state_dependent_std: bool = False
                 ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std
        if isinstance(activation, dict):
            self.hidden_act = activation['hidden']
            self.output_act = activation['output']
        else:
            self.hidden_act = self.output_act = activation
        self.enc = torch_utils.make_mlp_layers(layers, self.hidden_act)
        self.mu = torch_utils.make_mlp_layers([layers[-1], action_dim], self.output_act)
        if self.state_dependent_std:
            self.std = torch_utils.make_mlp_layers([layers[-1], action_dim], self.output_act)
            self.softplus = nn.Softplus()
        else:
            self.logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.epsilon_prob = torch_utils.make_mlp_layers([layers[-1], 1], torch.nn.Sigmoid, final_layer_activation=True)

    def forward(self, obs: torch.tensor) -> MixedDistribution:
        hidden = self.enc(obs)
        mu = self.mu(hidden)
        if self.state_dependent_std:
            std = self.softplus(self.std(hidden))
        else:
            std = self.logstd.expand_as(mu).exp()
        std = std + 1e-3
        normal = Normal(mu, std)
        eps_prob = torch.clamp(self.epsilon_prob(hidden), 1e-3, 1 - 1e-3)
        return MixedDistribution(normal, eps_prob.flatten())


class ContinuousActorNoEpsilon(nn.Module):
    COV_MIN = 1e-3
    COV_CLAMP_MIN = -5
    COV_CLAMP_MAX = 20

    def __init__(self,
                 action_dim: int,
                 layers: list[int],
                 activation: nn.Module | dict[str, any],
                 state_dependent_std: bool = False
                 ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std
        if isinstance(activation, dict):
            self.hidden_act = activation['hidden']
            self.output_act = activation['output']
        else:
            self.hidden_act = self.output_act = activation
        self.enc = torch_utils.make_mlp_layers(layers, self.hidden_act)
        self.mu = torch_utils.make_mlp_layers([layers[-1], action_dim], self.output_act)
        self.num_cholesky_entries = action_dim * (action_dim + 1) // 2
        if self.state_dependent_std:
            self.cholesky = torch_utils.make_mlp_layers(
                [layers[-1], self.num_cholesky_entries], self.output_act)
        else:
            self.cholesky = nn.Parameter(torch.zeros(1, self.num_cholesky_entries))
            diagonal_indices = torch.arange(1, action_dim + 1)
            diagonal_indices = diagonal_indices * (diagonal_indices + 1) // 2 - 1
            initial_diagonal = np.log(np.exp(1.0 - self.COV_MIN) - 1.0)
            with torch.no_grad():
                self.cholesky[:, diagonal_indices] = initial_diagonal

    def forward(self, obs: torch.tensor) -> MultivariateNormal:
        hidden = self.enc(obs)
        mu = self.mu(hidden)
        if self.state_dependent_std:
            cholesky_vector = self.cholesky(hidden)
        else:
            cholesky_vector = self.cholesky.expand(
                mu.shape[0], self.num_cholesky_entries)

        cholesky_vector = torch.clamp(
            cholesky_vector, self.COV_CLAMP_MIN, self.COV_CLAMP_MAX)
        diagonal_indices = torch.arange(
            1, self.action_dim + 1, device=mu.device)
        diagonal_indices = diagonal_indices * (diagonal_indices + 1) // 2 - 1
        positive_diagonal = F.softplus(
            cholesky_vector[:, diagonal_indices]) + self.COV_MIN
        cholesky_vector = cholesky_vector.scatter(
            1,
            diagonal_indices.unsqueeze(0).expand(mu.shape[0], -1),
            positive_diagonal,
        )

        tril_indices = torch.tril_indices(
            self.action_dim, self.action_dim, device=mu.device)
        scale_tril = torch.zeros(
            mu.shape[0], self.action_dim, self.action_dim,
            device=mu.device, dtype=mu.dtype)
        scale_tril[:, tril_indices[0], tril_indices[1]] = cholesky_vector
        return MultivariateNormal(loc=mu, scale_tril=scale_tril)
