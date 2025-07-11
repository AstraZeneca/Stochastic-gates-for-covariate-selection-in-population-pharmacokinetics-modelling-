"""
Stochastic Gates Layers and Models
Adapted from: https://github.com/runopti/stg
Original authors: Yutaro Yamada   https://github.com/runopti
License: MIT

Modification:
- introduced warm_start as mu-gates initialization
"""

import math
import numpy as np
import torch
import torch.nn as nn
from stg_PopPK.utils import get_batcnnorm, get_dropout, get_activation

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class FeatureSelector(nn.Module):
    """
    A layer for stochastic feature selection using learnable gating variables (mu).
    The gates are sampled stochastically during training to induce sparsity.

    Args:
        input_dim (int): Number of input features.
        sigma (float): Standard deviation of the noise used for gate sampling.
        warm_start (np.ndarray or list or None): Optional initial values for mu-gates.
    """

    def __init__(self, input_dim, sigma, warm_start=None):
        super(FeatureSelector, self).__init__()
        torch.manual_seed(10)
        torch.cuda.manual_seed_all(10)
        if warm_start is not None:
            assert len(warm_start) == input_dim
            tensor_start = torch.tensor(warm_start).float()
        else:
            tensor_start = 0
        tensor = (
            0.01
            * torch.randn(
                input_dim,
            )
            + tensor_start
        )
        self.mu = torch.nn.Parameter(tensor, requires_grad=True)
        self.noise = torch.randn(self.mu.size())
        self.sigma = sigma

    def forward(self, prev_x):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularizer(self, x):
        """Gaussian CDF."""
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self


class LinearLayer(nn.Sequential):
    def __init__(
        self, in_features, out_features, batch_norm=None, dropout=None, bias=None, activation=None
    ):
        if bias is None:
            bias = batch_norm is None

        modules = [nn.Linear(in_features, out_features, bias=bias)]
        if batch_norm is not None and batch_norm is not False:
            modules.append(get_batcnnorm(batch_norm, out_features, 1))
        if dropout is not None and dropout is not False:
            modules.append(get_dropout(dropout, 1))
        if activation is not None and activation is not False:
            modules.append(get_activation(activation))
        super().__init__(*modules)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        batch_norm=None,
        dropout=None,
        activation="relu",
        flatten=True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)
        modules = []

        nr_hiddens = len(hidden_dims)
        for i in range(nr_hiddens):
            layer = LinearLayer(
                dims[i], dims[i + 1], batch_norm=batch_norm, dropout=dropout, activation=activation
            )
            modules.append(layer)
        layer = nn.Linear(dims[-2], dims[-1], bias=True)
        modules.append(layer)
        self.mlp = nn.Sequential(*modules)
        self.flatten = flatten

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, input):
        if self.flatten:
            input = input.view(input.size(0), -1)
        return self.mlp(input)


class MLPModel(MLPLayer):
    def freeze_weights(self):
        for name, p in self.named_parameters():
            if name != "mu":
                p.requires_grad = False

    def get_gates(self, mode):
        if mode == "raw":
            return self.mu.detach().cpu().numpy()
        elif mode == "prob":
            return np.minimum(1.0, np.maximum(0.0, self.mu.detach().cpu().numpy() + 0.5))
        else:
            raise NotImplementedError()


class STGRegressionModel(MLPModel):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        device,
        batch_norm=None,
        dropout=None,
        activation="relu",
        sigma=1.0,
        lam=0.1,
        warm_start=None,
    ):
        super().__init__(
            input_dim,
            output_dim,
            hidden_dims,
            batch_norm=batch_norm,
            dropout=dropout,
            activation=activation,
        )
        self.FeatureSelector = FeatureSelector(input_dim, sigma, warm_start)
        self.loss = nn.MSELoss()
        # self.loss = nn.SmoothL1Loss()
        self.reg = self.FeatureSelector.regularizer
        self.lam = lam
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma

    def forward(self, X):
        x = self.FeatureSelector(X)
        pred = super().forward(x)
        return pred

    def calculate_loss(self, pred, y):
        loss = self.loss(pred, y)
        loss = torch.mean(loss)
        reg = torch.mean(self.reg((self.mu + 0.5) / self.sigma))

        total_loss = loss + self.lam * reg
        return total_loss, loss, reg
