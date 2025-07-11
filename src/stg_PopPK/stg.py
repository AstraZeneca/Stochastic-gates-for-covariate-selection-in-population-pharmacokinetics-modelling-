"""
Adapted from: https://github.com/runopti/stg
Original authors: Yutaro Yamada   https://github.com/runopti
License: MIT

Summary:
    This module implements the Stochastic Gates (STG) regression framework for feature selection in neural networks.
    The code was adapted from the Run Opti 'stg' repository, with  modifications:

    - Added early stopping functionality (via stop_epoch)
    - Tracked mus, losses, and best model checkpoint
    - Added convergence check for stochastic gates

Please see the original repository for algorithmic details and upstream updates.
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from typing import Optional, Any, Callable, Iterable, Union, Dict, Tuple
from sklearn.metrics import mean_squared_error, r2_score
from stg_PopPK.models import STGRegressionModel
from stg_PopPK.utils import (
    get_optimizer,
    as_tensor,
    as_float,
    as_numpy,
    FastTensorDataLoader,
    standard_truncnorm_sample,
)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class STG:
    """
    Implements the Stochastic Gates framework for neural-network-based regression with feature selection.

    Args:
        device (str, torch.device): 'cpu' or 'cuda'
        input_dim (int): Number of input features
        output_dim (int): Number of output dimensions
        hidden_dims (list[int]): List of MLP hidden layer sizes
        activation (str): Name of activation function
        sigma (float): Initial gate noise standard deviation
        lam (float): Regularization coefficient for gates
        optimizer (str): Optimizer name ('Adam', etc)
        learning_rate (float): Learning rate for optimizer
        batch_size (int): Training mini-batch size
        weight_decay (float): L2 regularization coefficient
        dropout (bool or float): Dropout setting
        opt_params (dict): Additional optimizer parameters
        save_mus (bool): If True, logs mu values per epoch
        save_best (bool): If True, keeps copy of best checkpoint
        save_loss (bool): If True, logs per-epoch losses
        warm_start (Optional[np.ndarray]): Optional initialization for mu

    Attributes (selected):
        model (STGRegressionModel): Underlying regression model
        optimizer (torch.optim.Optimizer): Current optimizer

    """

    def __init__(
        self,
        *,
        device: str,
        input_dim: int = 784,
        output_dim: int = 10,
        hidden_dims: list[int] = [400, 200],
        activation: str = "relu",
        sigma: float = 0.5,
        lam: float = 0.1,
        optimizer: str = "Adam",
        learning_rate: float = 1e-5,
        batch_size: int = 100,
        weight_decay: float = 1e-3,
        dropout: Union[bool, float] = False,
        opt_params: dict = {},
        save_mus: bool = False,
        save_best: bool = False,
        save_loss: bool = False,
        warm_start: Optional[np.ndarray] = None,
    ):
        self.batch_size = batch_size
        self.device = self.get_device(device)
        self.model = STGRegressionModel(
            input_dim,
            output_dim,
            hidden_dims,
            device=self.device,
            activation=activation,
            sigma=sigma,
            lam=lam,
            dropout=dropout,
            warm_start=warm_start,
        )
        self.model.apply(self.init_weights)
        self.optimizer = get_optimizer(
            optimizer, self.model, **opt_params, lr=learning_rate, weight_decay=weight_decay
        )
        self._model = self.model.to(self.device)

        self.save_best = save_best
        self.best_checkpoint = None
        self.save_loss = save_loss
        self.save_mus = save_mus
        torch.manual_seed(10)
        torch.cuda.manual_seed_all(10)

    def get_device(self, device: str) -> torch.device:
        if device == "cpu":
            device = torch.device("cpu")
        elif device == "cuda":
            args_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if args_cuda else "cpu")
        else:
            raise NotImplementedError("Only 'cpu' or 'cuda' is a valid option.")
        return device

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            stddev = torch.tensor(0.1)
            shape = m.weight.shape
            m.weight = nn.Parameter(
                standard_truncnorm_sample(
                    lower_bound=-2 * stddev, upper_bound=2 * stddev, sample_shape=shape
                )
            )
            torch.nn.init.zeros_(m.bias)

    def get_dataloader(self, X, y, shuffle):
        tensor_names = ("input", "label")
        data_loader = FastTensorDataLoader(
            torch.from_numpy(X).float().to(self.device),
            torch.from_numpy(y).float().to(self.device),
            tensor_names=tensor_names,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )
        return data_loader

    def train_step(self, feed_dict):
        assert self.model.training
        pred = self.model.forward(feed_dict["input"])
        total_loss, loss, reg = self.model.calculate_loss(pred, feed_dict["label"])
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        loss = as_float(loss)
        reg = as_float(reg)
        pred = as_numpy(pred)
        r2 = r2_score(feed_dict["label"].cpu(), pred)
        return loss, reg, r2

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []
        regs = []
        r2s = []
        for feed_dict in data_loader:
            loss, reg, r2 = self.train_step(feed_dict)
            losses.append(loss)
            regs.append(reg)
            r2s.append(r2)
        return np.mean(losses), np.mean(regs), np.mean(r2s)

    def validate(self, data_loader):
        losses = []
        regs = []
        r2s = []
        self.model.eval()
        for feed_dict in data_loader:
            with torch.no_grad():
                pred = self.model.forward(feed_dict["input"])
                _, loss, reg = self.model.calculate_loss(pred, feed_dict["label"])
                loss = as_float(loss)
                reg = as_float(reg)
                pred = as_numpy(pred)
            r2 = r2_score(feed_dict["label"].cpu(), pred)
            losses.append(loss)
            regs.append(reg)
            r2s.append(r2)
        return np.mean(losses), np.mean(regs), np.mean(r2s)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        nr_epochs: int,
        valid_X: Optional[np.ndarray] = None,
        valid_y: Optional[np.ndarray] = None,
        stop_epoch: Optional[int] = None,
        shuffle: bool = True,
        print_interval: int = 1000,
    ) -> Any:
        """
        Main entry for model training. In case optional validation dataset is provided it will use early stopping criteria with
        stop_epoch. If no validation data is provided it will train for nr_epochs.

        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            nr_epochs (int): Number of epochs
            valid_X (Optional[np.ndarray]): Validation features, if validation desired
            valid_y (Optional[np.ndarray]): Validation labels
            stop_epoch (Optional[int]): Early stopping patience in epochs
            shuffle (bool): Whether to shuffle train batches
            print_interval (int): Validation progress print interval

        Returns:
            Bool or Any: Converged state, or return from train_valid
        """
        if valid_X is not None:
            return self.train_valid(
                X, y, nr_epochs, valid_X, valid_y, stop_epoch, shuffle, print_interval
            )
        return self.train_only(X, y, nr_epochs, stop_epoch, shuffle)

    def train_only(
        self,
        X: np.ndarray,
        y: np.ndarray,
        nr_epochs: int,
        stop_epoch: Optional[int] = None,
        shuffle: bool = True,
    ) -> bool:
        """
        Trains only on main set, with optional stopping

        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Input targets
            nr_epochs (int): Max epochs
            stop_epoch (Optional[int]): Patience stop after gates converged(if None, equals nr_epochs)
            shuffle (bool): Whether to shuffle

        Returns:
            bool: Whether model converged
        """
        if stop_epoch is None:
            stop_epoch = nr_epochs
        data_loader = self.get_dataloader(X, y, shuffle)
        self.val_mse_best = -1
        self.val_mse_mean = -1
        self.val_r2_best = -1
        if self.save_loss:
            self.train_loss_arr = []
            self.valid_loss_arr = []
            self.train_r2_arr = []
            self.valid_r2_arr = []
        if self.save_mus:
            self.mus_arr = []
        counter = 0
        for epoch in range(nr_epochs):
            loss, reg, r2 = self.train_epoch(data_loader)
            S, mus, converged = self.gates_converged()
            if self.save_mus:
                self.mus_arr.append(mus)
            if self.save_loss:
                self.train_loss_arr.append(loss)
                self.train_r2_arr.append(r2)
            if converged:
                counter += 1
            if counter > stop_epoch:
                break
        self.Sbest = S
        if self.save_best:
            self.best_checkpoint = copy.deepcopy(self.model.state_dict())
        self.train_mse_best = loss
        self.train_r2_best = r2
        return converged

    def train_valid(
        self,
        X: np.ndarray,
        y: np.ndarray,
        nr_epochs: int,
        valid_X: np.ndarray,
        valid_y: np.ndarray,
        stop_epoch: Optional[int] = None,
        shuffle: bool = True,
        print_interval: int = 1000,
    ) -> bool:
        """
        Trains with periodic validation and checkpointing; supports early stop.

        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            nr_epochs (int): Max epochs
            valid_X (np.ndarray): Validation features
            valid_y (np.ndarray): Validation labels
            stop_epoch (Optional[int]): Early stop patience (if None, equals nr_epochs)
            shuffle (bool): Shuffle training inputs each epoch
            print_interval (int): Print frequency (in epochs)

        Returns:
            bool: True if model converged before stopping
        """
        if stop_epoch is None:
            stop_epoch = nr_epochs
        mean_val = np.ones_like(valid_y) * np.mean(valid_y)
        self.val_mse_mean = mean_squared_error(valid_y, mean_val)
        self.val_mse_best = 1000
        self.best_checkpoint = None
        if self.save_loss:
            self.train_loss_arr = []
            self.valid_loss_arr = []
            self.train_r2_arr = []
            self.valid_r2_arr = []
        if self.save_mus:
            self.mus_arr = []
        data_loader = self.get_dataloader(X, y, shuffle)
        val_data_loader = self.get_dataloader(valid_X, valid_y, None)
        counter = 0
        converged_past = False
        for epoch in range(nr_epochs):
            loss, reg, r2 = self.train_epoch(data_loader)
            val_loss, _, val_r2 = self.validate(val_data_loader)
            S, mus, converged = self.gates_converged()
            if self.save_mus:
                self.mus_arr.append(mus)
            if self.save_loss:
                self.train_loss_arr.append(loss)
                self.valid_loss_arr.append(val_loss)
                self.train_r2_arr.append(r2)
                self.valid_r2_arr.append(val_r2)
            if converged:
                converged_past = True
                if val_loss < self.val_mse_best:
                    self.val_mse_best = val_loss
                    self.val_r2_best = val_r2
                    self.Sbest = S
                    self.train_mse_best = loss
                    self.train_r2_best = r2
                    if self.save_best:
                        self.best_checkpoint = copy.deepcopy(self.model.state_dict())
                    counter = 0
                else:
                    counter += 1
            if counter > stop_epoch:
                break
            if val_loss / loss > 2:
                logger.info("Overfitting")
                break
            if epoch % print_interval == 0:
                logger.info(
                    f"Epoch {epoch} | ValMSE (mean): {self.val_mse_mean:.6f} | "
                    f"ValLoss: {val_loss:.6f} | TrainLoss: {loss:.6f} | "
                    f"BestValMSE: {self.val_mse_best:.6f}"
                )
                gates = self.model.get_gates("prob")
                logger.info(f"Gate probabilities: {gates}")

        if not converged:
            self.val_mse_best = val_loss
            self.val_r2_best = val_r2
            self.Sbest = S
            self.train_mse_best = loss
            self.train_r2_best = r2
            if self.save_best:
                self.best_checkpoint = copy.deepcopy(self.model.state_dict())
        return converged_past

    def get_gates(self, mode: str) -> np.ndarray:
        return self.model.get_gates(mode)

    def gates_converged(self):
        """
        Checks convergence of gates.

        Returns:
            tuple:
                S (float): sum of active features
                mus (np.ndarray): Mu values (gate params) snapshot
                converged (bool): True if all gates are either 0 or 1
        """
        mus = self.model.mu.cpu().detach().numpy().copy()
        zs = np.clip(mus + 0.5, 0, 1)
        S = (zs).sum()
        if any((zs < 1) & (zs > 0)):
            converged = False
        else:
            converged = True

        return S, mus, converged
