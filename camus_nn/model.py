"""Shared input→hidden→3 ReLU network with weighted multilabel cross-entropy."""

from numbers import Integral

import numpy as np
import torch
from torch.nn import functional as F


class ShallowNetwork:
    def __init__(
        self,
        seed=42,
        weight_decay=1e-4,
        max_iter=None,
        hidden_width=8,
    ):
        self._check_hidden_width(hidden_width)
        self.seed = seed
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.hidden_width = int(hidden_width)

    @staticmethod
    def _check_hidden_width(hidden_width):
        if (
            isinstance(hidden_width, bool)
            or not isinstance(hidden_width, Integral)
            or hidden_width < 1
        ):
            raise ValueError("Hidden width must be a positive integer")

    @staticmethod
    def unpack(parameters, input_width=None, hidden_width=8):
        ShallowNetwork._check_hidden_width(hidden_width)
        count = len(parameters)
        combined_width, remainder = divmod(count - 3, hidden_width)
        inferred_width = combined_width - 4
        if remainder or inferred_width < 1:
            raise ValueError("Invalid shallow-network parameter count for hidden width")
        if input_width is not None and input_width != inferred_width:
            raise ValueError(
                "Neural input width or hidden width does not match fitted parameters"
            )
        first = inferred_width * hidden_width
        return (
            parameters[:first].reshape(inferred_width, hidden_width),
            parameters[first : first + hidden_width],
            parameters[first + hidden_width : first + 4 * hidden_width].reshape(
                hidden_width, 3
            ),
            parameters[first + 4 * hidden_width : first + 4 * hidden_width + 3],
        )

    @staticmethod
    def logits(parameters, x, hidden_width=8):
        w1, b1, w2, b2 = ShallowNetwork.unpack(parameters, x.shape[-1], hidden_width)
        return F.linear(torch.relu(F.linear(x, w1.T, b1)), w2.T, b2)

    def objective(self, parameters, x, y, weights):
        parameters = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
        x = torch.as_tensor(x, dtype=torch.float64)
        y = torch.as_tensor(y, dtype=torch.float64)
        weights = torch.as_tensor(weights, dtype=torch.float64)
        hidden_width = getattr(self, "hidden_width", 8)
        logits = self.logits(parameters, x, hidden_width)
        loss = (
            weights[:, None]
            * F.binary_cross_entropy_with_logits(
                logits, y, reduction="none", pos_weight=torch.as_tensor(1., dtype=torch.float64)
            )
        ).sum() / 3
        w1, _, w2, _ = self.unpack(parameters, hidden_width=hidden_width)
        loss = loss + self.weight_decay * (w1.square().sum() + w2.square().sum()) / 2
        (gradient,) = torch.autograd.grad(loss, parameters)
        return loss.item(), gradient.detach().numpy()

    def fit(self, x, y, sample_weight):
        """Fit a fresh shared frequency model to convergence."""
        from scipy.optimize import minimize
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] < 1:
            raise ValueError("Neural features must be a two-dimensional matrix")
        input_width = x.shape[1]
        hidden_width = getattr(self, "hidden_width", 8)
        self._check_hidden_width(hidden_width)
        weights = np.asarray(sample_weight, dtype=float)
        weights = weights / weights.sum()
        mean = np.sum(x * weights[:, None], axis=0)
        scale = np.sqrt(np.sum((x - mean) ** 2 * weights[:, None], axis=0))
        scale[scale < 1e-12] = 1
        self.mean_, self.scale_ = mean, scale
        self.input_width = input_width
        x = (x - mean) / scale
        rng = np.random.default_rng(self.seed)
        prevalence = np.clip(np.sum(y * weights[:, None], axis=0), 1e-5, 1 - 1e-5)
        initial = np.concatenate(
            [
                rng.normal(0, np.sqrt(2 / input_width), input_width * hidden_width),
                rng.normal(0, 0.1, hidden_width),
                rng.normal(0, 0.1, 3 * hidden_width),
                np.log(prevalence / (1 - prevalence)),
            ]
        )
        self.initial_loss_ = float(self.objective(initial, x, y, weights)[0])
        iterations = 0
        while True:
            result = minimize(
                self.objective,
                initial,
                args=(x, y, weights),
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxiter": self.max_iter or 1000,
                    "maxfun": 100000,
                    "ftol": 1e-9,
                    "gtol": 1e-5,
                },
            )
            if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
                raise ValueError("Nonfinite neural-network optimization result")
            iterations += int(result.nit)
            self.parameters_ = result.x
            self.optimization_ = dict(
                converged=bool(result.success),
                iterations=iterations,
                loss=float(result.fun),
                message=str(result.message),
                gradient_max=float(np.max(np.abs(result.jac))),
            )
            print("Neural progress:", self.optimization_, flush=True)
            if result.success or self.max_iter is not None:
                break
            if result.status != 1 or result.nit == 0:
                raise RuntimeError(f"Optimization failed: {result.message}")
            initial = result.x
        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1:] != np.shape(self.mean_):
            raise ValueError("Neural input width does not match fitted scaling")
        with torch.no_grad():
            parameters = torch.as_tensor(self.parameters_, dtype=torch.float64)
            features = torch.as_tensor(
                (x - self.mean_) / self.scale_, dtype=torch.float64
            )
            return torch.sigmoid(
                self.logits(parameters, features, getattr(self, "hidden_width", 8))
            ).numpy()
