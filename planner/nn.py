"""Neural-network load forecaster, baselines and metrics."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from .config import SEED, set_seed
from .data import FEATURE_NAMES, N_LOAD_FEATURES


MAPE_MIN_ACTUAL_KW = 0.2

def mae(y, p) -> float: return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))

def rmse(y, p) -> float: return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))

def mape(y, p, min_actual: float = MAPE_MIN_ACTUAL_KW) -> float:
    """MAPE in % over points with actual >= min_actual; NaN if no such point."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    keep = y >= min_actual
    return float(np.mean(np.abs(y[keep] - p[keep]) / y[keep]) * 100) if keep.any() else float("nan")

class LoadForecastNet(nn.Module):
    def __init__(self, n_in: int, n_out: int = 24, hidden: tuple = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]; prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

@dataclass
class Forecaster:
    """Trained network + the scaling constants fitted on the training set."""
    model: LoadForecastNet
    load_mean: float
    load_std: float
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    best_epoch: int = 0

    def _scale(self, X: np.ndarray) -> np.ndarray:
        Xs = X.copy(); Xs[:, :N_LOAD_FEATURES] = (Xs[:, :N_LOAD_FEATURES] - self.load_mean) / self.load_std
        return Xs

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Forecast of shape (n_samples, 24) in kW, clipped at 0 (power cannot be negative)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected features of shape (n, {len(FEATURE_NAMES)}), got {X.shape}")
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(self._scale(X))).numpy()
        return np.clip(out * self.load_std + self.load_mean, 0.0, None)

def train_forecaster(X_tr, Y_tr, X_va, Y_va, seed: int = SEED, max_epochs: int = 300, patience: int = 20,
                     lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 64) -> Forecaster:
    """Train with early stopping. Scaling constants come from the TRAINING data only."""
    set_seed(seed)
    mean, std = float(X_tr[:, :N_LOAD_FEATURES].mean()), float(X_tr[:, :N_LOAD_FEATURES].std() + 1e-8)
    fc = Forecaster(LoadForecastNet(X_tr.shape[1], Y_tr.shape[1]), mean, std)
    xt = torch.from_numpy(fc._scale(X_tr)); yt = torch.from_numpy((Y_tr - mean) / std)
    xv = torch.from_numpy(fc._scale(X_va)); yv = torch.from_numpy((Y_va - mean) / std)
    opt = torch.optim.AdamW(fc.model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss(); gen = torch.Generator().manual_seed(seed)
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(1, max_epochs + 1):
        fc.model.train(); perm = torch.randperm(len(xt), generator=gen); total = 0.0
        for i in range(0, len(xt), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad(); loss = loss_fn(fc.model(xt[b]), yt[b]); loss.backward(); opt.step()
            total += loss.item() * len(b)
        fc.model.eval()
        with torch.no_grad(): val = loss_fn(fc.model(xv), yv).item()
        fc.train_losses.append(total / len(xt)); fc.val_losses.append(val)
        if val < best_val - 1e-6:
            best_val, bad, fc.best_epoch = val, 0, epoch
            best_state = {k: v.clone() for k, v in fc.model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    fc.model.load_state_dict(best_state)
    return fc

def baseline_yesterday(X: np.ndarray) -> np.ndarray: return X[:, :24]

def baseline_last_week(X: np.ndarray) -> np.ndarray: return X[:, 24:48]

def forecast_metrics(Y: np.ndarray, P: np.ndarray) -> dict:
    return {"MAE (kW)": mae(Y, P), "RMSE (kW)": rmse(Y, P), "MAPE (%)": mape(Y, P)}


def save_forecaster(fc: Forecaster, path) -> None:
    """Persist trained weights + scaling constants."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n_in = fc.model.net[0].in_features; n_out = fc.model.net[-1].out_features
    torch.save({"state": fc.model.state_dict(), "load_mean": fc.load_mean, "load_std": fc.load_std, "n_in": n_in, "n_out": n_out,
                "best_epoch": fc.best_epoch, "train_losses": fc.train_losses, "val_losses": fc.val_losses}, path)


def load_forecaster(path) -> Forecaster:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = LoadForecastNet(ck["n_in"], ck["n_out"]); model.load_state_dict(ck["state"]); model.eval()
    return Forecaster(model, ck["load_mean"], ck["load_std"], ck["train_losses"], ck["val_losses"], ck["best_epoch"])
