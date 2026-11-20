#!/usr/bin/env python
"""train_nn.py - train the load-forecasting network and compare it with the two baselines on the test set.
Usage: python train_nn.py [--seed 42] [--out models]
Writes models/forecaster.pt and results/nn_vs_baselines.csv"""
import argparse
from pathlib import Path
import pandas as pd
from planner.config import SEED
from planner.data import prepare_data
from planner.nn import baseline_last_week, baseline_yesterday, forecast_metrics, save_forecaster, train_forecaster

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=SEED); ap.add_argument("--out", default="models"); ap.add_argument("--results", default="results")
    a = ap.parse_args(); b = prepare_data()
    fc = train_forecaster(b.X_train, b.Y_train, b.X_val, b.Y_val, seed=a.seed)
    save_forecaster(fc, Path(a.out) / "forecaster.pt")
    preds = {"Neural Network": fc.predict_batch(b.X_test), "Yesterday (same hour)": baseline_yesterday(b.X_test), "Last week (same hour)": baseline_last_week(b.X_test)}
    table = pd.DataFrame({k: forecast_metrics(b.Y_test, p) for k, p in preds.items()}).T
    Path(a.results).mkdir(exist_ok=True); table.to_csv(Path(a.results) / "nn_vs_baselines.csv")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
    fig, ax = plt.subplots(figsize=(6.5, 3.2)); ax.plot(fc.train_losses, label="train"); ax.plot(fc.val_losses, label="validation")
    ax.axvline(fc.best_epoch - 1, ls="--", c="gray", label="best epoch (early stopping)"); ax.set(xlabel="epoch", ylabel="MSE (standardised)", title="Training and validation loss"); ax.legend()
    fig.savefig(Path(a.results) / "nn_training_loss.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    n = 7; fig, ax = plt.subplots(figsize=(11, 3.4)); sl = slice(0, n)
    ax.plot(b.Y_test[sl].ravel(), "k", lw=1.5, label="actual (real)")
    for k, c in (("Neural Network", "tab:red"), ("Yesterday (same hour)", "tab:blue"), ("Last week (same hour)", "tab:green")): ax.plot(preds[k][sl].ravel(), c=c, lw=1, label=k)
    ax.set(title=f"Forecast vs actual, first {n} test days (from {b.d_test[0].date()})", xlabel="hour", ylabel="kW"); ax.legend(ncol=4, fontsize=8)
    fig.savefig(Path(a.results) / "nn_forecast_vs_actual.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"epochs run: {len(fc.val_losses)}, best epoch: {fc.best_epoch}"); print(table.round(4).to_string())
