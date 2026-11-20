"""Experiments 1-3 (5 seeds each, mean +- std), the 'NN on vs off' comparison and the experiment plots."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import CFG, HORIZON, SEED
from .csp import CSPSolver, SOLVER_VARIANTS, DEFAULT_APPLIANCES, make_csp_instance
from .data import DataBundle
from .integration import evaluate_on_actual, plan_day
from .nn import Forecaster, baseline_last_week, baseline_yesterday, mae, rmse, train_forecaster
from .search import battery_dp, battery_search, check_heuristic_admissible, make_battery_problem

RUN_SEEDS = [SEED + i for i in range(5)]      # 42..46: every experiment uses these 5 seeds


def mean_std_table(df: pd.DataFrame, by: list, cols: list) -> pd.DataFrame:
    """Aggregate per-run rows to 'mean +- std' strings (std over runs, ddof=1)."""
    g = df.groupby(by)[cols].agg(["mean", "std"])
    return pd.DataFrame({c: [f"{m:.4g} +- {0 if np.isnan(s) else s:.3g}" for m, s in zip(g[(c, "mean")], g[(c, "std")])] for c in cols}, index=g.index)


def make_search_scenario(day_load: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Real daily load + a seeded 2-hour extra appliance block (SIMULATED) so that scenarios differ."""
    bump = np.zeros(HORIZON); s = int(rng.integers(0, 22)); bump[s:s + 2] = rng.uniform(0.5, 1.5)
    return day_load + bump


# ------------------------------------------------------------------ Experiment 1: UCS vs A*
def run_experiment_1(data: DataBundle, n_scenarios: int = 30) -> pd.DataFrame:
    rows = []
    for seed in RUN_SEEDS:
        rng = np.random.default_rng(seed); picks = rng.choice(len(data.Y_test), n_scenarios, replace=False)
        acc = {"UCS": [], "A*": []}; dp_ok = adm_ok = infeasible = 0
        for i in picks:
            pr = make_battery_problem(make_search_scenario(data.Y_test[i], rng)); V = battery_dp(pr)
            if not np.isfinite(V[0, pr.init_level]): infeasible += 1; continue
            adm_ok += check_heuristic_admissible(pr, V)
            for use_h in (False, True):
                r = battery_search(pr, use_h); dp_ok += abs(r.total_cost - V[0, pr.init_level]) < 1e-8
                acc[r.algorithm].append((r.total_cost, r.nodes_expanded, r.nodes_generated, r.time_s * 1e3))
        for algo, vals in acc.items():
            m = np.mean(vals, axis=0)
            rows.append({"seed": seed, "algorithm": algo, "cost EUR (simulated)": m[0], "nodes expanded": m[1], "nodes generated": m[2],
                         "time ms": m[3], "scenarios solved": len(vals), "infeasible": infeasible,
                         "cost == DP": dp_ok == 2 * len(vals), "heuristic admissible (all states)": adm_ok == len(vals)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Experiment 2: CSP algorithms
def run_experiment_2(data: DataBundle, sizes=(3, 5, 8)) -> pd.DataFrame:
    rows = []
    scenarios = {"normal limit (5.0 kW)": CFG, "tight limit (4.2 kW)": replace(CFG, household_limit_kw=4.2)}
    for seed in RUN_SEEDS:
        forecast = data.Y_test[np.random.default_rng(seed).integers(len(data.Y_test))]
        for scen, cfg_s in scenarios.items():
            for n in sizes:
                csp = make_csp_instance(n, seed, forecast, cfg_s)
                for name, (fc, ac) in SOLVER_VARIANTS.items():
                    sol, st = CSPSolver(fc, ac).solve(csp)
                    if sol is not None: assert not csp.check_solution(sol), f"{name} returned an invalid schedule"
                    rows.append({"seed": seed, "scenario": scen, "appliances": n, "algorithm": name, "solved": sol is not None,
                                 "aborted": st.aborted, "time ms": st.time_s * 1e3, "assignments": st.assignments, "rejected": st.rejected,
                                 "backtracks": st.backtracks, "domain reductions": st.domain_reductions})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Experiment 3: NN vs baselines in the planner
def run_experiment_3(data: DataBundle, n_days: int = 30, nn_cache: Optional[dict] = None) -> pd.DataFrame:
    nn_cache = {} if nn_cache is None else nn_cache
    rows = []
    for seed in RUN_SEEDS:
        if seed not in nn_cache:
            nn_cache[seed] = train_forecaster(data.X_train, data.Y_train, data.X_val, data.Y_val, seed=seed)
        picks = np.random.default_rng(seed).choice(len(data.Y_test), n_days, replace=False)
        methods = {"Neural Network": nn_cache[seed].predict_batch(data.X_test), "Yesterday": baseline_yesterday(data.X_test),
                   "Last week": baseline_last_week(data.X_test), "Oracle (real load, reference)": data.Y_test}
        for name, F in methods.items():
            ev = [evaluate_on_actual(plan_day(F[i]), data.Y_test[i], DEFAULT_APPLIANCES) for i in picks]
            rows.append({"seed": seed, "method": name, "MAE kW (all test days)": mae(data.Y_test, F), "RMSE kW (all test days)": rmse(data.Y_test, F),
                         "MAE kW (scenario days)": mae(data.Y_test[picks], F[picks]),
                         "cost EUR/day (simulated)": np.mean([e["cost"] for e in ev]),
                         "household-limit violations (h / 30 days)": sum(e["viol_household"] for e in ev),
                         "grid-limit violations (h / 30 days)": sum(e["viol_grid"] for e in ev),
                         "plan failures (fallback used)": sum(e["failed"] for e in ev)})
    return pd.DataFrame(rows)


def compare_nn_on_off(data: DataBundle, fc: Forecaster, n_days: int = 30, seed: int = SEED) -> pd.DataFrame:
    """System with the NN ON (NN forecast) vs OFF (no learned model: 'yesterday' persistence forecast) on identical days."""
    picks = np.random.default_rng(seed).choice(len(data.Y_test), n_days, replace=False)
    variants = {"NN ON (neural-network forecast)": fc.predict_batch(data.X_test), "NN OFF (yesterday persistence)": baseline_yesterday(data.X_test)}
    rows = []
    for name, F in variants.items():
        ev = [evaluate_on_actual(plan_day(F[i]), data.Y_test[i], DEFAULT_APPLIANCES) for i in picks]
        rows.append({"system": name, "forecast MAE kW": mae(data.Y_test[picks], F[picks]), "cost EUR/day (simulated)": np.mean([e["cost"] for e in ev]),
                     "household-limit violations (h)": sum(e["viol_household"] for e in ev), "grid-limit violations (h)": sum(e["viol_grid"] for e in ev),
                     "plan failures": sum(e["failed"] for e in ev)})
    return pd.DataFrame(rows).set_index("system")


# ------------------------------------------------------------------ plots
def plot_experiment_1(exp1: pd.DataFrame, path: Optional[Path] = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col in zip(axes, ["nodes expanded", "nodes generated", "time ms"]):
        g = exp1.groupby("algorithm")[col].agg(["mean", "std"]); ax.bar(g.index, g["mean"], yerr=g["std"].fillna(0), capsize=4, color=["tab:green", "tab:blue"])
        ax.set_title(f"UCS vs A*: {col} (mean +- std, 5 runs)", fontsize=9)
    plt.tight_layout(); _finish(fig, path)


def plot_experiment_2(exp2: pd.DataFrame, path: Optional[Path] = None) -> None:
    scens = list(exp2["scenario"].unique()); fig, axes = plt.subplots(len(scens), 3, figsize=(15, 3.5 * len(scens)), squeeze=False)
    for r, scen in enumerate(scens):
        for ax, col in zip(axes[r], ["assignments", "backtracks", "time ms"]):
            pv = exp2[exp2.scenario == scen].groupby(["appliances", "algorithm"])[col].agg(["mean", "std"]).reset_index()
            for k, name in enumerate(SOLVER_VARIANTS):
                d = pv[pv.algorithm == name]; ax.bar(np.arange(len(d)) + 0.27 * k - 0.27, d["mean"], 0.27, yerr=d["std"].fillna(0), label=name, capsize=2)
            ax.set_xticks(range(3)); ax.set_xticklabels(["3", "5", "8"]); ax.set_title(f"{col} - {scen}", fontsize=9); ax.set_xlabel("appliances"); ax.legend(fontsize=7)
    plt.tight_layout(); _finish(fig, path)


def plot_experiment_3(exp3: pd.DataFrame, path: Optional[Path] = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, col in zip(axes, ["MAE kW (all test days)", "cost EUR/day (simulated)", "grid-limit violations (h / 30 days)"]):
        g = exp3.groupby("method")[col].agg(["mean", "std"]); ax.bar(range(len(g)), g["mean"], yerr=g["std"].fillna(0), capsize=3, color=["tab:red", "gray", "tab:green", "tab:blue"][:len(g)])
        ax.set_xticks(range(len(g))); ax.set_xticklabels([m.split(" (")[0] for m in g.index], rotation=15); ax.set_title(col, fontsize=9)
    plt.tight_layout(); _finish(fig, path)


def _finish(fig, path: Optional[Path]) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
