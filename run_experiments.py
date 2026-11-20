#!/usr/bin/env python
"""run_experiments.py - the three experiments (5 seeds each, mean +- std). Tables -> results/*.csv, plots -> results/*.png.
Usage: python run_experiments.py [--out results] [--only 1 2 3]"""
import argparse
from pathlib import Path
from planner.data import prepare_data
from planner.evaluation import (compare_nn_on_off, mean_std_table, plot_experiment_1, plot_experiment_2, plot_experiment_3,
                                run_experiment_1, run_experiment_2, run_experiment_3)
from planner.nn import load_forecaster

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); ap.add_argument("--only", type=int, nargs="*", default=[1, 2, 3])
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True, parents=True); data = prepare_data()
    if 1 in a.only:
        print("\n=== Experiment 1: UCS vs A* (30 scenarios per run, 5 runs) ===")
        e1 = run_experiment_1(data); e1.to_csv(out / "exp1_runs.csv", index=False)
        t = mean_std_table(e1, ["algorithm"], ["cost EUR (simulated)", "nodes expanded", "nodes generated", "time ms"]); t.to_csv(out / "exp1_summary.csv"); print(t.to_string())
        print("cost equals DP optimum in all runs:", bool(e1["cost == DP"].all()), "| heuristic admissible on every state:", bool(e1["heuristic admissible (all states)"].all()))
        plot_experiment_1(e1, out / "exp1_ucs_vs_astar.png")
    if 2 in a.only:
        print("\n=== Experiment 2: BT vs BT+FC vs BT+FC+AC3 (3, 5, 8 appliances) ===")
        e2 = run_experiment_2(data); e2.to_csv(out / "exp2_runs.csv", index=False)
        t = mean_std_table(e2, ["scenario", "appliances", "algorithm"], ["time ms", "assignments", "rejected", "backtracks", "domain reductions"]); t.to_csv(out / "exp2_summary.csv"); print(t.to_string())
        print(e2.groupby(["scenario", "appliances", "algorithm"])[["solved", "aborted"]].sum().to_string())
        plot_experiment_2(e2, out / "exp2_csp_comparison.png")
    if 3 in a.only:
        print("\n=== Experiment 3: NN vs baselines inside the planner (30 test days per run, 5 runs) ===")
        e3 = run_experiment_3(data); e3.to_csv(out / "exp3_runs.csv", index=False)
        cols = [c for c in e3.columns if c not in ("seed", "method")]
        t = mean_std_table(e3, ["method"], cols); t.to_csv(out / "exp3_summary.csv"); print(t.T.to_string())
        plot_experiment_3(e3, out / "exp3_nn_vs_baselines.png")
    print(f"\nAll tables and plots written to {out}/")
