#!/usr/bin/env python
"""run_demo.py - integrated demo: input -> NN forecast -> CSP schedule -> battery search -> final plan + plots.
Also shows (for the live demo): CSP pruning with/without FC and AC-3, the UCS/A* search trace, the CSP<->search feedback loop,
the 'no solution' message and the system with the NN ON vs OFF.

Usage:  python run_demo.py [--date 2010-02-17] [--out outputs] [--no-show] [--skip-nn-compare]
Plots are saved to <out>/ as PNG (and shown unless --no-show). All costs are SIMULATED."""
import argparse
from dataclasses import replace
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--date", default="2010-02-17", help="target day (nearest valid test-period day is used)")
ap.add_argument("--out", default="outputs"); ap.add_argument("--no-show", action="store_true"); ap.add_argument("--retrain", action="store_true")
ap.add_argument("--skip-nn-compare", action="store_true", help="skip the 30-day NN on/off comparison (faster)")
args = ap.parse_args()
if args.no_show: matplotlib.use("Agg")
import matplotlib.pyplot as plt

from planner.config import CFG, HORIZON, SEED
from planner.csp import Appliance, ApplianceCSP, CSPSolver, CSPStats, DEFAULT_APPLIANCES, SOLVER_VARIANTS, explain_no_solution, make_csp_instance
from planner.data import prepare_data
from planner.evaluation import compare_nn_on_off
from planner.integration import plan_day, run_home_energy_planner
from planner.nn import baseline_yesterday, load_forecaster, save_forecaster, train_forecaster
from planner.search import battery_search, format_trace, make_battery_problem

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
def banner(t): print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)
def show(fig, name):
    fig.savefig(out / name, dpi=130, bbox_inches="tight"); print(f"[plot saved] {out / name}")
    plt.show() if not args.no_show else plt.close(fig)

banner("0) Data and model")
data = prepare_data(); model_path = Path("models/forecaster.pt")
if model_path.exists() and not args.retrain: fc = load_forecaster(model_path); print(f"[model] loaded {model_path}")
else:
    fc = train_forecaster(data.X_train, data.Y_train, data.X_val, data.Y_val, seed=SEED); save_forecaster(fc, model_path); print(f"[model] trained (best epoch {fc.best_epoch}) and saved to {model_path}")
date = data.d_test[np.argmin(np.abs((data.d_test - pd.Timestamp(args.date)).days))]
print(f"Demo day (real test-period day, never used for training): {date.date()}")

banner("1) INTEGRATED RUN: input -> forecast -> CSP -> battery search -> plan")
import planner.plotting as P
P.set_figure_output(out / "demo_plan.png", show=not args.no_show)
plan = run_home_energy_planner(date, data.daily, fc, plot=True)
if not plan.success: raise SystemExit("planner failed - see message above")

banner("2a) AC-3 pruning on the default 5-appliance problem (dryer must start after the washer has finished)")
csp5 = ApplianceCSP(DEFAULT_APPLIANCES, plan.forecast, CFG.household_limit_kw, CFG.circuit_limit_kw)
before = {v: len(d) for v, d in csp5.domains.items()}
doms = {v: list(d) for v, d in csp5.domains.items()}; st0 = CSPStats(); CSPSolver(False, True)._ac3(csp5, doms, st0)
print("Domain sizes before AC-3:", before); print("Domain sizes after  AC-3:", {v: len(d) for v, d in doms.items()}, f"(values removed: {st0.domain_reductions}, revisions: {st0.ac3_revisions})")
print("washing_machine start hours removed:", sorted(set(csp5.domains['washing_machine']) - set(doms['washing_machine'])), "| dryer start hours removed:", sorted(set(csp5.domains['dryer']) - set(doms['dryer'])))

banner("2b) BT vs BT+FC vs BT+FC+AC3 on an 8-appliance instance with a tight 4.2 kW household limit")
tight = replace(CFG, household_limit_kw=4.2)
def _effort(seed):
    c = make_csp_instance(8, seed, data.Y_test[np.random.default_rng(seed).integers(len(data.Y_test))], tight)
    s, st = CSPSolver(False, False).solve(c); return (st.assignments + st.rejected) if s is not None else -1
demo_seed = max(range(SEED, SEED + 40), key=_effort)      # most informative of 40 seeded instances (selection is stated here, not hidden)
print(f"Instance: seed {demo_seed} = the instance (of 40 seeded ones) where plain BT works hardest. Other seeds are easier.")
csp = make_csp_instance(8, demo_seed, data.Y_test[np.random.default_rng(demo_seed).integers(len(data.Y_test))], tight)
rows = []
for name, (f_, a_) in SOLVER_VARIANTS.items():
    sol, st = CSPSolver(f_, a_).solve(csp)
    rows.append({"algorithm": name, "solved": sol is not None, "assignments": st.assignments, "rejected": st.rejected, "backtracks": st.backtracks,
                 "domain reductions": st.domain_reductions, "time (ms)": round(st.time_s * 1e3, 3)})
tbl = pd.DataFrame(rows); print(tbl.to_string(index=False))
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
for ax, col in zip(axes, ["assignments", "rejected", "domain reductions"]): ax.bar(tbl["algorithm"], tbl[col], color=["gray", "tab:blue", "tab:green"]); ax.set_title(col)
plt.tight_layout(); show(fig, "demo_csp_pruning.png")

banner("3) SEARCH TRACE: UCS vs A* on the demo day's battery problem (first expansions)")
problem = make_battery_problem(plan.total_load)
for use_h in (False, True):
    tr = []; r = battery_search(problem, use_h, trace=tr)
    print(f"--- {r.algorithm}: expanded {r.nodes_expanded}, generated {r.nodes_generated}, cost {r.total_cost:.4f} EUR (simulated)"); print(format_trace(tr, problem, limit=10))

banner("4) FEEDBACK LOOP (synthetic stress scenario): battery failure -> new CSP restriction -> re-run")
stress_cfg = replace(CFG, household_limit_kw=6.0, grid_limit_kw=4.2, battery_capacity_kwh=6.0, battery_max_charge_kw=1.0, battery_max_discharge_kw=1.0, battery_initial_kwh=6.0, battery_final_min_kwh=0.0)
sp = plan_day(np.full(HORIZON, 1.5), stress_cfg, [Appliance("ev_charger", 4, 3.3, 0, 24, "garage"), Appliance("iron", 1, 1.2, 0, 24, "laundry")])
print(f"success={sp.success}, iterations={sp.iterations}"); print(pd.DataFrame(sp.feedback_log).to_string(index=False)); print("final schedule:", sp.schedule)

banner("5) NO-SOLUTION message")
ip = plan_day(np.full(HORIZON, 3.0), CFG, [Appliance("ev_charger", 4, 3.3, 0, 24, "garage")])
print("success:", ip.success)
for m in ip.messages: print(" -", m)

if not args.skip_nn_compare:
    banner("6) SYSTEM WITH NN ON vs OFF (30 identical test days; OFF = 'yesterday' persistence forecast; costs SIMULATED)")
    cmp_ = compare_nn_on_off(data, fc); print(cmp_.round(4).T.to_string())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, col in zip(axes, ["forecast MAE kW", "cost EUR/day (simulated)", "grid-limit violations (h)"]): ax.bar(["NN ON", "NN OFF"], cmp_[col], color=["tab:green", "gray"]); ax.set_title(col, fontsize=9)
    plt.tight_layout(); show(fig, "demo_nn_on_off.png")
print("\nDemo finished.")
