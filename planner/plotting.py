"""Plot helpers for the demo and experiments."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from .config import CFG, HORIZON, SimConfig
from .csp import ApplianceCSP
from .integration import PlanResult, hourly_cost, naive_schedule, profile_of
_OUTPUT = {"path": None, "show": True}

def set_figure_output(path=None, show: bool = True) -> None:
    """Where the next plot_plan / plot_battery_plan figure is saved (None = do not save) and whether it is shown."""
    _OUTPUT.update(path=path, show=show)

def finish_figure() -> None:
    fig = plt.gcf()
    if _OUTPUT["path"] is not None: fig.savefig(_OUTPUT["path"], dpi=130, bbox_inches="tight")
    if _OUTPUT["show"]: plt.show()
    plt.close(fig)


def plot_schedule_timeline(csp: ApplianceCSP, assignment: dict, ax=None, title="Appliance schedule") -> None:
    ax = ax or plt.subplots(figsize=(10, 3))[1]
    for i, a in enumerate(csp.appliances):
        if a.name in assignment:
            ax.barh(i, a.duration_h, left=assignment[a.name], color=f"C{i}", edgecolor="k")
            ax.text(assignment[a.name] + a.duration_h / 2, i, f"{a.power_kw} kW", ha="center", va="center", color="w", fontsize=8)
        ax.plot([a.earliest_start, a.latest_end], [i + 0.42] * 2, c="gray", lw=1)
    ax.set_yticks(range(len(csp.appliances))); ax.set_yticklabels([a.name for a in csp.appliances])
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2)); ax.set(title=title, xlabel="hour (gray line = allowed window)"); ax.invert_yaxis()

def plot_battery_plan(steps: list, load: np.ndarray, prices: np.ndarray, cfg: SimConfig = CFG) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6)); hrs = np.arange(len(steps))
    soc = [steps[0].soc_before] + [s.soc_after for s in steps]
    axes[0].step(np.arange(len(soc)), soc, where="post"); axes[0].axhline(cfg.battery_capacity_kwh, c="r", ls="--", lw=0.8)
    axes[0].set(title="Battery energy level", xlabel="hour", ylabel="kWh", ylim=(-0.2, cfg.battery_capacity_kwh + 0.5))
    delta = [s.kwh if s.kind == "charge" else -s.kwh if s.kind == "discharge" else 0 for s in steps]
    axes[1].bar(hrs, delta, color=["tab:green" if d > 0 else "tab:red" if d < 0 else "gray" for d in delta])
    axes[1].set(title="Battery actions (green = charge, red = discharge)", xlabel="hour", ylabel="kWh")
    axes[2].plot(hrs, load, label="net load"); axes[2].plot(hrs, [s.grid_kwh for s in steps], label="grid import")
    axes[2].axhline(cfg.grid_limit_kw, c="r", ls="--", label="grid limit"); ax2 = axes[2].twinx(); ax2.step(hrs, prices, c="orange", alpha=.5, where="mid")
    ax2.set_ylabel("price EUR/kWh", color="orange"); ax2.grid(False)
    axes[2].set(title="Load vs grid import", xlabel="hour", ylabel="kW"); axes[2].legend(fontsize=8)
    plt.tight_layout(); finish_figure()

def plot_plan(plan: PlanResult, actual: Optional[np.ndarray], cfg: SimConfig, appliances: list, title: str = "") -> None:
    prices = cfg.prices(); hrs = np.arange(HORIZON)
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    ax = axes[0, 0]; ax.plot(hrs, plan.forecast, "r-o", ms=3, label="NN forecast")
    if actual is not None: ax.plot(hrs, actual, "k-", label="actual (real)")
    ax.set(title="Forecast vs actual household load", xlabel="hour", ylabel="kW"); ax.legend()
    plot_schedule_timeline(plan.csp, plan.schedule, ax=axes[0, 1], title="Appliance schedule (CSP)")
    soc = [plan.steps[0].soc_before] + [s.soc_after for s in plan.steps]
    axes[1, 0].step(range(len(soc)), soc, where="post"); axes[1, 0].set(title="Battery energy level (plan)", xlabel="hour", ylabel="kWh")
    delta = [s.kwh if s.kind == "charge" else -s.kwh if s.kind == "discharge" else 0 for s in plan.steps]
    axes[1, 1].bar(hrs, delta, color=["tab:green" if d > 0 else "tab:red" if d < 0 else "gray" for d in delta])
    axes[1, 1].set(title="Battery actions (green charge / red discharge)", xlabel="hour", ylabel="kWh")
    ax = axes[2, 0]; ax.bar(hrs - 0.2, plan.total_load, 0.4, label="load before battery", alpha=.7); ax.bar(hrs + 0.2, plan.grid_import, 0.4, label="grid import", alpha=.7)
    ax.axhline(cfg.grid_limit_kw, c="r", ls="--", label="grid limit"); ax.set(title="Hourly grid electricity consumption", xlabel="hour", ylabel="kW"); ax.legend(fontsize=8)
    naive = profile_of(appliances, naive_schedule(plan.csp.appliances)) + plan.forecast
    costs = {"naive schedule,\nno battery": hourly_cost(naive, prices).sum(),
             "CSP schedule,\nno battery": hourly_cost(plan.total_load, prices).sum(),
             "full planner\n(CSP + battery)": plan.planned_cost}
    axes[2, 1].bar(list(costs), list(costs.values()), color=["gray", "tab:blue", "tab:green"])
    axes[2, 1].set(title="Simulated cost comparison (forecast load)", ylabel="EUR (simulated)")
    for i, v in enumerate(costs.values()): axes[2, 1].text(i, v, f"{v:.2f}", ha="center", va="bottom")
    fig.suptitle(title, y=1.0); plt.tight_layout(); finish_figure()
