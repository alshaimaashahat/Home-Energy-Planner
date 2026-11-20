"""Integration layer: forecast -> CSP -> battery search with the feedback loop."""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field, replace
from typing import Callable, Optional
import numpy as np
import pandas as pd
from .config import CFG, EPS, HORIZON, SimConfig
from .csp import ApplianceCSP, Appliance, CSPSolver, CSPStats, DEFAULT_APPLIANCES, explain_no_solution
from .data import make_feature_vector
from .nn import Forecaster
from .search import SearchResult, battery_search, make_battery_problem


@dataclass
class PlanResult:
    success: bool
    forecast: np.ndarray
    csp: Optional[ApplianceCSP] = None
    schedule: dict = field(default_factory=dict)
    appliance_profile: np.ndarray = field(default_factory=lambda: np.zeros(HORIZON))
    total_load: np.ndarray = field(default_factory=lambda: np.zeros(HORIZON))
    steps: list = field(default_factory=list)
    grid_import: np.ndarray = field(default_factory=lambda: np.zeros(HORIZON))
    planned_cost: float = float("nan")
    iterations: int = 0
    feedback_log: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    constraints_ok: dict = field(default_factory=dict)
    csp_stats: Optional[CSPStats] = None
    search_result: Optional[SearchResult] = None

def naive_schedule(appliances: list) -> dict:
    """Reference behaviour: every appliance starts at the beginning of its window (ignores prices, limits, circuits)."""
    return {a.name: a.earliest_start for a in appliances}

def profile_of(appliances: list, schedule: dict) -> np.ndarray:
    prof = np.zeros(HORIZON)
    for a in appliances:
        prof[schedule[a.name]:schedule[a.name] + a.duration_h] += a.power_kw
    return prof

def hourly_cost(grid: np.ndarray, prices: np.ndarray) -> np.ndarray: return np.asarray(grid) * np.asarray(prices)

def execute_plan(steps: list, actual_load: np.ndarray, prices: np.ndarray, cfg: SimConfig = CFG) -> dict:
    """Run a battery plan against a (possibly different) real load. Energy limits are enforced physically."""
    soc, grid, soc_trace = cfg.battery_initial_kwh, np.zeros(HORIZON), [cfg.battery_initial_kwh]
    for h in range(HORIZON):
        kind, amount = (steps[h].kind, steps[h].kwh) if steps else ("idle", 0.0)
        if kind == "charge": q = min(amount, cfg.battery_capacity_kwh - soc); soc += q; grid[h] = actual_load[h] + q
        elif kind == "discharge": q = min(amount, soc, actual_load[h]); soc -= q; grid[h] = actual_load[h] - q
        else: grid[h] = actual_load[h]
        soc_trace.append(soc)
    return {"grid": grid, "soc": np.array(soc_trace), "cost": float(hourly_cost(grid, prices).sum())}

def find_conflict_hours(total_load: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Hours where even maximal discharge cannot bring the grid import under the limit."""
    max_cover = np.minimum(np.minimum(cfg.battery_max_discharge_kw, cfg.battery_capacity_kwh), total_load)
    bad = np.where(total_load - max_cover > cfg.grid_limit_kw + EPS)[0]
    return bad if len(bad) else np.where(total_load > cfg.grid_limit_kw + EPS)[0]

def derive_restriction(csp: ApplianceCSP, schedule: dict, total_load: np.ndarray, cfg: SimConfig,
                       extra: list, blocked: dict) -> Optional[str]:
    """Add one new CSP restriction explaining the battery failure. Returns a description or None if impossible."""
    bad = find_conflict_hours(total_load, cfg)
    for h in sorted(bad, key=lambda t: -total_load[t]):
        active = sorted([a for a in csp.appliances if schedule[a.name] <= h < schedule[a.name] + a.duration_h],
                        key=lambda a: -a.power_kw)
        for a, b in itertools.combinations(active, 2):
            pair = tuple(sorted((a.name, b.name)))
            if pair not in [tuple(sorted(p)) for p in extra]:
                extra.append(pair); return f"hour {h}: {a.name} + {b.name} overlap -> forbid overlapping"
        if len(active) == 1 and h not in blocked.get(active[0].name, set()):
            blocked.setdefault(active[0].name, set()).add(h); return f"hour {h}: {active[0].name} alone too high -> block hour {h} for it"
    return None

def relaxed_limit_report(kind: str, base: float, try_fn: Callable[[float], bool], max_steps: int) -> str:
    for k in range(1, max_steps + 1):
        lim = base * (1 + 0.1 * k)
        if try_fn(lim):
            return f"Feasible if the {kind} limit is relaxed by {10 * k}% to {lim:.2f} kW (violation of the original {base:.2f} kW limit: up to {lim - base:.2f} kW)."
    return f"Still infeasible after relaxing the {kind} limit by {10 * max_steps}%."

def plan_day(forecast: np.ndarray, cfg: SimConfig = CFG, appliances: list = DEFAULT_APPLIANCES,
             max_iterations: int = 5, use_astar: bool = True) -> PlanResult:
    """CSP -> battery search with the feedback loop. Never returns an invalid plan silently."""
    forecast = np.asarray(forecast, float); prices = cfg.prices()
    result = PlanResult(False, forecast); extra, blocked = [], {}
    for it in range(1, max_iterations + 1):
        result.iterations = it
        csp = ApplianceCSP(appliances, forecast, cfg.household_limit_kw, cfg.circuit_limit_kw, prices, tuple(extra), blocked)
        schedule, cstats = CSPSolver(True, True).solve(csp); result.csp, result.csp_stats = csp, cstats
        if schedule is None:
            result.messages.append(explain_no_solution(csp, cstats) + (f" (after {it - 1} feedback restriction(s))" if it > 1 else ""))
            def try_csp(lim):
                return CSPSolver(True, True).solve(ApplianceCSP(appliances, forecast, lim, cfg.circuit_limit_kw, prices))[0] is not None
            result.messages.append(relaxed_limit_report("household", cfg.household_limit_kw, try_csp, 5)); return result
        profile = profile_of(appliances, schedule); total = forecast + profile
        problem = make_battery_problem(total, cfg, prices)
        search = battery_search(problem, use_astar); result.search_result = search
        if search.found:
            result.success, result.schedule, result.appliance_profile, result.total_load = True, schedule, profile, total
            result.steps, result.planned_cost = search.steps, search.total_cost
            result.grid_import = np.array([s.grid_kwh for s in search.steps])
            soc = [s.soc_after for s in search.steps]
            result.constraints_ok = {
                "CSP constraints (independent check)": not csp.check_solution(schedule),
                "household limit (consumption)": bool((total <= cfg.household_limit_kw + 1e-6).all()),
                "grid limit (import)": bool((result.grid_import <= cfg.grid_limit_kw + 1e-6).all()),
                "battery within [0, capacity]": bool(min(soc) >= -1e-9 and max(soc) <= cfg.battery_capacity_kwh + 1e-9),
                "battery final energy >= minimum": bool(soc[-1] >= cfg.battery_final_min_kwh - 1e-9)}
            return result
        note = derive_restriction(csp, schedule, total, cfg, extra, blocked)
        result.feedback_log.append({"iteration": it, "battery_failure": search.message, "restriction": note or "none possible"})
        if note is None: break
    result.messages.append(f"Battery planning failed after {result.iterations} iteration(s): no further restriction can fix it.")
    result.messages.append(relaxed_limit_report("grid", cfg.grid_limit_kw,
        lambda lim: battery_search(make_battery_problem(total, replace(cfg, grid_limit_kw=lim), prices), True).found, 10))
    return result

def evaluate_on_actual(plan: PlanResult, actual: np.ndarray, appliances: list, cfg: SimConfig = CFG) -> dict:
    """Simulated cost and limit violations when the plan meets the REAL load. Failed plans fall back to the naive,
    battery-less behaviour (and are counted)."""
    prices = cfg.prices()
    if plan.success:
        prof, steps = plan.appliance_profile, plan.steps
    else:
        prof, steps = profile_of(appliances, naive_schedule(appliances)), []
    run = execute_plan(steps, actual + prof, prices, cfg) if steps else execute_plan([], actual + prof, prices, cfg)
    return {"cost": run["cost"], "viol_household": int(((actual + prof) > cfg.household_limit_kw + 1e-9).sum()),
            "viol_grid": int((run["grid"] > cfg.grid_limit_kw + 1e-9).sum()), "failed": int(not plan.success), "grid": run["grid"], "soc": run["soc"]}

def forecast_for_date(date: pd.Timestamp, daily_mat: pd.DataFrame, fc: Forecaster) -> np.ndarray:
    date = pd.Timestamp(date).normalize()
    for back in (1, 7):
        d = date - pd.Timedelta(days=back)
        if d not in daily_mat.index or daily_mat.loc[d].isna().any():
            raise ValueError(f"history for {d.date()} (needed to forecast {date.date()}) is missing or invalid")
    return fc.predict_batch(make_feature_vector(daily_mat.loc[date - pd.Timedelta(days=1)].values,
                                                 daily_mat.loc[date - pd.Timedelta(days=7)].values, date)[None, :])[0]


def run_home_energy_planner(date, daily_mat: pd.DataFrame, fc: Forecaster, cfg: SimConfig = CFG, appliances: list = DEFAULT_APPLIANCES,
                            verbose: bool = True, plot: bool = False) -> PlanResult:
    """Main entry point: history -> NN forecast -> CSP schedule -> battery search (+feedback) -> verified plan.
    All costs are SIMULATED. On failure a PlanResult with success=False and explanatory messages is returned."""
    date = pd.Timestamp(date).normalize()
    try:
        forecast = forecast_for_date(date, daily_mat, fc)
    except ValueError as exc:
        print(f"[ERROR] cannot forecast: {exc}"); res = PlanResult(False, np.zeros(HORIZON)); res.messages.append(str(exc)); return res
    plan = plan_day(forecast, cfg, appliances)
    actual = daily_mat.loc[date].values if date in daily_mat.index and not daily_mat.loc[date].isna().any() else None
    plan.actual = actual
    if not plan.success:
        print(f"[PLAN FAILED] {date.date()}")
        for m in plan.messages: print("  -", m)
        for f in plan.feedback_log: print("  feedback:", f)
        return plan
    if verbose:
        prices = cfg.prices()
        print(f"=== Home Energy Planner - {date.date()} (all costs SIMULATED) ===")
        print(f"1) INPUT history (real): day-1 mean {daily_mat.loc[date - pd.Timedelta(days=1)].mean():.2f} kW, day-7 mean {daily_mat.loc[date - pd.Timedelta(days=7)].mean():.2f} kW")
        print(f"2) NN forecast: mean {plan.forecast.mean():.2f} kW, peak {plan.forecast.max():.2f} kW at {int(plan.forecast.argmax()):02d}:00")
        print(f"3) CSP (BT+FC+AC3): assignments={plan.csp_stats.assignments}, backtracks={plan.csp_stats.backtracks}, domain reductions={plan.csp_stats.domain_reductions}")
        print(pd.DataFrame([{"appliance": a.name, "start": f"{plan.schedule[a.name]:02d}:00", "end": f"{plan.schedule[a.name] + a.duration_h:02d}:00",
                             "power kW": a.power_kw, "window": f"{a.earliest_start:02d}-{a.latest_end:02d}"} for a in appliances]).to_string(index=False))
        sr = plan.search_result
        print(f"4) Battery search ({sr.algorithm}): expanded={sr.nodes_expanded}, generated={sr.nodes_generated}, time={sr.time_s * 1e3:.1f} ms")
        print(f"5) Feedback iterations used: {plan.iterations}" + (f" - {plan.feedback_log}" if plan.feedback_log else " (no battery failure occurred)"))
        print(pd.DataFrame([{"hour": s.hour, "action": s.kind, "kWh": s.kwh, "battery after": s.soc_after, "grid kWh": round(s.grid_kwh, 2),
                             "price": prices[s.hour], "cost EUR": round(s.cost, 3)} for s in plan.steps]).to_string(index=False))
        print(f"6) Simulated cost for the planned (forecast) load: {plan.planned_cost:.3f} EUR")
        if actual is not None:
            ev = evaluate_on_actual(plan, actual, appliances, cfg)
            print(f"   Executed on the REAL load: simulated cost {ev['cost']:.3f} EUR | household-limit violations: {ev['viol_household']} h | grid-limit violations: {ev['viol_grid']} h")
        print("7) Constraint check:")
        for k, ok in plan.constraints_ok.items(): print(f"   [{'OK' if ok else 'FAIL'}] {k}")
    if plot:
        from .plotting import plot_plan
        plot_plan(plan, actual, cfg, appliances, title=f"Home Energy Planner - {date.date()} (simulated costs)")
    return plan
