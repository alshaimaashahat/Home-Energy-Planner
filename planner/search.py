"""Battery planning with Uniform Cost Search and A*, plus an independent Dynamic-Programming checker."""
from __future__ import annotations
import heapq, itertools, math, time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from .config import CFG, EPS, SimConfig


@dataclass
class BatteryProblem:
    load_kwh: np.ndarray          # net load per hour (forecast + appliances)
    prices: np.ndarray            # EUR/kWh per hour
    capacity_kwh: float
    step_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    grid_limit_kw: float
    initial_kwh: float
    final_min_kwh: float

    def __post_init__(self):
        self.load_kwh = np.asarray(self.load_kwh, float); self.prices = np.asarray(self.prices, float)
        if self.load_kwh.ndim != 1 or self.load_kwh.shape != self.prices.shape: raise ValueError("load and prices must be 1-D arrays of equal length")
        if len(self.load_kwh) and (not np.isfinite(self.load_kwh).all() or (self.load_kwh < 0).any()): raise ValueError("load must be finite and >= 0")
        if len(self.prices) and (not np.isfinite(self.prices).all() or (self.prices < 0).any()): raise ValueError("prices must be finite and >= 0")
        if self.capacity_kwh <= 0 or self.step_kwh <= 0: raise ValueError("battery capacity and step must be > 0")
        if self.max_charge_kw < 0 or self.max_discharge_kw < 0 or self.grid_limit_kw <= 0: raise ValueError("invalid power limits")
        ratio = self.capacity_kwh / self.step_kwh
        if abs(ratio - round(ratio)) > 1e-6: raise ValueError("capacity must be a multiple of the step")
        if not (0 <= self.initial_kwh <= self.capacity_kwh) or not (0 <= self.final_min_kwh <= self.capacity_kwh):
            raise ValueError("initial/final battery energy must lie within [0, capacity]")
        self.horizon = len(self.load_kwh)
        self.n_levels = int(round(ratio)) + 1
        self.init_level = int(round(self.initial_kwh / self.step_kwh))
        self.final_level = int(math.ceil(self.final_min_kwh / self.step_kwh - 1e-9))
        self.max_charge_units = int(math.floor(self.max_charge_kw / self.step_kwh + 1e-9))
        self.max_discharge_units = int(math.floor(self.max_discharge_kw / self.step_kwh + 1e-9))
        # suffix tables for the heuristic
        self.suffix_load = np.append(np.cumsum(self.load_kwh[::-1])[::-1], 0.0)
        self.suffix_min_price = np.append(np.minimum.accumulate(self.prices[::-1])[::-1], 0.0)

    def is_goal(self, hour: int, level: int) -> bool: return hour == self.horizon and level >= self.final_level

    def successors(self, hour: int, level: int) -> list:
        """List of (kind, units, next_level, grid_kwh, step_cost) for all LEGAL actions."""
        L, price, out = self.load_kwh[hour], self.prices[hour], []
        if L <= self.grid_limit_kw + EPS: out.append(("idle", 0, level, L, price * L))
        for q in range(1, self.max_charge_units + 1):
            if level + q >= self.n_levels: break
            grid = L + q * self.step_kwh
            if grid <= self.grid_limit_kw + EPS: out.append(("charge", q, level + q, grid, price * grid))
        for q in range(1, min(self.max_discharge_units, level) + 1):
            if q * self.step_kwh > L + EPS: break                      # no export to the grid
            grid = L - q * self.step_kwh
            if grid <= self.grid_limit_kw + EPS: out.append(("discharge", q, level - q, grid, price * grid))
        return out

    def heuristic(self, hour: int, level: int) -> float:
        """Admissible + consistent lower bound on the remaining cost (see Markdown)."""
        energy_to_buy = self.suffix_load[hour] - level * self.step_kwh + self.final_level * self.step_kwh
        return float(self.suffix_min_price[hour] * max(0.0, energy_to_buy))

@dataclass
class BatteryStep:
    hour: int; kind: str; kwh: float; soc_before: float; soc_after: float; grid_kwh: float; cost: float

@dataclass
class SearchResult:
    algorithm: str
    found: bool
    steps: list = field(default_factory=list)
    total_cost: float = float("inf")
    nodes_expanded: int = 0
    nodes_generated: int = 0
    reopened: int = 0
    time_s: float = 0.0
    message: str = ""

def battery_search(problem: BatteryProblem, use_heuristic: bool, trace: Optional[list] = None) -> SearchResult:
    """Best-first graph search. use_heuristic=False -> Uniform Cost Search, True -> A*.
    Nodes generated = all successor nodes created by expansions; expanded = nodes popped and expanded.
    If `trace` is a list, one tuple (expansion no., hour, level, g, f) is appended per expansion (search trace)."""
    t0 = time.perf_counter()
    res = SearchResult("A*" if use_heuristic else "UCS", False)
    start = (0, problem.init_level)
    best_g, parent, expanded_set, counter = {start: 0.0}, {}, set(), itertools.count()
    h0 = problem.heuristic(*start) if use_heuristic else 0.0
    frontier = [(h0, next(counter), 0.0, start)]
    while frontier:
        f, _, g, state = heapq.heappop(frontier)
        if g > best_g[state] + 1e-12: continue                       # stale queue entry
        if problem.is_goal(*state):
            steps, s = [], state
            while s in parent:
                p, kind, q, grid, c = parent[s]
                steps.append(BatteryStep(p[0], kind, q * problem.step_kwh, p[1] * problem.step_kwh,
                                         s[1] * problem.step_kwh, grid, c)); s = p
            res.steps, res.total_cost, res.found = steps[::-1], g, True
            break
        if state in expanded_set: res.reopened += 1
        expanded_set.add(state); res.nodes_expanded += 1
        hour, level = state
        if trace is not None: trace.append((res.nodes_expanded, hour, level, round(g, 4), round(f, 4)))
        if hour == problem.horizon: continue                        # terminal but goal not satisfied (level too low)
        for kind, q, nlevel, grid, cost in problem.successors(hour, level):
            res.nodes_generated += 1
            nxt, g2 = (hour + 1, nlevel), g + cost
            if g2 < best_g.get(nxt, float("inf")) - 1e-12:
                best_g[nxt] = g2; parent[nxt] = (state, kind, q, grid, cost)
                heapq.heappush(frontier, (g2 + (problem.heuristic(*nxt) if use_heuristic else 0.0), next(counter), g2, nxt))
    res.time_s = time.perf_counter() - t0
    if not res.found:
        res.message = "NO SOLUTION: no battery plan keeps grid import under the limit (battery power/energy too small for the load)."
    return res

def battery_dp(problem: BatteryProblem) -> np.ndarray:
    """Exact cost-to-go V[hour, level] (inf if infeasible). Independent of battery_search."""
    H, N, step = problem.horizon, problem.n_levels, problem.step_kwh
    V = np.full((H + 1, N), np.inf); V[H, problem.final_level:] = 0.0
    for h in range(H - 1, -1, -1):
        L, p = problem.load_kwh[h], problem.prices[h]
        for l in range(N):
            for l2 in range(N):
                d = l2 - l
                if d > problem.max_charge_units or -d > problem.max_discharge_units: continue
                grid = L + d * step
                if d < 0 and -d * step > L + EPS: continue
                if grid > problem.grid_limit_kw + EPS: continue
                V[h, l] = min(V[h, l], p * grid + V[h + 1, l2])
    return V

def format_trace(trace: list, problem: BatteryProblem, limit: int = 12) -> str:
    """Readable search trace: the order in which states were expanded, with path cost g and priority f."""
    lines = [f"{'#':>3} {'hour':>4} {'battery kWh':>11} {'g (EUR)':>9} {'f=g+h':>9}"]
    for n, h, l, g, f in trace[:limit]:
        lines.append(f"{n:>3} {h:>4} {l * problem.step_kwh:>11.1f} {g:>9.3f} {f:>9.3f}")
    if len(trace) > limit: lines.append(f"... ({len(trace) - limit} more expansions)")
    return "\n".join(lines)

def dp_optimal_cost(problem: BatteryProblem) -> float: return float(battery_dp(problem)[0, problem.init_level])

def make_battery_problem(load_kwh, cfg: SimConfig = CFG, prices=None) -> BatteryProblem:
    return BatteryProblem(load_kwh, cfg.prices() if prices is None else prices, cfg.battery_capacity_kwh, cfg.battery_step_kwh,
                          cfg.battery_max_charge_kw, cfg.battery_max_discharge_kw, cfg.grid_limit_kw,
                          cfg.battery_initial_kwh, cfg.battery_final_min_kwh)

def check_heuristic_admissible(problem: BatteryProblem, V: Optional[np.ndarray] = None) -> bool:
    """h(s) <= exact cost-to-go for EVERY reachable-in-principle state with finite cost-to-go."""
    V = battery_dp(problem) if V is None else V
    return all(problem.heuristic(h, l) <= V[h, l] + 1e-9 for h in range(problem.horizon + 1) for l in range(problem.n_levels) if np.isfinite(V[h, l]))

def random_small_problem(rng: np.random.Generator) -> BatteryProblem:
    H = 6
    return BatteryProblem(rng.uniform(0.2, 2.5, H), rng.uniform(0.05, 0.4, H), capacity_kwh=3.0, step_kwh=1.0,
                          max_charge_kw=2.0, max_discharge_kw=2.0, grid_limit_kw=float(rng.uniform(2.0, 4.0)),
                          initial_kwh=float(rng.integers(0, 4)), final_min_kwh=float(rng.integers(0, 2)))
