"""Appliance-scheduling CSP: Backtracking, Forward Checking and AC-3 (hand-written, switchable)."""
from __future__ import annotations
import itertools, time
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Optional
import numpy as np
from .config import CFG, EPS, HORIZON, SimConfig


def intervals_overlap(s1: int, d1: int, s2: int, d2: int) -> bool:
    """True if [s1, s1+d1) and [s2, s2+d2) share at least one hour."""
    return s1 < s2 + d2 and s2 < s1 + d1

@dataclass(frozen=True)
class Appliance:
    """SIMULATED flexible appliance. Window: may run in hours [earliest_start, latest_end)."""
    name: str
    duration_h: int
    power_kw: float
    earliest_start: int
    latest_end: int
    circuit: str
    after: Optional[str] = None      # name of an appliance that must have finished before this one starts

    def __post_init__(self):
        if self.duration_h < 1: raise ValueError(f"{self.name}: duration must be >= 1 hour")
        if self.power_kw <= 0: raise ValueError(f"{self.name}: power must be > 0")
        if not (0 <= self.earliest_start < self.latest_end <= HORIZON):
            raise ValueError(f"{self.name}: invalid window [{self.earliest_start}, {self.latest_end})")

APPLIANCE_POOL = [
    Appliance("washing_machine", 2, 2.0, 7, 22, "laundry"),
    Appliance("dryer",           2, 2.5, 8, 23, "laundry", after="washing_machine"),
    Appliance("dishwasher",      2, 1.8, 12, 24, "kitchen"),
    Appliance("ev_charger",      4, 3.3, 0, 24, "garage"),
    Appliance("electric_oven",   1, 2.2, 11, 21, "kitchen"),
    Appliance("water_heater",    2, 2.4, 0, 24, "utility"),
    Appliance("pool_pump",       3, 1.1, 6, 20, "utility"),
    Appliance("iron",            1, 1.2, 9, 21, "laundry"),
]

DEFAULT_APPLIANCES = APPLIANCE_POOL[:5]     # washer, dryer, dishwasher, EV charger, oven

class ApplianceCSP:
    """CSP: variables = appliance start hours (see the Markdown cell above)."""
    def __init__(self, appliances: list, forecast_kw: np.ndarray, limit_kw: float, circuit_limit_kw: float,
                 prices: Optional[np.ndarray] = None, extra_no_overlap: tuple = (), blocked_hours: Optional[dict] = None):
        forecast_kw = np.asarray(forecast_kw, dtype=float)
        if forecast_kw.shape != (HORIZON,) or not np.isfinite(forecast_kw).all():
            raise ValueError(f"forecast must be 24 finite values, got shape {forecast_kw.shape}")
        if limit_kw <= 0 or circuit_limit_kw <= 0: raise ValueError("power limits must be positive")
        names = [a.name for a in appliances]
        if len(set(names)) != len(names): raise ValueError("appliance names must be unique")
        for a in appliances:
            if a.after is not None and a.after not in names:
                raise ValueError(f"{a.name} must follow unknown appliance '{a.after}'")
        self.appliances, self.by_name = list(appliances), {a.name: a for a in appliances}
        self.variables = names
        self.forecast, self.limit, self.circuit_limit = forecast_kw, float(limit_kw), float(circuit_limit_kw)
        self.extra_no_overlap = tuple(tuple(sorted(p)) for p in extra_no_overlap)
        self.blocked_hours = {k: set(v) for k, v in (blocked_hours or {}).items()}
        self.domains = {a.name: self._initial_domain(a, prices) for a in appliances}
        # binary constraints: constraints[(x, y)] = list of functions f(start_x, start_y) -> bool
        self.constraints: dict = {}
        for a in appliances:
            if a.after is not None:
                p = self.by_name[a.after]
                self._add(p.name, a.name, lambda sp, sa, p=p: sa >= sp + p.duration_h)          # dryer after washer
        for a, b in itertools.combinations(appliances, 2):
            shared = a.circuit == b.circuit and a.power_kw + b.power_kw > self.circuit_limit + EPS
            if shared or tuple(sorted((a.name, b.name))) in self.extra_no_overlap:
                self._add(a.name, b.name, lambda sa, sb, a=a, b=b: not intervals_overlap(sa, a.duration_h, sb, b.duration_h))
        self.neighbors = {v: sorted({y for (x, y) in self.constraints if x == v}) for v in self.variables}

    def _add(self, x: str, y: str, fn: Callable[[int, int], bool]) -> None:
        self.constraints.setdefault((x, y), []).append(fn)
        self.constraints.setdefault((y, x), []).append(lambda sy, sx, fn=fn: fn(sx, sy))

    def _initial_domain(self, a: Appliance, prices) -> list:
        dom = []
        for s in range(a.earliest_start, a.latest_end - a.duration_h + 1):
            hrs = range(s, s + a.duration_h)
            if any(h in self.blocked_hours.get(a.name, ()) for h in hrs): continue
            if (self.forecast[s:s + a.duration_h] + a.power_kw > self.limit + EPS).any(): continue   # node consistency
            dom.append(s)
        if prices is not None:                                  # value ordering: cheapest start first
            dom.sort(key=lambda s: (float(np.sum(prices[s:s + a.duration_h])) * a.power_kw, s))
        return dom

    def arcs(self) -> list: return list(self.constraints.keys())
    def allowed(self, x: str, vx: int, y: str, vy: int) -> bool:
        return all(f(vx, vy) for f in self.constraints[(x, y)])

    def load_profile(self, assignment: dict) -> np.ndarray:
        """Appliance power per hour (kW) for a (partial) assignment, WITHOUT the forecast."""
        prof = np.zeros(HORIZON)
        for n, s in assignment.items():
            a = self.by_name[n]; prof[s:s + a.duration_h] += a.power_kw
        return prof

    def consistent(self, assignment: dict, var: str, val: int) -> bool:
        """Would var=val be consistent with the current partial assignment?"""
        for other, so in assignment.items():
            if other != var and (var, other) in self.constraints and not self.allowed(var, val, other, so):
                return False
        trial = dict(assignment); trial[var] = val
        return bool((self.forecast + self.load_profile(trial) <= self.limit + EPS).all())

    def check_solution(self, assignment: dict) -> list:
        """INDEPENDENT verification from the raw appliance specs. Returns a list of violations ([] = valid)."""
        bad = []
        for a in self.appliances:
            if a.name not in assignment: bad.append(f"{a.name}: not scheduled"); continue
            s = assignment[a.name]
            if s < a.earliest_start or s + a.duration_h > a.latest_end: bad.append(f"{a.name}: outside window")
            if any(h in self.blocked_hours.get(a.name, ()) for h in range(s, s + a.duration_h)): bad.append(f"{a.name}: runs in a blocked hour")
            if a.after and assignment.get(a.after) is not None and s < assignment[a.after] + self.by_name[a.after].duration_h:
                bad.append(f"{a.name}: starts before {a.after} finished")
        for a, b in itertools.combinations(self.appliances, 2):
            if a.name in assignment and b.name in assignment:
                forbidden = (a.circuit == b.circuit and a.power_kw + b.power_kw > self.circuit_limit + EPS) or \
                            tuple(sorted((a.name, b.name))) in self.extra_no_overlap
                if forbidden and intervals_overlap(assignment[a.name], a.duration_h, assignment[b.name], b.duration_h):
                    bad.append(f"{a.name} and {b.name} overlap but must not")
        over = np.where(self.forecast + self.load_profile(assignment) > self.limit + 1e-6)[0]
        if len(over): bad.append(f"household limit {self.limit:.2f} kW exceeded in hours {over.tolist()}")
        return bad

@dataclass
class CSPStats:
    assignments: int = 0
    rejected: int = 0
    backtracks: int = 0
    domain_reductions: int = 0
    ac3_revisions: int = 0
    time_s: float = 0.0
    aborted: bool = False

class CSPSolver:
    """Backtracking search with optional Forward Checking and AC-3 pre-processing (all implemented here)."""
    def __init__(self, use_forward_checking: bool = False, use_ac3: bool = False, node_limit: int = 100_000):
        self.fc, self.ac3, self.node_limit = use_forward_checking, use_ac3, node_limit

    @property
    def name(self) -> str:
        return "BT" + ("+FC" if self.fc else "") + ("+AC3" if self.ac3 else "")

    def solve(self, csp: ApplianceCSP) -> tuple:
        stats, t0 = CSPStats(), time.perf_counter()
        domains = {v: list(d) for v, d in csp.domains.items()}
        solution = None
        if all(domains.values()):
            if not self.ac3 or self._ac3(csp, domains, stats):
                solution = self._backtrack(csp, {}, domains, stats)
        stats.time_s = time.perf_counter() - t0
        return solution, stats

    # ---- AC-3: make every arc (x, y) consistent; re-queue neighbours when a domain shrinks
    def _ac3(self, csp, domains, stats) -> bool:
        queue = deque(csp.arcs())
        while queue:
            x, y = queue.popleft(); stats.ac3_revisions += 1
            keep = [vx for vx in domains[x] if any(csp.allowed(x, vx, y, vy) for vy in domains[y])]
            removed = len(domains[x]) - len(keep)
            if removed:
                stats.domain_reductions += removed; domains[x] = keep
                if not keep: return False                       # domain wipe-out -> no solution
                queue.extend((z, x) for z in csp.neighbors[x] if z != y)
        return True

    # ---- Forward checking: prune unassigned domains against the current partial assignment
    def _forward_check(self, csp, assignment, domains, stats) -> bool:
        for other in csp.variables:
            if other in assignment: continue
            keep = [v for v in domains[other] if csp.consistent(assignment, other, v)]
            stats.domain_reductions += len(domains[other]) - len(keep); domains[other] = keep
            if not keep: return False
        return True

    def _backtrack(self, csp, assignment, domains, stats):
        if len(assignment) == len(csp.variables): return dict(assignment)
        var = next(v for v in csp.variables if v not in assignment)       # static variable order
        for val in domains[var]:
            if stats.assignments + stats.rejected >= self.node_limit:
                stats.aborted = True; return None
            if not csp.consistent(assignment, var, val):
                stats.rejected += 1; continue
            stats.assignments += 1; assignment[var] = val
            next_domains, ok = domains, True
            if self.fc:
                next_domains = {k: list(v) for k, v in domains.items()}
                ok = self._forward_check(csp, assignment, next_domains, stats)
            if ok:
                result = self._backtrack(csp, assignment, next_domains, stats)
                if result is not None: return result
                if stats.aborted: return None
            del assignment[var]; stats.backtracks += 1
        return None

def explain_no_solution(csp: ApplianceCSP, stats: CSPStats) -> str:
    """Human-readable reason why no schedule was found."""
    empty = [v for v in csp.variables if not csp.domains[v]]
    if empty:
        return ("NO SOLUTION: no valid start hour exists for: " + ", ".join(empty) +
                f" (window, blocked hours or the {csp.limit:.2f} kW household limit leave no start hour).")
    if stats.aborted: return "UNDECIDED: node limit reached before a solution or a proof of impossibility."
    return "NO SOLUTION: the precedence, shared-circuit and household-limit constraints cannot be satisfied together."

def make_csp_instance(n_appliances: int, seed: int, forecast_kw: np.ndarray, cfg: SimConfig = CFG) -> ApplianceCSP:
    """Seeded benchmark instance: first n appliances of the pool, windows randomly narrowed."""
    rng = np.random.default_rng(seed); apps = []
    for a in APPLIANCE_POOL[:n_appliances]:
        lo = a.earliest_start + int(rng.integers(0, 3)); hi = a.latest_end - int(rng.integers(0, 3))
        if hi - lo < a.duration_h: lo, hi = a.earliest_start, a.latest_end
        apps.append(replace(a, earliest_start=lo, latest_end=hi))
    return ApplianceCSP(apps, forecast_kw, cfg.household_limit_kw, cfg.circuit_limit_kw)

SOLVER_VARIANTS = {"BT": (False, False), "BT+FC": (True, False), "BT+FC+AC3": (True, True)}
