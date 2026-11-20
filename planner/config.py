"""Simulated configuration (all values are ASSUMPTIONS, not data), seeding helpers."""
from __future__ import annotations
import os, random
import numpy as np
import torch
from dataclasses import dataclass


SEED = 42

EPS = 1e-9          # numerical tolerance for float comparisons

HORIZON = 24        # planning horizon in hours

def set_seed(seed: int = SEED) -> None:
    """Seed every random number generator we use."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

@dataclass(frozen=True)
class SimConfig:
    """SIMULATED parameters (not measured data). Units in the comments."""
    price_offpeak: float = 0.12                         # EUR/kWh
    price_mid: float = 0.20                             # EUR/kWh
    price_peak: float = 0.32                            # EUR/kWh
    peak_hours: tuple = (17, 18, 19, 20)
    offpeak_hours: tuple = (22, 23, 0, 1, 2, 3, 4, 5)
    household_limit_kw: float = 5.0                     # total consumption limit (CSP)
    grid_limit_kw: float = 4.0                          # grid import limit (battery search)
    circuit_limit_kw: float = 3.6                       # per shared circuit
    battery_capacity_kwh: float = 5.0
    battery_step_kwh: float = 0.5                       # discretisation of the energy level
    battery_max_charge_kw: float = 2.5
    battery_max_discharge_kw: float = 2.5
    battery_initial_kwh: float = 2.5
    battery_final_min_kwh: float = 2.5

    def prices(self) -> np.ndarray:
        """Hourly tariff vector of length 24 (EUR/kWh)."""
        p = np.full(HORIZON, self.price_mid, dtype=float)
        p[list(self.peak_hours)] = self.price_peak
        p[list(self.offpeak_hours)] = self.price_offpeak
        return p

CFG = SimConfig()
