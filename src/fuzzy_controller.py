"""Mamdani Fuzzy Logic Controller for EV Charging Urgency.

Implements parameterized triangular and trapezoidal membership functions,
vectorized Mamdani min-max inference, centroid defuzzification, and
high-performance lookup table acceleration.
"""

from typing import Dict, Tuple
import numpy as np


def trimf_scalar(x: float, a: float, b: float, c: float) -> float:
    """Scalar triangular membership function."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (c - x) / (c - b) if c > b else 1.0


def trapmf_scalar(x: float, a: float, b: float, c: float, d: float) -> float:
    """Scalar trapezoidal membership function."""
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (d - x) / (d - c) if d > c else 1.0


class FuzzyUrgencyController:
    """Mamdani Fuzzy Urgency Controller.

    Inputs:
        - Remaining energy demand (kWh)
        - Charging slack (hours)

    Output:
        - Urgency score in [0.0, 1.0]
    """

    def __init__(self, resolution: int = 101, use_lut: bool = True, n_demand: int = 121, n_slack: int = 201):
        self.resolution = resolution
        self.use_lut = use_lut
        self.n_demand = n_demand
        self.n_slack = n_slack
        self.demand_min, self.demand_max = 0.0, 60.0
        self.slack_min, self.slack_max = -5.0, 15.0
        self.u_universe = np.linspace(0.0, 1.0, resolution)

        # Output Urgency Membership Functions on universe [0, 1]
        self.mf_low = np.array([trapmf_scalar(u, 0.0, 0.0, 0.15, 0.35) for u in self.u_universe], dtype=np.float64)
        self.mf_medium = np.array([trimf_scalar(u, 0.25, 0.45, 0.65) for u in self.u_universe], dtype=np.float64)
        self.mf_high = np.array([trimf_scalar(u, 0.55, 0.72, 0.88) for u in self.u_universe], dtype=np.float64)
        self.mf_very_high = np.array([trapmf_scalar(u, 0.78, 0.90, 1.0, 1.0) for u in self.u_universe], dtype=np.float64)

        # 9 Mamdani Rules as specified:
        self.rules = [
            ("low", "critical", "high"),
            ("medium", "critical", "very_high"),
            ("high", "critical", "very_high"),
            ("low", "limited", "medium"),
            ("medium", "limited", "high"),
            ("high", "limited", "high"),
            ("low", "comfortable", "low"),
            ("medium", "comfortable", "low"),
            ("high", "comfortable", "medium"),
        ]

        if self.use_lut:
            self._init_lookup_table()

    def _init_lookup_table(self):
        """Precomputes a 2D lookup table for ultra-fast, deterministic O(1) query."""
        self.demand_vals = np.linspace(self.demand_min, self.demand_max, self.n_demand)
        self.slack_vals = np.linspace(self.slack_min, self.slack_max, self.n_slack)

        self.lut = np.zeros((self.n_demand, self.n_slack), dtype=np.float64)
        for i, d in enumerate(self.demand_vals):
            for j, s in enumerate(self.slack_vals):
                self.lut[i, j] = self.evaluate_exact(d, s)

    def _fuzzify_demand(self, demand_kwh: float) -> Tuple[float, float, float]:
        """Calculates membership degrees for (low, medium, high) demand."""
        mu_low = trapmf_scalar(demand_kwh, 0.0, 0.0, 8.0, 16.0)
        mu_med = trimf_scalar(demand_kwh, 10.0, 20.0, 30.0)
        mu_high = trapmf_scalar(demand_kwh, 24.0, 32.0, 60.0, 60.0)
        return mu_low, mu_med, mu_high

    def _fuzzify_slack(self, slack_hours: float) -> Tuple[float, float, float]:
        """Calculates membership degrees for (critical, limited, comfortable) slack."""
        mu_crit = trapmf_scalar(slack_hours, -5.0, -5.0, 0.25, 0.75)
        mu_lim = trimf_scalar(slack_hours, 0.5, 1.5, 2.5)
        mu_comf = trapmf_scalar(slack_hours, 2.0, 3.5, 15.0, 15.0)
        return mu_crit, mu_lim, mu_comf

    def compute_slack(
        self,
        remaining_energy_kwh: float,
        remaining_time_hours: float,
        max_charging_power_kw: float,
    ) -> float:
        """Computes charging slack in hours: slack = remaining_time - (remaining_energy / max_charging_power)."""
        if remaining_energy_kwh <= 0:
            return float(max(0.0, remaining_time_hours))
        if max_charging_power_kw <= 0:
            return -999.0
        required_charging_time = remaining_energy_kwh / max_charging_power_kw
        return float(remaining_time_hours - required_charging_time)

    def evaluate_exact(self, remaining_energy_kwh: float, slack_hours: float) -> float:
        """Executes exact Mamdani min-max inference and centroid defuzzification."""
        if remaining_energy_kwh <= 0.0:
            return 0.0

        d_low, d_med, d_high = self._fuzzify_demand(remaining_energy_kwh)
        s_crit, s_lim, s_comf = self._fuzzify_slack(slack_hours)

        # Firing strengths of the 9 rules
        r1 = min(d_low, s_crit)     # High
        r2 = min(d_med, s_crit)     # Very High
        r3 = min(d_high, s_crit)    # Very High
        r4 = min(d_low, s_lim)      # Medium
        r5 = min(d_med, s_lim)      # High
        r6 = min(d_high, s_lim)     # High
        r7 = min(d_low, s_comf)     # Low
        r8 = min(d_med, s_comf)     # Low
        r9 = min(d_high, s_comf)    # Medium

        # Aggregate output term activations (max s-norm)
        act_low = max(r7, r8)
        act_med = max(r4, r9)
        act_high = max(r1, r5, r6)
        act_very_high = max(r2, r3)

        # Implication (min t-norm) and Aggregation (max s-norm)
        agg_mf = np.maximum.reduce([
            np.minimum(act_low, self.mf_low),
            np.minimum(act_med, self.mf_medium),
            np.minimum(act_high, self.mf_high),
            np.minimum(act_very_high, self.mf_very_high),
        ])

        sum_w = float(np.sum(agg_mf))
        if sum_w < 1e-12:
            return 0.5

        centroid = float(np.sum(self.u_universe * agg_mf) / sum_w)
        return float(np.clip(centroid, 0.0, 1.0))

    def evaluate(self, remaining_energy_kwh: float, slack_hours: float) -> float:
        """Evaluates fuzzy urgency in [0, 1] using LUT bilinear interpolation or exact evaluation."""
        if remaining_energy_kwh <= 0.0:
            return 0.0

        if not self.use_lut:
            return self.evaluate_exact(remaining_energy_kwh, slack_hours)

        # Bilinear interpolation on precomputed grid
        d = float(np.clip(remaining_energy_kwh, self.demand_min, self.demand_max))
        s = float(np.clip(slack_hours, self.slack_min, self.slack_max))

        # Normalized coordinates
        di = (d - self.demand_min) / (self.demand_max - self.demand_min) * (self.n_demand - 1)
        si = (s - self.slack_min) / (self.slack_max - self.slack_min) * (self.n_slack - 1)

        i0 = int(di)
        i1 = min(i0 + 1, self.n_demand - 1)
        j0 = int(si)
        j1 = min(j0 + 1, self.n_slack - 1)

        alpha = di - i0
        beta = si - j0

        val = (
            (1 - alpha) * (1 - beta) * self.lut[i0, j0]
            + alpha * (1 - beta) * self.lut[i1, j0]
            + (1 - alpha) * beta * self.lut[i0, j1]
            + alpha * beta * self.lut[i1, j1]
        )
        return float(np.clip(val, 0.0, 1.0))
