"""Chronological schedule simulation, constraint verification, and objective evaluation."""

from typing import List, Literal, Tuple
import numpy as np
from .models import EV, StationConfig, SimulationMetrics
from .fuzzy_controller import FuzzyUrgencyController
from .repair import repair_slot_proportional, repair_slot_fuzzy_urgency


def simulate_and_repair_schedule(
    candidate_schedule: np.ndarray,  # Shape: (N, T) in kW
    evs: List[EV],
    station: StationConfig,
    repair_method: Literal["proportional", "fuzzy_urgency"] = "proportional",
    fuzzy_controller: FuzzyUrgencyController = None,
    epsilon: float = 0.01,
    delta: float = 0.1,
    numerical_tolerance: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically simulates and repairs a candidate charging schedule.

    Args:
        candidate_schedule: 2D array of shape (N, T) representing proposed charging powers.
        evs: Fleet of EV objects.
        station: StationConfig object.
        repair_method: 'proportional' or 'fuzzy_urgency'.
        fuzzy_controller: FuzzyUrgencyController instance (required if fuzzy_urgency).
        epsilon: Parameter for fuzzy urgency weighting.
        delta: Parameter for candidate request weighting.
        numerical_tolerance: Numerical comparison tolerance.

    Returns:
        repaired_schedule: Shape (N, T) strictly satisfying all constraints.
        delivered_energy: Shape (N,) total energy delivered to each EV (kWh).
        unmet_energy: Shape (N,) unmet energy for each EV (kWh).
    """
    n_evs = len(evs)
    n_slots = station.num_slots
    slot_dt = station.slot_duration_hours

    if candidate_schedule.shape != (n_evs, n_slots):
        raise ValueError(
            f"Schedule shape {candidate_schedule.shape} does not match (N={n_evs}, T={n_slots})."
        )

    if repair_method == "fuzzy_urgency" and fuzzy_controller is None:
        fuzzy_controller = FuzzyUrgencyController()

    repaired_schedule = np.zeros((n_evs, n_slots), dtype=np.float64)
    remaining_energy = np.array([ev.required_energy_kwh for ev in evs], dtype=np.float64)

    for t in range(n_slots):
        cap_t = station.get_capacity(t)
        cand_t = candidate_schedule[:, t]

        if repair_method == "proportional":
            slot_power = repair_slot_proportional(
                candidate_power=cand_t,
                evs=evs,
                station_capacity_kw=cap_t,
                remaining_energy_kwh=remaining_energy,
                slot_idx=t,
                slot_duration_hours=slot_dt,
                numerical_tolerance=numerical_tolerance,
            )
        elif repair_method == "fuzzy_urgency":
            slot_power = repair_slot_fuzzy_urgency(
                candidate_power=cand_t,
                evs=evs,
                station_capacity_kw=cap_t,
                remaining_energy_kwh=remaining_energy,
                slot_idx=t,
                fuzzy_controller=fuzzy_controller,
                slot_duration_hours=slot_dt,
                epsilon=epsilon,
                delta=delta,
                numerical_tolerance=numerical_tolerance,
            )
        else:
            raise ValueError(f"Unknown repair method: {repair_method}")

        repaired_schedule[:, t] = slot_power

        # Update remaining energy
        delivered_t = slot_power * slot_dt
        remaining_energy = np.maximum(0.0, remaining_energy - delivered_t)

    # Final energy accounting
    total_delivered = np.sum(repaired_schedule * slot_dt, axis=1)
    req_energy = np.array([ev.required_energy_kwh for ev in evs], dtype=np.float64)
    # Clip delivered energy to required (no excess delivery)
    total_delivered = np.minimum(req_energy, total_delivered)
    unmet = np.maximum(0.0, req_energy - total_delivered)

    return repaired_schedule, total_delivered, unmet


def calculate_objective(
    repaired_schedule: np.ndarray,
    unmet_energy: np.ndarray,
    station: StationConfig,
    lambda_unmet: float = 50.0,
) -> Tuple[float, float, float]:
    """Calculates the scalar optimization objective and its components.

    Objective: J = J_cost + lambda_unmet * J_unmet
    Where:
        J_cost = sum_t (sum_i P[i,t] * price[t] * delta_t)
        J_unmet = sum_i (unmet_energy[i])

    Returns:
        total_objective, electricity_cost, unmet_energy_sum
    """
    slot_dt = station.slot_duration_hours
    # Total station power at each slot t (kW)
    station_power_per_slot = np.sum(repaired_schedule, axis=0)  # Shape (T,)

    # Electricity cost in currency units
    electricity_cost = float(np.sum(station_power_per_slot * station.electricity_prices * slot_dt))

    # Total unmet energy in kWh
    total_unmet = float(np.sum(unmet_energy))

    total_objective = electricity_cost + (lambda_unmet * total_unmet)
    return total_objective, electricity_cost, total_unmet


def evaluate_schedule_metrics(
    repaired_schedule: np.ndarray,
    evs: List[EV],
    station: StationConfig,
    delivered_energy: np.ndarray,
    unmet_energy: np.ndarray,
    runtime_seconds: float = 0.0,
    lambda_unmet: float = 50.0,
    energy_tolerance_kwh: float = 0.01,
    numerical_tolerance: float = 1e-6,
) -> SimulationMetrics:
    """Evaluates comprehensive metrics for a repaired schedule."""
    n_evs = len(evs)
    req_energy = np.array([ev.required_energy_kwh for ev in evs], dtype=np.float64)
    total_req = float(np.sum(req_energy))
    total_del = float(np.sum(delivered_energy))
    total_unmet = float(np.sum(unmet_energy))

    # 1. Deadline satisfaction (% of EVs fully charged within tolerance)
    satisfied_count = sum(1 for u in unmet_energy if u <= energy_tolerance_kwh)
    deadline_sat_pct = (satisfied_count / n_evs) * 100.0

    # 2. Objective components
    obj_val, elec_cost, _ = calculate_objective(
        repaired_schedule, unmet_energy, station, lambda_unmet
    )

    # 3. Capacity violations
    station_power = np.sum(repaired_schedule, axis=0)
    caps = np.array([station.get_capacity(t) for t in range(station.num_slots)])
    excess_power = np.maximum(0.0, station_power - caps)
    max_cap_violation = float(np.max(excess_power))
    violating_slots = int(np.sum(excess_power > numerical_tolerance))

    # 4. Jain's Fairness Index
    service_ratios = np.zeros(n_evs, dtype=np.float64)
    for i in range(n_evs):
        if req_energy[i] > numerical_tolerance:
            service_ratios[i] = min(1.0, delivered_energy[i] / req_energy[i])
        else:
            service_ratios[i] = 1.0

    sum_r = float(np.sum(service_ratios))
    sum_r_sq = float(np.sum(service_ratios**2))
    if sum_r_sq > numerical_tolerance:
        jains_index = (sum_r**2) / (n_evs * sum_r_sq)
    else:
        jains_index = 1.0

    return SimulationMetrics(
        deadline_satisfaction_pct=round(deadline_sat_pct, 2),
        total_unmet_energy_kwh=round(total_unmet, 3),
        electricity_cost=round(elec_cost, 3),
        max_capacity_violation_kw=round(max_cap_violation, 6),
        num_violating_slots=violating_slots,
        runtime_seconds=round(runtime_seconds, 4),
        jains_fairness_index=round(jains_index, 4),
        total_delivered_energy_kwh=round(total_del, 3),
        total_requested_energy_kwh=round(total_req, 3),
        objective_value=round(obj_val, 3),
    )
