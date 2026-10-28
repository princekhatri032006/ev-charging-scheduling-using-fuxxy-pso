"""Capacity repair mechanisms for EV charging schedules.

Provides:
- repair_proportional: Standard proportional power-scaling repair.
- repair_fuzzy_urgency: Priority-weighted iterative capped power repair using Mamdani fuzzy urgency.
"""

from typing import List, Optional
import numpy as np
from .models import EV, StationConfig
from .fuzzy_controller import FuzzyUrgencyController


def repair_slot_proportional(
    candidate_power: np.ndarray,  # 1D array of length N (kW)
    evs: List[EV],
    station_capacity_kw: float,
    remaining_energy_kwh: np.ndarray,  # 1D array of length N (kWh)
    slot_idx: int,
    slot_duration_hours: float = 0.25,
    numerical_tolerance: float = 1e-6,
) -> np.ndarray:
    """Repairs a single time slot using proportional power scaling.

    Steps:
    1. Set power of unavailable EVs to 0.
    2. Clip power to [0, max_charging_power].
    3. Remove power for EVs with no remaining energy demand, and cap at remaining energy / delta_t.
    4. If total requested power <= station capacity, retain requests.
    5. Else, scale all requests proportionally by (capacity / total_requested).
    6. Verify final capacity and individual limits.
    """
    n_evs = len(evs)
    repaired = np.zeros(n_evs, dtype=np.float64)

    for i, ev in enumerate(evs):
        # 1. Availability check
        if not ev.is_available_slot(slot_idx, slot_duration_hours, numerical_tolerance):
            continue

        # 2 & 3. Demand and individual limits
        rem_e = remaining_energy_kwh[i]
        if rem_e <= numerical_tolerance:
            continue

        max_slot_power = min(ev.max_charging_power_kw, rem_e / slot_duration_hours)
        req = float(candidate_power[i])
        repaired[i] = np.clip(req, 0.0, max_slot_power)

    # 4. Total power check
    total_requested = float(np.sum(repaired))
    if total_requested <= station_capacity_kw + numerical_tolerance:
        return repaired

    # 5. Proportional scaling
    if total_requested > 0.0:
        scale = station_capacity_kw / total_requested
        repaired = repaired * scale

    # 6. Numerical safety clip
    repaired = np.clip(repaired, 0.0, None)
    total_after = float(np.sum(repaired))
    if total_after > station_capacity_kw:
        repaired = repaired * (station_capacity_kw / total_after)

    return repaired


def repair_slot_fuzzy_urgency(
    candidate_power: np.ndarray,  # 1D array of length N (kW)
    evs: List[EV],
    station_capacity_kw: float,
    remaining_energy_kwh: np.ndarray,  # 1D array of length N (kWh)
    slot_idx: int,
    fuzzy_controller: FuzzyUrgencyController,
    slot_duration_hours: float = 0.25,
    epsilon: float = 0.01,
    delta: float = 0.1,
    numerical_tolerance: float = 1e-6,
) -> np.ndarray:
    """Repairs a single time slot using fuzzy urgency-weighted iterative capped allocation.

    Steps:
    1. Set unavailable EVs to 0.
    2. Clip requests to individual charging limits and remaining energy bounds.
    3. If total request <= capacity, retain valid requests.
    4. If total request > capacity:
       - Calculate fuzzy urgency for each available EV needing charge.
       - Compute weights: w_i = (urgency_i + epsilon) * (request_i + delta).
       - Allocate capacity via iterative capped redistribution until capacity is exhausted
         or all active EVs reach their maximum physical/energy capacity for this slot.
    """
    n_evs = len(evs)
    current_time_h = slot_idx * slot_duration_hours

    # Pre-filter and determine per-EV upper caps for this slot
    caps = np.zeros(n_evs, dtype=np.float64)
    clipped_requests = np.zeros(n_evs, dtype=np.float64)
    active_indices = []

    for i, ev in enumerate(evs):
        if not ev.is_available_slot(slot_idx, slot_duration_hours, numerical_tolerance):
            continue

        rem_e = remaining_energy_kwh[i]
        if rem_e <= numerical_tolerance:
            continue

        max_slot_p = min(ev.max_charging_power_kw, rem_e / slot_duration_hours)
        if max_slot_p > numerical_tolerance:
            caps[i] = max_slot_p
            clipped_requests[i] = np.clip(candidate_power[i], 0.0, max_slot_p)
            active_indices.append(i)

    # If no active EVs, return all zeros
    if not active_indices:
        return np.zeros(n_evs, dtype=np.float64)

    total_requested = float(np.sum(clipped_requests))
    if total_requested <= station_capacity_kw + numerical_tolerance:
        return clipped_requests

    # Calculate fuzzy urgencies and allocation weights
    weights = np.zeros(n_evs, dtype=np.float64)
    for i in active_indices:
        ev = evs[i]
        rem_e = remaining_energy_kwh[i]
        rem_t = ev.departure_time_hours - current_time_h
        slack = fuzzy_controller.compute_slack(rem_e, rem_t, ev.max_charging_power_kw)
        urgency = fuzzy_controller.evaluate(rem_e, slack)
        req = clipped_requests[i]
        weights[i] = (urgency + epsilon) * (req + delta)

    # Iterative capped allocation
    allocated = np.zeros(n_evs, dtype=np.float64)
    rem_capacity = float(station_capacity_kw)
    active_set = set(active_indices)

    while active_set and rem_capacity > numerical_tolerance:
        sum_active_weights = sum(weights[i] for i in active_set)
        if sum_active_weights <= 1e-12:
            # Fallback: distribute equally among active EVs
            equal_share = rem_capacity / len(active_set)
            tentative_add = {i: equal_share for i in active_set}
        else:
            tentative_add = {
                i: rem_capacity * (weights[i] / sum_active_weights) for i in active_set
            }

        # Check for EVs that would hit or exceed their remaining headroom
        newly_capped = []
        for i in active_set:
            headroom = caps[i] - allocated[i]
            if tentative_add[i] >= headroom - numerical_tolerance:
                newly_capped.append(i)

        if newly_capped:
            # Freeze newly capped EVs at their caps and deduct from remaining capacity
            for i in newly_capped:
                headroom = max(0.0, caps[i] - allocated[i])
                allocated[i] = caps[i]
                rem_capacity -= headroom
                active_set.remove(i)
            # Loop continues to redistribute leftover capacity among remaining active EVs
        else:
            # All active EVs can absorb their tentative share without exceeding cap
            for i in active_set:
                allocated[i] += tentative_add[i]
            rem_capacity = 0.0
            break

    # Final sanity bounds enforcement
    allocated = np.clip(allocated, 0.0, caps)
    total_allocated = float(np.sum(allocated))
    if total_allocated > station_capacity_kw:
        allocated = allocated * (station_capacity_kw / total_allocated)

    return allocated
