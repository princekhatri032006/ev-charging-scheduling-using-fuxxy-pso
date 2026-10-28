"""Synthetic EV instance and scenario generator."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from .models import EV, StationConfig


def generate_tou_pricing(
    horizon_hours: float = 12.0,
    slot_duration_hours: float = 0.25,
    pricing_config: Dict = None,
) -> np.ndarray:
    """Generates a Time-of-Use (TOU) electricity pricing profile across time slots.

    Default tiers:
    - 00:00 - 04:00 (Off-Peak): 0.12 currency/kWh
    - 04:00 - 07:00 (Shoulder): 0.22 currency/kWh
    - 07:00 - 10:00 (Peak):     0.40 currency/kWh
    - 10:00 - 12:00 (Shoulder): 0.25 currency/kWh
    """
    if pricing_config is None:
        pricing_config = {
            "off_peak_price": 0.12,
            "shoulder_price": 0.22,
            "peak_price": 0.40,
            "off_peak_end_hour": 4.0,
            "shoulder_1_end_hour": 7.0,
            "peak_end_hour": 10.0,
        }

    num_slots = int(round(horizon_hours / slot_duration_hours))
    prices = np.zeros(num_slots, dtype=np.float64)

    for t in range(num_slots):
        time_h = t * slot_duration_hours
        if time_h < pricing_config["off_peak_end_hour"]:
            prices[t] = pricing_config["off_peak_price"]
        elif time_h < pricing_config["shoulder_1_end_hour"]:
            prices[t] = pricing_config["shoulder_price"]
        elif time_h < pricing_config["peak_end_hour"]:
            prices[t] = pricing_config["peak_price"]
        else:
            prices[t] = pricing_config.get("shoulder_2_price", 0.25)

    return prices


def generate_ev_fleet(
    num_evs: int,
    horizon_hours: float,
    arrival_range: Tuple[float, float],
    duration_range: Tuple[float, float],
    energy_range: Tuple[float, float],
    power_choices: List[float],
    seed: int,
) -> List[EV]:
    """Generates a synthetic fleet of EVs using a reproducible random seed."""
    rng = np.random.default_rng(seed)
    evs: List[EV] = []

    for i in range(num_evs):
        ev_id = f"EV_{i+1:02d}"

        # Sample arrival time
        arrival = float(rng.uniform(arrival_range[0], arrival_range[1]))
        arrival = round(arrival * 4) / 4  # Round to 15-minute grid

        # Sample dwell duration
        duration = float(rng.uniform(duration_range[0], duration_range[1]))
        duration = max(0.5, round(duration * 4) / 4)

        departure = min(horizon_hours, arrival + duration)
        # Ensure departure is strictly greater than arrival
        if departure <= arrival:
            departure = min(horizon_hours, arrival + 1.0)
            if departure <= arrival:
                arrival = max(0.0, departure - 1.0)

        # Sample required energy
        req_energy = float(rng.uniform(energy_range[0], energy_range[1]))
        req_energy = round(req_energy, 1)

        # Sample max charging power
        max_power = float(rng.choice(power_choices))

        ev = EV(
            ev_id=ev_id,
            arrival_time_hours=arrival,
            departure_time_hours=departure,
            required_energy_kwh=req_energy,
            max_charging_power_kw=max_power,
        )
        evs.append(ev)

    return evs


def generate_scenario(
    scenario_name: str,
    scenario_params: Dict,
    pricing_config: Dict = None,
    seed: int = 42,
) -> Tuple[List[EV], StationConfig]:
    """Generates a complete paired scenario (fleet and station config)."""
    horizon = float(scenario_params.get("horizon_hours", 12.0))
    slot_duration = float(scenario_params.get("slot_duration_hours", 0.25))
    capacity_kw = float(scenario_params["station_capacity_kw"])
    num_evs = int(scenario_params["num_evs"])

    prices = generate_tou_pricing(
        horizon_hours=horizon,
        slot_duration_hours=slot_duration,
        pricing_config=pricing_config,
    )

    station = StationConfig(
        capacity_kw=capacity_kw,
        horizon_hours=horizon,
        slot_duration_hours=slot_duration,
        electricity_prices=prices,
    )

    arrival_range = (
        float(scenario_params["arrival_min_hours"]),
        float(scenario_params["arrival_max_hours"]),
    )
    duration_range = (
        float(scenario_params["duration_min_hours"]),
        float(scenario_params["duration_max_hours"]),
    )
    energy_range = (
        float(scenario_params["energy_min_kwh"]),
        float(scenario_params["energy_max_kwh"]),
    )
    power_choices = [float(p) for p in scenario_params["power_choices_kw"]]

    evs = generate_ev_fleet(
        num_evs=num_evs,
        horizon_hours=horizon,
        arrival_range=arrival_range,
        duration_range=duration_range,
        energy_range=energy_range,
        power_choices=power_choices,
        seed=seed,
    )

    return evs, station


def characterize_scenario(
    evs: List[EV],
    station: StationConfig,
) -> Dict[str, float]:
    """Computes descriptive characteristics of a scenario instance.

    Metrics include total energy requested, available station energy,
    peak concurrent connected EVs, and theoretically feasible EV count.
    """
    total_req_energy = sum(ev.required_energy_kwh for ev in evs)
    total_station_energy = station.capacity_kw * station.horizon_hours
    num_feasible = sum(1 for ev in evs if ev.is_theoretically_feasible())

    # Calculate concurrency per slot
    num_slots = station.num_slots
    concurrent_counts = np.zeros(num_slots, dtype=int)
    for t in range(num_slots):
        for ev in evs:
            if ev.is_available_slot(t, station.slot_duration_hours):
                concurrent_counts[t] += 1

    peak_concurrency = int(np.max(concurrent_counts))
    mean_concurrency = float(np.mean(concurrent_counts))

    return {
        "num_evs": len(evs),
        "station_capacity_kw": station.capacity_kw,
        "total_requested_energy_kwh": round(total_req_energy, 2),
        "total_station_energy_kwh": round(total_station_energy, 2),
        "energy_demand_ratio": round(total_req_energy / total_station_energy, 3),
        "theoretically_feasible_evs": num_feasible,
        "feasible_evs_pct": round(num_feasible / len(evs) * 100, 1),
        "peak_concurrent_evs": peak_concurrency,
        "mean_concurrent_evs": round(mean_concurrency, 2),
    }


def save_ev_instances_to_csv(evs: List[EV], filepath: str) -> None:
    """Saves generated EV fleet instances to a transparent CSV file."""
    records = []
    for ev in evs:
        records.append({
            "ev_id": ev.ev_id,
            "arrival_time_hours": ev.arrival_time_hours,
            "departure_time_hours": ev.departure_time_hours,
            "dwell_time_hours": round(ev.dwell_time_hours(), 2),
            "required_energy_kwh": ev.required_energy_kwh,
            "max_charging_power_kw": ev.max_charging_power_kw,
            "min_charging_time_hours": round(ev.min_charging_time_hours(), 2),
            "is_theoretically_feasible": ev.is_theoretically_feasible(),
        })
    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
