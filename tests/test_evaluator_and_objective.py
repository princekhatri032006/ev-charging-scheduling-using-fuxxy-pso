"""Unit tests for chronological schedule evaluator and objective calculation."""

import pytest
import numpy as np
from src.models import EV, StationConfig
from src.evaluator import simulate_and_repair_schedule, calculate_objective, evaluate_schedule_metrics
from src.fuzzy_controller import FuzzyUrgencyController


@pytest.fixture
def test_fleet_and_station():
    evs = [
        EV("EV_01", 0.0, 3.0, 10.0, 11.0),
        EV("EV_02", 1.0, 4.0, 15.0, 7.4),
        EV("EV_03", 2.0, 6.0, 20.0, 22.0),
    ]
    station = StationConfig(capacity_kw=25.0, horizon_hours=6.0, slot_duration_hours=0.25)
    return evs, station


def test_no_overcharging_and_non_negative_remaining(test_fleet_and_station):
    evs, station = test_fleet_and_station
    n_evs = len(evs)
    n_slots = station.num_slots

    # Extreme candidate schedule requesting max power every single slot
    candidate = np.full((n_evs, n_slots), 50.0)

    for method in ["proportional", "fuzzy_urgency"]:
        repaired, delivered, unmet = simulate_and_repair_schedule(
            candidate_schedule=candidate,
            evs=evs,
            station=station,
            repair_method=method,
        )

        # 1. Total delivered energy must never exceed required energy
        req_energy = np.array([ev.required_energy_kwh for ev in evs])
        for i in range(n_evs):
            assert delivered[i] <= req_energy[i] + 1e-6
            assert unmet[i] >= 0.0
            assert pytest.approx(delivered[i] + unmet[i], rel=1e-5) == req_energy[i]

        # 2. Charging must stop once required energy is reached
        # Verify that after an EV reaches full charge, subsequent slot power is 0
        cum_energy = np.cumsum(repaired * station.slot_duration_hours, axis=1)
        for i in range(n_evs):
            full_slots = np.where(cum_energy[i, :] >= req_energy[i] - 1e-6)[0]
            if len(full_slots) > 1:
                # All slots after the first full slot must have 0 power
                first_full = full_slots[0]
                if first_full + 1 < n_slots:
                    assert np.all(repaired[i, first_full+1:] <= 1e-6)

        # 3. Station capacity strictly respected in all slots
        slot_totals = np.sum(repaired, axis=0)
        assert np.all(slot_totals <= station.capacity_kw + 1e-6)


def test_objective_calculation_slot_duration():
    # 2 slots, flat price 0.20 currency/kWh, slot duration 0.25h
    prices = np.array([0.20, 0.20])
    station = StationConfig(capacity_kw=50.0, horizon_hours=0.5, slot_duration_hours=0.25, electricity_prices=prices)

    # Schedule: EV1 charges at 10 kW for slot 0 and 20 kW for slot 1
    # Total energy = 10 * 0.25 + 20 * 0.25 = 2.5 + 5.0 = 7.5 kWh
    # Expected cost = 7.5 kWh * 0.20 = 1.50 currency units
    repaired = np.array([[10.0, 20.0]])
    unmet = np.array([2.0])  # 2.0 kWh unmet
    lambda_unmet = 50.0

    total_obj, cost, unmet_sum = calculate_objective(
        repaired_schedule=repaired,
        unmet_energy=unmet,
        station=station,
        lambda_unmet=lambda_unmet,
    )

    assert pytest.approx(cost, rel=1e-5) == 1.50
    assert pytest.approx(unmet_sum, rel=1e-5) == 2.0
    assert pytest.approx(total_obj, rel=1e-5) == 1.50 + (50.0 * 2.0)  # 101.50


def test_identical_objective_conventions_for_both_methods(test_fleet_and_station):
    """Verifies that both repair methods feed into the exact same objective and metrics calculation."""
    evs, station = test_fleet_and_station
    n_evs, n_slots = len(evs), station.num_slots
    candidate = np.full((n_evs, n_slots), 10.0)

    rep_p, deliv_p, unmet_p = simulate_and_repair_schedule(candidate, evs, station, "proportional")
    rep_f, deliv_f, unmet_f = simulate_and_repair_schedule(candidate, evs, station, "fuzzy_urgency")

    metrics_p = evaluate_schedule_metrics(rep_p, evs, station, deliv_p, unmet_p)
    metrics_f = evaluate_schedule_metrics(rep_f, evs, station, deliv_f, unmet_f)

    # Both must report 0 capacity violations
    assert metrics_p.max_capacity_violation_kw == 0.0
    assert metrics_f.max_capacity_violation_kw == 0.0
    assert metrics_p.num_violating_slots == 0
    assert metrics_f.num_violating_slots == 0
