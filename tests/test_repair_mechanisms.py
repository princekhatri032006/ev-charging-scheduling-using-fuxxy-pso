"""Unit tests for capacity repair mechanisms (Proportional & Fuzzy Urgency)."""

import pytest
import numpy as np
from src.models import EV
from src.fuzzy_controller import FuzzyUrgencyController
from src.repair import repair_slot_proportional, repair_slot_fuzzy_urgency


@pytest.fixture
def sample_evs():
    # Fleet of 4 EVs
    # EV1: arrives 0.0, departs 4.0, req 20.0, max 10.0
    # EV2: arrives 0.0, departs 1.0, req 10.0, max 10.0 (Tight deadline!)
    # EV3: arrives 2.0, departs 6.0, req 15.0, max 7.4 (Unavailable at t=0!)
    # EV4: arrives 0.0, departs 5.0, req 0.0, max 11.0 (Fully charged!)
    return [
        EV("EV_01", 0.0, 4.0, 20.0, 10.0),
        EV("EV_02", 0.0, 1.0, 10.0, 10.0),
        EV("EV_03", 2.0, 6.0, 15.0, 7.4),
        EV("EV_04", 0.0, 5.0, 0.0, 11.0),
    ]


@pytest.fixture
def fuzzy_ctrl():
    return FuzzyUrgencyController(use_lut=True)


def test_proportional_repair_station_capacity(sample_evs):
    # Total requested = 10 + 10 + 5 + 5 = 30 kW. Station capacity = 15 kW.
    cand = np.array([10.0, 10.0, 5.0, 5.0])
    rem_e = np.array([20.0, 10.0, 15.0, 0.0])

    rep = repair_slot_proportional(
        candidate_power=cand,
        evs=sample_evs,
        station_capacity_kw=15.0,
        remaining_energy_kwh=rem_e,
        slot_idx=0,  # t=0h
        slot_duration_hours=0.25,
    )

    # 1. Total power must not exceed 15 kW
    assert np.sum(rep) <= 15.0 + 1e-6
    # 2. EV3 is unavailable at t=0 -> 0 kW
    assert rep[2] == 0.0
    # 3. EV4 has 0 remaining demand -> 0 kW
    assert rep[3] == 0.0
    # 4. EV1 and EV2 requested 10 kW each, so total valid request = 20 kW
    # Scaled by 15 / 20 = 0.75 -> 7.5 kW each
    assert pytest.approx(rep[0], rel=1e-5) == 7.5
    assert pytest.approx(rep[1], rel=1e-5) == 7.5
    # 5. Nonnegative
    assert np.all(rep >= 0.0)


def test_fuzzy_repair_station_capacity(sample_evs, fuzzy_ctrl):
    # Capacity = 15 kW. EV1 and EV2 both request 10 kW.
    # But EV2 has tight deadline (departure 1.0h, req 10 kWh -> slack = 0.0h) -> very high urgency!
    # EV1 has departure 4.0h, req 20 kWh -> slack = 2.0h -> lower urgency.
    cand = np.array([10.0, 10.0, 5.0, 5.0])
    rem_e = np.array([20.0, 10.0, 15.0, 0.0])

    rep = repair_slot_fuzzy_urgency(
        candidate_power=cand,
        evs=sample_evs,
        station_capacity_kw=15.0,
        remaining_energy_kwh=rem_e,
        slot_idx=0,
        fuzzy_controller=fuzzy_ctrl,
        slot_duration_hours=0.25,
    )

    # Total power <= 15 kW
    assert np.sum(rep) <= 15.0 + 1e-6
    # EV3 unavailable -> 0 kW
    assert rep[2] == 0.0
    # EV4 zero remaining energy -> 0 kW
    assert rep[3] == 0.0
    # Individual limits respected
    assert rep[0] <= sample_evs[0].max_charging_power_kw
    assert rep[1] <= sample_evs[1].max_charging_power_kw
    # Urgency prioritization: EV2 has much tighter deadline than EV1, so rep[1] > rep[0]
    assert rep[1] > rep[0]
    # Nonnegative
    assert np.all(rep >= 0.0)


def test_zero_requested_power_safety(sample_evs, fuzzy_ctrl):
    cand_zeros = np.zeros(len(sample_evs))
    rem_e = np.array([20.0, 10.0, 15.0, 0.0])

    # Proportional repair on zero requests
    rep_prop = repair_slot_proportional(
        candidate_power=cand_zeros,
        evs=sample_evs,
        station_capacity_kw=20.0,
        remaining_energy_kwh=rem_e,
        slot_idx=0,
    )
    assert np.all(rep_prop == 0.0)

    # Fuzzy repair with zero requests: total request (0) <= capacity (20), so retains 0.0
    rep_fuzzy = repair_slot_fuzzy_urgency(
        candidate_power=cand_zeros,
        evs=sample_evs,
        station_capacity_kw=20.0,
        remaining_energy_kwh=rem_e,
        slot_idx=0,
        fuzzy_controller=fuzzy_ctrl,
    )
    assert np.all(rep_fuzzy == 0.0)


def test_zero_station_capacity(sample_evs, fuzzy_ctrl):
    cand = np.array([10.0, 10.0, 5.0, 5.0])
    rem_e = np.array([20.0, 10.0, 15.0, 0.0])

    rep_prop = repair_slot_proportional(
        candidate_power=cand,
        evs=sample_evs,
        station_capacity_kw=0.0,
        remaining_energy_kwh=rem_e,
        slot_idx=0,
    )
    assert np.sum(rep_prop) <= 1e-6

    rep_fuzzy = repair_slot_fuzzy_urgency(
        candidate_power=cand,
        evs=sample_evs,
        station_capacity_kw=0.0,
        remaining_energy_kwh=rem_e,
        slot_idx=0,
        fuzzy_controller=fuzzy_ctrl,
    )
    assert np.sum(rep_fuzzy) <= 1e-6


def test_iterative_capped_allocation_redistribution(fuzzy_ctrl):
    """Verifies that when one EV hits its physical cap, leftover capacity is redistributed to other eligible EVs."""
    # EV1 has max power 5.0 kW (cap = 5.0 kW)
    # EV2 has max power 20.0 kW (cap = 20.0 kW)
    # Both available, both need energy. Station capacity = 20.0 kW.
    # Total requested = 25 kW > 20 kW.
    ev1 = EV("EV1", 0.0, 4.0, 20.0, 5.0)
    ev2 = EV("EV2", 0.0, 4.0, 40.0, 20.0)
    evs = [ev1, ev2]

    cand = np.array([10.0, 20.0])
    rem_e = np.array([20.0, 40.0])

    rep = repair_slot_fuzzy_urgency(
        candidate_power=cand,
        evs=evs,
        station_capacity_kw=20.0,
        remaining_energy_kwh=rem_e,
        slot_idx=0,
        fuzzy_controller=fuzzy_ctrl,
    )

    # EV1 must not exceed 5.0 kW
    assert rep[0] <= 5.0 + 1e-6
    # Total allocated must utilize full 20 kW capacity since EV2 can absorb the rest (up to 20 kW)
    assert pytest.approx(np.sum(rep), rel=1e-4) == 20.0
    # EV2 receives the rest (>= 15.0 kW)
    assert rep[1] >= 15.0 - 1e-4
