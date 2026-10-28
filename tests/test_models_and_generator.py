"""Tests for EV, StationConfig, TOU pricing, and scenario generation."""

import pytest
import numpy as np
from src.models import EV, StationConfig
from src.generator import generate_tou_pricing, generate_ev_fleet, generate_scenario, characterize_scenario


def test_ev_initialization_and_validation():
    ev = EV(
        ev_id="EV_01",
        arrival_time_hours=1.0,
        departure_time_hours=4.0,
        required_energy_kwh=20.0,
        max_charging_power_kw=11.0,
    )
    assert ev.ev_id == "EV_01"
    assert ev.dwell_time_hours() == 3.0
    assert pytest.approx(ev.min_charging_time_hours(), rel=1e-3) == 20.0 / 11.0
    assert ev.is_theoretically_feasible() is True

    # Infeasible dwelling time
    with pytest.raises(ValueError, match="departure .* must be after arrival"):
        EV("EV_bad", 5.0, 3.0, 20.0, 11.0)

    # Negative energy
    with pytest.raises(ValueError, match="required energy must be non-negative"):
        EV("EV_bad", 1.0, 4.0, -5.0, 11.0)

    # Non-positive power
    with pytest.raises(ValueError, match="max charging power must be positive"):
        EV("EV_bad", 1.0, 4.0, 20.0, 0.0)


def test_ev_availability_convention():
    # EV arrives at 1.0h (slot 4) and departs at 3.0h (slot 12)
    ev = EV("EV_01", 1.0, 3.0, 15.0, 7.4)

    # Slots: slot 0 is [0.0, 0.25) -> not available
    assert not ev.is_available_slot(0, slot_duration_hours=0.25)
    assert not ev.is_available_slot(3, slot_duration_hours=0.25)  # [0.75, 1.0)

    # Slot 4 is [1.0, 1.25) -> available
    assert ev.is_available_slot(4, slot_duration_hours=0.25)
    # Slot 11 is [2.75, 3.0) -> available
    assert ev.is_available_slot(11, slot_duration_hours=0.25)
    # Slot 12 is [3.0, 3.25) -> not available
    assert not ev.is_available_slot(12, slot_duration_hours=0.25)


def test_station_config_and_pricing():
    station = StationConfig(capacity_kw=50.0, horizon_hours=12.0, slot_duration_hours=0.25)
    assert station.num_slots == 48
    assert station.get_capacity(0) == 50.0
    assert len(station.electricity_prices) == 48
    assert np.all(station.electricity_prices >= 0.0)

    # Bad configuration
    with pytest.raises(ValueError):
        StationConfig(capacity_kw=0.0)


def test_scenario_generation_and_characterization():
    sc_params = {
        "num_evs": 20,
        "station_capacity_kw": 50.0,
        "arrival_min_hours": 1.0,
        "arrival_max_hours": 5.0,
        "duration_min_hours": 3.0,
        "duration_max_hours": 6.0,
        "energy_min_kwh": 15.0,
        "energy_max_kwh": 35.0,
        "power_choices_kw": [7.4, 11.0, 22.0],
    }
    evs, station = generate_scenario("test_sc", sc_params, seed=123)
    assert len(evs) == 20
    assert station.capacity_kw == 50.0

    chars = characterize_scenario(evs, station)
    assert chars["num_evs"] == 20
    assert chars["station_capacity_kw"] == 50.0
    assert chars["total_requested_energy_kwh"] > 0
    assert chars["total_station_energy_kwh"] == 50.0 * 12.0
    assert 0 <= chars["feasible_evs_pct"] <= 100
