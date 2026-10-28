"""Unit tests for experimental reproducibility across seeds and runs."""

import pytest
import numpy as np
from src.models import EV, StationConfig
from src.pso import PSOOptimizer
from src.generator import generate_scenario


def test_seed_reproducibility():
    """Identical seed and parameters must produce bit-for-bit identical schedules and metrics."""
    sc_params = {
        "num_evs": 5,
        "station_capacity_kw": 30.0,
        "arrival_min_hours": 0.0,
        "arrival_max_hours": 4.0,
        "duration_min_hours": 2.0,
        "duration_max_hours": 5.0,
        "energy_min_kwh": 10.0,
        "energy_max_kwh": 25.0,
        "power_choices_kw": [6.6, 11.0],
    }

    evs1, station1 = generate_scenario("rep_test", sc_params, seed=777)
    evs2, station2 = generate_scenario("rep_test", sc_params, seed=777)

    # Generated EVs must match
    for e1, e2 in zip(evs1, evs2):
        assert e1.ev_id == e2.ev_id
        assert e1.arrival_time_hours == e2.arrival_time_hours
        assert e1.departure_time_hours == e2.departure_time_hours
        assert e1.required_energy_kwh == e2.required_energy_kwh
        assert e1.max_charging_power_kw == e2.max_charging_power_kw

    for method in ["proportional", "fuzzy_urgency"]:
        pso1 = PSOOptimizer(evs1, station1, repair_method=method, swarm_size=10, max_iterations=10, random_seed=777)
        pso2 = PSOOptimizer(evs2, station2, repair_method=method, swarm_size=10, max_iterations=10, random_seed=777)

        res1 = pso1.optimize()
        res2 = pso2.optimize()

        # Convergence history must match exactly
        np.testing.assert_allclose(res1.convergence_history, res2.convergence_history, rtol=1e-8)
        # Final schedule must match exactly
        np.testing.assert_allclose(res1.schedule, res2.schedule, rtol=1e-8)
        # Metrics must match exactly
        assert res1.metrics.objective_value == res2.metrics.objective_value
        assert res1.metrics.total_unmet_energy_kwh == res2.metrics.total_unmet_energy_kwh
        assert res1.metrics.electricity_cost == res2.metrics.electricity_cost
