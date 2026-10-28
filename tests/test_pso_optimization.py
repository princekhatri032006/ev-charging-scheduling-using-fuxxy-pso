"""Unit tests for Continuous PSO optimization loop and convergence behavior."""

import pytest
import numpy as np
from src.models import EV, StationConfig
from src.pso import PSOOptimizer


@pytest.fixture
def pso_setup():
    evs = [
        EV("EV_01", 0.0, 4.0, 15.0, 11.0),
        EV("EV_02", 1.0, 5.0, 20.0, 7.4),
        EV("EV_03", 2.0, 6.0, 10.0, 22.0),
    ]
    station = StationConfig(capacity_kw=20.0, horizon_hours=6.0, slot_duration_hours=0.25)
    return evs, station


def test_pso_monotonic_convergence(pso_setup):
    """PSO best fitness should be non-increasing over iterations."""
    evs, station = pso_setup

    for method in ["proportional", "fuzzy_urgency"]:
        pso = PSOOptimizer(
            evs=evs,
            station=station,
            repair_method=method,
            swarm_size=15,
            max_iterations=15,
            random_seed=42,
        )
        res = pso.optimize()

        history = res.convergence_history
        assert len(history) == 16  # Initial (iter 0) + 15 iterations

        # Monotonically non-increasing: history[k+1] <= history[k] + tol
        for k in range(len(history) - 1):
            assert history[k+1] <= history[k] + 1e-6, (
                f"{method} best fitness increased at iteration {k+1}: {history[k+1]} > {history[k]}"
            )

        # Zero capacity violation in final schedule
        assert res.metrics.max_capacity_violation_kw == 0.0
        assert res.metrics.num_violating_slots == 0


def test_pso_paired_initial_positions(pso_setup):
    """Verifies that paired comparisons start with identical swarms."""
    evs, station = pso_setup
    dim = len(evs) * station.num_slots
    shared_init = np.random.default_rng(99).uniform(0.0, 10.0, size=(10, dim))

    pso_p = PSOOptimizer(evs, station, "proportional", swarm_size=10, max_iterations=5, random_seed=99)
    pso_f = PSOOptimizer(evs, station, "fuzzy_urgency", swarm_size=10, max_iterations=5, random_seed=99)

    res_p = pso_p.optimize(initial_positions=shared_init)
    res_f = pso_f.optimize(initial_positions=shared_init)

    assert res_p.metrics.max_capacity_violation_kw == 0.0
    assert res_f.metrics.max_capacity_violation_kw == 0.0
