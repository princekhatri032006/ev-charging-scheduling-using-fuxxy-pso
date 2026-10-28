"""Unit tests for the Mamdani Fuzzy Urgency Controller."""

import pytest
import numpy as np
from src.fuzzy_controller import FuzzyUrgencyController, trimf_scalar, trapmf_scalar


@pytest.fixture
def controller():
    return FuzzyUrgencyController(use_lut=True)


def test_membership_functions():
    # Triangular MF: trimf(x, 10, 20, 30)
    assert trimf_scalar(5.0, 10.0, 20.0, 30.0) == 0.0
    assert trimf_scalar(10.0, 10.0, 20.0, 30.0) == 0.0
    assert trimf_scalar(15.0, 10.0, 20.0, 30.0) == 0.5
    assert trimf_scalar(20.0, 10.0, 20.0, 30.0) == 1.0
    assert trimf_scalar(25.0, 10.0, 20.0, 30.0) == 0.5
    assert trimf_scalar(30.0, 10.0, 20.0, 30.0) == 0.0
    assert trimf_scalar(35.0, 10.0, 20.0, 30.0) == 0.0

    # Trapezoidal MF: trapmf(x, 0, 0, 8, 16)
    assert trapmf_scalar(0.0, 0.0, 0.0, 8.0, 16.0) == 1.0
    assert trapmf_scalar(5.0, 0.0, 0.0, 8.0, 16.0) == 1.0
    assert trapmf_scalar(8.0, 0.0, 0.0, 8.0, 16.0) == 1.0
    assert trapmf_scalar(12.0, 0.0, 0.0, 8.0, 16.0) == 0.5
    assert trapmf_scalar(16.0, 0.0, 0.0, 8.0, 16.0) == 0.0
    assert trapmf_scalar(20.0, 0.0, 0.0, 8.0, 16.0) == 0.0


def test_zero_remaining_demand_always_zero_urgency(controller):
    """If an EV has no remaining energy demand, urgency must be strictly 0.0."""
    assert controller.evaluate(remaining_energy_kwh=0.0, slack_hours=0.0) == 0.0
    assert controller.evaluate(remaining_energy_kwh=0.0, slack_hours=-2.0) == 0.0
    assert controller.evaluate(remaining_energy_kwh=-5.0, slack_hours=0.5) == 0.0
    assert controller.evaluate_exact(remaining_energy_kwh=0.0, slack_hours=0.0) == 0.0


def test_urgency_bounded_and_deterministic(controller):
    """Urgency output must be strictly in [0, 1] and 100% deterministic."""
    test_cases = [
        (5.0, 0.1),
        (20.0, 0.0),
        (35.0, -1.0),
        (8.0, 1.5),
        (25.0, 2.0),
        (10.0, 4.0),
        (40.0, 5.0),
    ]

    for d, s in test_cases:
        u1 = controller.evaluate(d, s)
        u2 = controller.evaluate(d, s)
        assert u1 == u2, f"Evaluation not deterministic for d={d}, s={s}"
        assert 0.0 <= u1 <= 1.0, f"Urgency {u1} out of bounds [0, 1]"


def test_rule_base_behavior(controller):
    """Verifies expected relative urgency levels for low/med/high demand and critical/comfortable slack."""
    # Critical slack + high demand -> Very High urgency (> 0.80)
    u_high_crit = controller.evaluate(remaining_energy_kwh=35.0, slack_hours=-0.5)
    assert u_high_crit > 0.80

    # Critical slack + medium demand -> Very High urgency (> 0.80)
    u_med_crit = controller.evaluate(remaining_energy_kwh=20.0, slack_hours=0.0)
    assert u_med_crit > 0.80

    # Critical slack + low demand -> High urgency (~0.65 - 0.80)
    u_low_crit = controller.evaluate(remaining_energy_kwh=5.0, slack_hours=0.0)
    assert 0.60 <= u_low_crit <= 0.85

    # Comfortable slack + low demand -> Low urgency (< 0.30)
    u_low_comf = controller.evaluate(remaining_energy_kwh=5.0, slack_hours=5.0)
    assert u_low_comf < 0.30

    # Comfortable slack + high demand -> Medium urgency (~0.35 - 0.60)
    u_high_comf = controller.evaluate(remaining_energy_kwh=35.0, slack_hours=5.0)
    assert 0.30 <= u_high_comf <= 0.65

    # Monotonicity check: For same demand, critical slack MUST have higher urgency than comfortable slack
    assert u_high_crit > u_high_comf
    assert u_med_crit > controller.evaluate(remaining_energy_kwh=20.0, slack_hours=5.0)
    assert u_low_crit > u_low_comf


def test_lut_accuracy_against_exact(controller):
    """Verifies that the accelerated LUT closely approximates exact Mamdani centroid calculation."""
    for d in [4.0, 12.0, 22.0, 33.0, 45.0]:
        for s in [-1.0, 0.2, 1.2, 2.5, 4.0]:
            exact = controller.evaluate_exact(d, s)
            lut_val = controller.evaluate(d, s)
            assert abs(exact - lut_val) < 0.02, f"LUT mismatch at d={d}, s={s}: {lut_val} vs exact {exact}"


def test_compute_slack(controller):
    # EV with 10 kWh needed, 10 kW max power -> 1.0 h needed. Remaining stay: 2.5 h -> slack = 1.5 h
    slack = controller.compute_slack(remaining_energy_kwh=10.0, remaining_time_hours=2.5, max_charging_power_kw=10.0)
    assert pytest.approx(slack, rel=1e-5) == 1.5

    # EV with 20 kWh needed, 10 kW max power -> 2.0 h needed. Remaining stay: 1.0 h -> slack = -1.0 h
    slack_neg = controller.compute_slack(remaining_energy_kwh=20.0, remaining_time_hours=1.0, max_charging_power_kw=10.0)
    assert pytest.approx(slack_neg, rel=1e-5) == -1.0

    # Fully charged EV -> slack = remaining_time
    slack_zero = controller.compute_slack(remaining_energy_kwh=0.0, remaining_time_hours=3.0, max_charging_power_kw=10.0)
    assert slack_zero == 3.0
