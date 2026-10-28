"""Data structures and configuration models for EV charging simulation."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class EV:
    """Represents an Electric Vehicle arriving at the charging station.

    Attributes:
        ev_id: Unique identifier for the EV (e.g., 'EV_01').
        arrival_time_hours: Arrival timestamp in hours [0, horizon).
        departure_time_hours: Departure deadline in hours (arrival, horizon].
        required_energy_kwh: Energy demanded by the EV in kWh (> 0).
        max_charging_power_kw: Maximum physical onboard charging power in kW.
    """
    ev_id: str
    arrival_time_hours: float
    departure_time_hours: float
    required_energy_kwh: float
    max_charging_power_kw: float

    def __post_init__(self):
        if self.departure_time_hours <= self.arrival_time_hours:
            raise ValueError(
                f"EV {self.ev_id}: departure ({self.departure_time_hours}h) "
                f"must be after arrival ({self.arrival_time_hours}h)."
            )
        if self.required_energy_kwh < 0:
            raise ValueError(f"EV {self.ev_id}: required energy must be non-negative.")
        if self.max_charging_power_kw <= 0:
            raise ValueError(f"EV {self.ev_id}: max charging power must be positive.")

    def is_available_slot(
        self,
        slot_idx: int,
        slot_duration_hours: float = 0.25,
        numerical_tolerance: float = 1e-6,
    ) -> bool:
        """Determines if the EV is connected and available to charge in slot `slot_idx`.

        Convention:
            Slot t represents the time interval [t * delta_t, (t + 1) * delta_t].
            The EV is available during slot t if:
            - The slot start is at or after EV arrival: t * delta_t >= arrival - tol
            - The slot end is at or before EV departure: (t + 1) * delta_t <= departure + tol
        """
        slot_start = slot_idx * slot_duration_hours
        slot_end = (slot_idx + 1) * slot_duration_hours

        return (
            slot_start >= self.arrival_time_hours - numerical_tolerance
            and slot_end <= self.departure_time_hours + numerical_tolerance
        )

    def min_charging_time_hours(self) -> float:
        """Theoretical minimum time needed to charge at maximum power."""
        return self.required_energy_kwh / self.max_charging_power_kw

    def dwell_time_hours(self) -> float:
        """Total parking duration in hours."""
        return self.departure_time_hours - self.arrival_time_hours

    def is_theoretically_feasible(self) -> bool:
        """Returns True if dwell time >= minimum charging time."""
        return self.dwell_time_hours() >= self.min_charging_time_hours()


@dataclass
class StationConfig:
    """Represents the charging station configuration and time horizon.

    Attributes:
        capacity_kw: Maximum grid capacity in kW (constant or per-slot).
        horizon_hours: Total simulation duration in hours (e.g., 12.0).
        slot_duration_hours: Duration of each discrete time slot in hours (e.g., 0.25).
        electricity_prices: 1D array of electricity prices (currency/kWh) per slot.
    """
    capacity_kw: float
    horizon_hours: float = 12.0
    slot_duration_hours: float = 0.25
    electricity_prices: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.capacity_kw <= 0:
            raise ValueError("Station capacity must be positive.")
        if self.horizon_hours <= 0 or self.slot_duration_hours <= 0:
            raise ValueError("Horizon and slot duration must be positive.")

        num_slots = self.num_slots
        if self.electricity_prices is None:
            # Default flat price 0.20 if not specified
            self.electricity_prices = np.full(num_slots, 0.20, dtype=np.float64)
        else:
            self.electricity_prices = np.asarray(self.electricity_prices, dtype=np.float64)
            if len(self.electricity_prices) != num_slots:
                raise ValueError(
                    f"Electricity price array length ({len(self.electricity_prices)}) "
                    f"does not match expected number of slots ({num_slots})."
                )

    @property
    def num_slots(self) -> int:
        """Number of time slots in the horizon."""
        return int(round(self.horizon_hours / self.slot_duration_hours))

    def get_capacity(self, slot_idx: int) -> float:
        """Returns station capacity in kW for the given slot."""
        return self.capacity_kw


@dataclass
class SimulationMetrics:
    """Metrics evaluated for a complete repaired charging schedule."""
    deadline_satisfaction_pct: float
    total_unmet_energy_kwh: float
    electricity_cost: float
    max_capacity_violation_kw: float
    num_violating_slots: int
    runtime_seconds: float
    jains_fairness_index: float
    total_delivered_energy_kwh: float
    total_requested_energy_kwh: float
    objective_value: float


@dataclass
class ScheduleResult:
    """Complete output of a scheduling optimization run."""
    algorithm_name: str
    scenario_name: str
    seed: int
    schedule: np.ndarray  # Shape: (N, T) in kW
    metrics: SimulationMetrics
    convergence_history: List[float] = field(default_factory=list)
