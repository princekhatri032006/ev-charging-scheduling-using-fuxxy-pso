"""Continuous Particle Swarm Optimization (PSO) for EV Charging Scheduling."""

import time
from typing import Callable, List, Literal, Optional, Tuple
import numpy as np
from .models import EV, StationConfig, ScheduleResult
from .fuzzy_controller import FuzzyUrgencyController
from .evaluator import simulate_and_repair_schedule, calculate_objective, evaluate_schedule_metrics


class PSOOptimizer:
    """Continuous Particle Swarm Optimizer for EV Charging Scheduling."""

    def __init__(
        self,
        evs: List[EV],
        station: StationConfig,
        repair_method: Literal["proportional", "fuzzy_urgency"] = "proportional",
        swarm_size: int = 30,
        max_iterations: int = 50,
        inertia_weight: float = 0.7,
        cognitive_coefficient: float = 1.5,
        social_coefficient: float = 1.5,
        velocity_clamp: float = 5.0,
        lambda_unmet: float = 50.0,
        random_seed: int = 42,
        fuzzy_controller: Optional[FuzzyUrgencyController] = None,
        epsilon: float = 0.01,
        delta: float = 0.1,
    ):
        self.evs = evs
        self.station = station
        self.repair_method = repair_method
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w = inertia_weight
        self.c1 = cognitive_coefficient
        self.c2 = social_coefficient
        self.v_clamp = velocity_clamp
        self.lambda_unmet = lambda_unmet
        self.random_seed = random_seed
        self.epsilon = epsilon
        self.delta = delta

        self.n_evs = len(evs)
        self.n_slots = station.num_slots
        self.dim = self.n_evs * self.n_slots

        if self.repair_method == "fuzzy_urgency":
            self.fuzzy_controller = fuzzy_controller or FuzzyUrgencyController()
        else:
            self.fuzzy_controller = None

        # Precompute individual max power and availability masks
        self.max_power_matrix = np.zeros((self.n_evs, self.n_slots), dtype=np.float64)
        self.availability_mask = np.zeros((self.n_evs, self.n_slots), dtype=bool)

        for i, ev in enumerate(self.evs):
            self.max_power_matrix[i, :] = ev.max_charging_power_kw
            for t in range(self.n_slots):
                self.availability_mask[i, t] = ev.is_available_slot(
                    t, station.slot_duration_hours
                )

        self.max_power_flat = self.max_power_matrix.flatten()
        self.availability_flat = self.availability_mask.flatten()

    def _evaluate_particle(self, position_flat: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluates a single particle's fitness after repair.

        Returns:
            fitness, repaired_matrix, delivered_energy, unmet_energy
        """
        # Reshape to (N, T)
        cand_matrix = position_flat.reshape((self.n_evs, self.n_slots))

        # Enforce basic availability and nonnegative power bounds
        cand_matrix = np.where(self.availability_mask, cand_matrix, 0.0)
        cand_matrix = np.clip(cand_matrix, 0.0, self.max_power_matrix)

        # Chronological repair & simulation
        repaired, delivered, unmet = simulate_and_repair_schedule(
            candidate_schedule=cand_matrix,
            evs=self.evs,
            station=self.station,
            repair_method=self.repair_method,
            fuzzy_controller=self.fuzzy_controller,
            epsilon=self.epsilon,
            delta=self.delta,
        )

        fitness, _, _ = calculate_objective(
            repaired_schedule=repaired,
            unmet_energy=unmet,
            station=self.station,
            lambda_unmet=self.lambda_unmet,
        )

        return fitness, repaired, delivered, unmet

    def optimize(
        self,
        initial_positions: Optional[np.ndarray] = None,
        scenario_name: str = "default",
    ) -> ScheduleResult:
        """Executes the PSO optimization loop.

        Args:
            initial_positions: Optional (swarm_size, dim) initial particle positions for paired runs.
            scenario_name: Name of the scenario for result tagging.

        Returns:
            ScheduleResult containing the best schedule, metrics, and convergence history.
        """
        start_time = time.perf_counter()
        rng = np.random.default_rng(self.random_seed)

        # Initialize particle positions and velocities
        if initial_positions is not None:
            if initial_positions.shape != (self.swarm_size, self.dim):
                raise ValueError(
                    f"initial_positions shape {initial_positions.shape} != ({self.swarm_size}, {self.dim})"
                )
            positions = initial_positions.copy()
        else:
            # Sample uniformly within [0, max_power] where available
            positions = np.zeros((self.swarm_size, self.dim), dtype=np.float64)
            for p in range(self.swarm_size):
                rand_pos = rng.uniform(0.0, self.max_power_flat)
                positions[p] = np.where(self.availability_flat, rand_pos, 0.0)

        velocities = rng.uniform(-self.v_clamp * 0.5, self.v_clamp * 0.5, size=(self.swarm_size, self.dim))

        # Initialize personal bests
        pbest_positions = positions.copy()
        pbest_fitnesses = np.full(self.swarm_size, np.inf, dtype=np.float64)

        gbest_fitness = np.inf
        gbest_position = np.zeros(self.dim, dtype=np.float64)
        best_repaired_schedule = np.zeros((self.n_evs, self.n_slots), dtype=np.float64)
        best_delivered = np.zeros(self.n_evs, dtype=np.float64)
        best_unmet = np.zeros(self.n_evs, dtype=np.float64)

        # Initial evaluation
        for p in range(self.swarm_size):
            fit, rep, deliv, unmet = self._evaluate_particle(positions[p])
            pbest_fitnesses[p] = fit
            if fit < gbest_fitness:
                gbest_fitness = fit
                gbest_position = positions[p].copy()
                best_repaired_schedule = rep.copy()
                best_delivered = deliv.copy()
                best_unmet = unmet.copy()

        convergence_history = [float(gbest_fitness)]

        # PSO main iteration loop
        for it in range(self.max_iterations):
            r1 = rng.uniform(0.0, 1.0, size=(self.swarm_size, self.dim))
            r2 = rng.uniform(0.0, 1.0, size=(self.swarm_size, self.dim))

            # Velocity update
            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest_positions - positions)
                + self.c2 * r2 * (gbest_position - positions)
            )

            # Velocity clamping
            velocities = np.clip(velocities, -self.v_clamp, self.v_clamp)

            # Position update
            positions = positions + velocities

            # Bound enforcement on positions
            positions = np.clip(positions, 0.0, self.max_power_flat)
            positions = np.where(self.availability_flat, positions, 0.0)

            # Fitness evaluation
            for p in range(self.swarm_size):
                fit, rep, deliv, unmet = self._evaluate_particle(positions[p])

                # Update personal best
                if fit < pbest_fitnesses[p]:
                    pbest_fitnesses[p] = fit
                    pbest_positions[p] = positions[p].copy()

                    # Update global best
                    if fit < gbest_fitness:
                        gbest_fitness = fit
                        gbest_position = positions[p].copy()
                        best_repaired_schedule = rep.copy()
                        best_delivered = deliv.copy()
                        best_unmet = unmet.copy()

            convergence_history.append(float(gbest_fitness))

        elapsed_time = time.perf_counter() - start_time

        # Compile final metrics
        metrics = evaluate_schedule_metrics(
            repaired_schedule=best_repaired_schedule,
            evs=self.evs,
            station=self.station,
            delivered_energy=best_delivered,
            unmet_energy=best_unmet,
            runtime_seconds=elapsed_time,
            lambda_unmet=self.lambda_unmet,
        )

        return ScheduleResult(
            algorithm_name=f"PSO_{self.repair_method}",
            scenario_name=scenario_name,
            seed=self.random_seed,
            schedule=best_repaired_schedule,
            metrics=metrics,
            convergence_history=convergence_history,
        )
