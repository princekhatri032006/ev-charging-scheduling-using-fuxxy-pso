"""Experiment runner for paired comparisons across scenarios and seeds."""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml
from .models import ScheduleResult, StationConfig, EV
from .generator import generate_scenario, characterize_scenario, save_ev_instances_to_csv
from .fuzzy_controller import FuzzyUrgencyController
from .pso import PSOOptimizer


def run_paired_seed_comparison(
    scenario_name: str,
    scenario_params: Dict,
    pso_params: Dict,
    pricing_config: Dict,
    seed: int,
    fuzzy_controller: FuzzyUrgencyController,
    fuzzy_repair_params: Dict,
) -> Tuple[ScheduleResult, ScheduleResult]:
    """Runs a paired comparison between proportional and fuzzy urgency PSO on the same EV instance."""
    # Generate shared EV instance and station config
    evs, station = generate_scenario(
        scenario_name=scenario_name,
        scenario_params=scenario_params,
        pricing_config=pricing_config,
        seed=seed,
    )

    # Generate identical initial particle positions for both algorithms
    dim = len(evs) * station.num_slots
    swarm_size = pso_params.get("swarm_size", 30)
    init_rng = np.random.default_rng(seed)

    # Precompute availability and bounds for initial position sampling
    max_power_flat = np.repeat([ev.max_charging_power_kw for ev in evs], station.num_slots)
    avail_mask = np.zeros((len(evs), station.num_slots), dtype=bool)
    for i, ev in enumerate(evs):
        for t in range(station.num_slots):
            avail_mask[i, t] = ev.is_available_slot(t, station.slot_duration_hours)
    avail_flat = avail_mask.flatten()

    shared_initial_positions = np.zeros((swarm_size, dim), dtype=np.float64)
    for p in range(swarm_size):
        rand_pos = init_rng.uniform(0.0, max_power_flat)
        shared_initial_positions[p] = np.where(avail_flat, rand_pos, 0.0)

    # Method A: PSO with Proportional Repair
    pso_prop = PSOOptimizer(
        evs=evs,
        station=station,
        repair_method="proportional",
        swarm_size=swarm_size,
        max_iterations=pso_params.get("max_iterations", 50),
        inertia_weight=pso_params.get("inertia_weight", 0.7),
        cognitive_coefficient=pso_params.get("cognitive_coefficient", 1.5),
        social_coefficient=pso_params.get("social_coefficient", 1.5),
        velocity_clamp=pso_params.get("velocity_clamp", 5.0),
        lambda_unmet=scenario_params.get("lambda_unmet", 50.0),
        random_seed=seed,
    )
    result_prop = pso_prop.optimize(
        initial_positions=shared_initial_positions,
        scenario_name=scenario_name,
    )

    # Method B: PSO with Fuzzy Urgency-Guided Repair
    pso_fuzzy = PSOOptimizer(
        evs=evs,
        station=station,
        repair_method="fuzzy_urgency",
        swarm_size=swarm_size,
        max_iterations=pso_params.get("max_iterations", 50),
        inertia_weight=pso_params.get("inertia_weight", 0.7),
        cognitive_coefficient=pso_params.get("cognitive_coefficient", 1.5),
        social_coefficient=pso_params.get("social_coefficient", 1.5),
        velocity_clamp=pso_params.get("velocity_clamp", 5.0),
        lambda_unmet=scenario_params.get("lambda_unmet", 50.0),
        random_seed=seed,
        fuzzy_controller=fuzzy_controller,
        epsilon=fuzzy_repair_params.get("epsilon", 0.01),
        delta=fuzzy_repair_params.get("delta", 0.1),
    )
    result_fuzzy = pso_fuzzy.optimize(
        initial_positions=shared_initial_positions,
        scenario_name=scenario_name,
    )

    return result_prop, result_fuzzy


def run_full_experiments(
    config_path: str,
    output_dir: str = "results",
    num_seeds: int = 10,
    seed_start: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs reproducible paired experiments across all scenarios and seeds.

    Saves:
        - results/raw/run_metrics.csv
        - results/summaries/scenario_summary.csv
        - results/raw/convergence_histories.csv
        - results/raw/representative_schedules.csv
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "summaries"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    fuzzy_controller = FuzzyUrgencyController()
    scenarios = cfg["scenarios"]
    pso_params = cfg["pso"]
    pricing_config = cfg.get("pricing", {})
    fuzzy_repair_params = cfg.get("fuzzy_repair", {})
    global_lambda = cfg["simulation"]["lambda_unmet"]

    raw_records = []
    conv_records = []
    representative_schedules = []

    seeds = [seed_start + i for i in range(num_seeds)]
    print(f"=== Starting Experiments: {len(scenarios)} scenarios, {num_seeds} seeds/scenario ({len(scenarios)*num_seeds*2} total PSO runs) ===")

    for sc_name, sc_params in scenarios.items():
        sc_params_merged = sc_params.copy()
        if "lambda_unmet" not in sc_params_merged:
            sc_params_merged["lambda_unmet"] = global_lambda
        if "horizon_hours" not in sc_params_merged:
            sc_params_merged["horizon_hours"] = cfg["simulation"]["horizon_hours"]
        if "slot_duration_hours" not in sc_params_merged:
            sc_params_merged["slot_duration_hours"] = cfg["simulation"]["slot_duration_hours"]

        print(f"\n--- Running Scenario: {sc_name} ({sc_params.get('description', '')}) ---")

        # Save representative EV instance
        rep_evs, rep_station = generate_scenario(
            scenario_name=sc_name,
            scenario_params=sc_params_merged,
            pricing_config=pricing_config,
            seed=seeds[0],
        )
        save_ev_instances_to_csv(
            rep_evs,
            os.path.join(output_dir, "raw", f"ev_instances_{sc_name}_seed_{seeds[0]}.csv")
        )

        for s_idx, seed in enumerate(seeds):
            res_prop, res_fuzzy = run_paired_seed_comparison(
                scenario_name=sc_name,
                scenario_params=sc_params_merged,
                pso_params=pso_params,
                pricing_config=pricing_config,
                seed=seed,
                fuzzy_controller=fuzzy_controller,
                fuzzy_repair_params=fuzzy_repair_params,
            )

            # Record run metrics
            for res in [res_prop, res_fuzzy]:
                m = res.metrics
                raw_records.append({
                    "scenario": sc_name,
                    "seed": seed,
                    "algorithm": res.algorithm_name,
                    "deadline_satisfaction_pct": m.deadline_satisfaction_pct,
                    "total_unmet_energy_kwh": m.total_unmet_energy_kwh,
                    "electricity_cost": m.electricity_cost,
                    "max_capacity_violation_kw": m.max_capacity_violation_kw,
                    "num_violating_slots": m.num_violating_slots,
                    "runtime_seconds": m.runtime_seconds,
                    "jains_fairness_index": m.jains_fairness_index,
                    "total_delivered_energy_kwh": m.total_delivered_energy_kwh,
                    "total_requested_energy_kwh": m.total_requested_energy_kwh,
                    "objective_value": m.objective_value,
                })

                # Convergence history
                for it, fit in enumerate(res.convergence_history):
                    conv_records.append({
                        "scenario": sc_name,
                        "seed": seed,
                        "algorithm": res.algorithm_name,
                        "iteration": it,
                        "best_fitness": fit,
                    })

            # Save representative schedule for first seed of each scenario
            if s_idx == 0:
                for res in [res_prop, res_fuzzy]:
                    n_evs, n_slots = res.schedule.shape
                    for ev_idx in range(n_evs):
                        for slot_idx in range(n_slots):
                            representative_schedules.append({
                                "scenario": sc_name,
                                "seed": seed,
                                "algorithm": res.algorithm_name,
                                "ev_id": f"EV_{ev_idx+1:02d}",
                                "slot_idx": slot_idx,
                                "time_hours": slot_idx * rep_station.slot_duration_hours,
                                "charging_power_kw": round(res.schedule[ev_idx, slot_idx], 3),
                            })

            print(f"  Seed {seed}: Prop Unmet={res_prop.metrics.total_unmet_energy_kwh:.2f} kWh, "
                  f"Fuzzy Unmet={res_fuzzy.metrics.total_unmet_energy_kwh:.2f} kWh | "
                  f"Prop Sat={res_prop.metrics.deadline_satisfaction_pct:.1f}%, "
                  f"Fuzzy Sat={res_fuzzy.metrics.deadline_satisfaction_pct:.1f}%")

    # Build DataFrames
    df_raw = pd.DataFrame(raw_records)
    df_conv = pd.DataFrame(conv_records)
    df_sched = pd.DataFrame(representative_schedules)

    # Save raw CSVs
    raw_path = os.path.join(output_dir, "raw", "run_metrics.csv")
    conv_path = os.path.join(output_dir, "raw", "convergence_histories.csv")
    sched_path = os.path.join(output_dir, "raw", "representative_schedules.csv")

    df_raw.to_csv(raw_path, index=False)
    df_conv.to_csv(conv_path, index=False)
    df_sched.to_csv(sched_path, index=False)

    # Build summary statistics (mean and std)
    metric_cols = [
        "deadline_satisfaction_pct",
        "total_unmet_energy_kwh",
        "electricity_cost",
        "runtime_seconds",
        "jains_fairness_index",
        "max_capacity_violation_kw",
        "objective_value",
    ]

    summary_rows = []
    for (sc, alg), grp in df_raw.groupby(["scenario", "algorithm"]):
        row = {"scenario": sc, "algorithm": alg, "num_runs": len(grp)}
        for m in metric_cols:
            row[f"{m}_mean"] = round(float(grp[m].mean()), 3)
            row[f"{m}_std"] = round(float(grp[m].std()), 3)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "summaries", "scenario_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print(f"\n=== Experiments Complete ===")
    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {conv_path}")
    print(f"Saved: {sched_path}")

    return df_raw, df_summary
