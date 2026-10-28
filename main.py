"""Command Line Interface (CLI) for EV Charging Scheduling using PSO and Fuzzy Urgency Repair."""

import argparse
import os
import sys
import yaml
import pandas as pd

from src.fuzzy_controller import FuzzyUrgencyController
from src.generator import generate_scenario, characterize_scenario
from src.experiment import run_paired_seed_comparison, run_full_experiments
from src.visualization import generate_all_plots


def run_demo(config_path: str = "config/default_config.yaml"):
    """Runs a quick paired demonstration comparing Proportional vs. Fuzzy Urgency PSO."""
    print("=" * 70)
    print(" DEMO: EV Charging Scheduling - PSO with Proportional vs. Fuzzy Urgency Repair")
    print("=" * 70)

    if not os.path.exists(config_path):
        # Check relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Use high_demand scenario for an illustrative comparison
    scenario_name = "high_demand"
    sc_params = cfg["scenarios"][scenario_name].copy()
    sc_params["horizon_hours"] = cfg["simulation"]["horizon_hours"]
    sc_params["slot_duration_hours"] = cfg["simulation"]["slot_duration_hours"]
    sc_params["lambda_unmet"] = cfg["simulation"]["lambda_unmet"]

    # Demo PSO params with smaller iteration budget for speed
    demo_pso_params = cfg["pso"].copy()
    demo_pso_params["max_iterations"] = 25
    demo_pso_params["swarm_size"] = 20

    seed = 42
    fuzzy_controller = FuzzyUrgencyController()

    print(f"\nScenario: {scenario_name} (Capacity: {sc_params['station_capacity_kw']} kW, EVs: {sc_params['num_evs']})")
    print(f"Running paired optimization with seed {seed} (25 iterations, swarm size 20)...")

    res_prop, res_fuzzy = run_paired_seed_comparison(
        scenario_name=scenario_name,
        scenario_params=sc_params,
        pso_params=demo_pso_params,
        pricing_config=cfg.get("pricing", {}),
        seed=seed,
        fuzzy_controller=fuzzy_controller,
        fuzzy_repair_params=cfg.get("fuzzy_repair", {}),
    )

    m_prop = res_prop.metrics
    m_fuzzy = res_fuzzy.metrics

    print("\n" + "=" * 65)
    print(f"{'Metric':<32} | {'Proportional':<14} | {'Fuzzy Urgency':<14}")
    print("-" * 65)
    print(f"{'Deadline Satisfaction (%)':<32} | {m_prop.deadline_satisfaction_pct:<14.1f} | {m_fuzzy.deadline_satisfaction_pct:<14.1f}")
    print(f"{'Total Unmet Energy (kWh)':<32} | {m_prop.total_unmet_energy_kwh:<14.2f} | {m_fuzzy.total_unmet_energy_kwh:<14.2f}")
    print(f"{'Total Delivered Energy (kWh)':<32} | {m_prop.total_delivered_energy_kwh:<14.2f} | {m_fuzzy.total_delivered_energy_kwh:<14.2f}")
    print(f"{'Electricity Cost (Currency)':<32} | {m_prop.electricity_cost:<14.2f} | {m_fuzzy.electricity_cost:<14.2f}")
    print(f"{'Objective Function J':<32} | {m_prop.objective_value:<14.2f} | {m_fuzzy.objective_value:<14.2f}")
    print(f"{'Jain Fairness Index':<32} | {m_prop.jains_fairness_index:<14.4f} | {m_fuzzy.jains_fairness_index:<14.4f}")
    print(f"{'Max Capacity Violation (kW)':<32} | {m_prop.max_capacity_violation_kw:<14.6f} | {m_fuzzy.max_capacity_violation_kw:<14.6f}")
    print(f"{'Capacity Violating Slots':<32} | {m_prop.num_violating_slots:<14d} | {m_fuzzy.num_violating_slots:<14d}")
    print(f"{'Runtime (seconds)':<32} | {m_prop.runtime_seconds:<14.2f} | {m_fuzzy.runtime_seconds:<14.2f}")
    print("=" * 65)

    assert m_prop.num_violating_slots == 0, "Capacity violation detected in Proportional schedule!"
    assert m_fuzzy.num_violating_slots == 0, "Capacity violation detected in Fuzzy schedule!"
    print("\n[OK] All physical constraints and station limits verified with 0 capacity violations.")
    print("Demo completed successfully. Run `python main.py --mode experiments` for multi-seed statistical analysis.")


def main():
    parser = argparse.ArgumentParser(
        description="Deadline-Aware EV Charging Scheduling using PSO and Fuzzy Urgency Repair."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "experiments", "plot"],
        help="Execution mode: 'demo' (quick single run), 'experiments' (multi-seed study), or 'plot' (render saved charts).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of random seeds per scenario for experiments mode (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save experimental results and plots.",
    )

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(args.config)
    elif args.mode == "experiments":
        df_raw, df_summary = run_full_experiments(
            config_path=args.config,
            output_dir=args.output_dir,
            num_seeds=args.seeds,
        )
        print("\nGenerating publication-quality charts from experiment results...")
        generate_all_plots(results_dir=args.output_dir)
    elif args.mode == "plot":
        generate_all_plots(results_dir=args.output_dir)


if __name__ == "__main__":
    main()
