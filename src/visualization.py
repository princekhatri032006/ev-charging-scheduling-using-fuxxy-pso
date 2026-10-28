"""Publication-quality visualization generation from experimental results."""

import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plotting_style():
    """Sets a clean, publication-ready plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "font.family": "sans-serif",
    })


def plot_metric_bar(
    df_raw: pd.DataFrame,
    metric_name: str,
    metric_label: str,
    output_path: str,
    title: str,
    higher_is_better: bool = False,
):
    """Generates a grouped bar chart comparing algorithms across scenarios with error bars."""
    scenarios = df_raw["scenario"].unique()
    algorithms = ["PSO_proportional", "PSO_fuzzy_urgency"]
    alg_labels = {
        "PSO_proportional": "PSO + Proportional Repair",
        "PSO_fuzzy_urgency": "PSO + Fuzzy Urgency Repair",
    }
    colors = {"PSO_proportional": "#4575b4", "PSO_fuzzy_urgency": "#d73027"}

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    x = np.arange(len(scenarios))
    width = 0.35

    for idx, alg in enumerate(algorithms):
        sub_df = df_raw[df_raw["algorithm"] == alg]
        means = [sub_df[sub_df["scenario"] == sc][metric_name].mean() for sc in scenarios]
        stds = [sub_df[sub_df["scenario"] == sc][metric_name].std() for sc in scenarios]

        offset = (idx - 0.5) * width
        rects = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=alg_labels[alg],
            color=colors[alg],
            alpha=0.88,
            edgecolor="black",
            linewidth=0.8,
        )

        # Add data labels
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    sc_display_labels = [sc.replace("_", " ").title() for sc in scenarios]
    ax.set_ylabel(metric_label, fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sc_display_labels, fontweight="bold")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved: {output_path}")


def plot_convergence_curves(
    df_conv: pd.DataFrame,
    output_path: str,
):
    """Plots PSO best-fitness convergence curves across iterations for all scenarios."""
    scenarios = df_conv["scenario"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    axes = axes.flatten()

    alg_labels = {
        "PSO_proportional": "PSO + Proportional",
        "PSO_fuzzy_urgency": "PSO + Fuzzy Urgency",
    }
    colors = {"PSO_proportional": "#4575b4", "PSO_fuzzy_urgency": "#d73027"}

    for idx, sc in enumerate(scenarios):
        ax = axes[idx]
        sc_data = df_conv[df_conv["scenario"] == sc]

        for alg in ["PSO_proportional", "PSO_fuzzy_urgency"]:
            alg_data = sc_data[sc_data["algorithm"] == alg]
            # Group by iteration: mean and std across seeds
            iter_grp = alg_data.groupby("iteration")["best_fitness"]
            mean_fit = iter_grp.mean()
            std_fit = iter_grp.std().fillna(0)
            iters = mean_fit.index.values

            ax.plot(
                iters,
                mean_fit,
                label=alg_labels[alg],
                color=colors[alg],
                linewidth=2.0,
            )
            ax.fill_between(
                iters,
                mean_fit - std_fit,
                mean_fit + std_fit,
                color=colors[alg],
                alpha=0.15,
            )

        ax.set_title(f"Scenario: {sc.replace('_', ' ').title()}", fontweight="bold")
        ax.set_xlabel("PSO Iteration")
        ax.set_ylabel("Best Objective J")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("PSO Convergence Histories by Scenario (Mean ± 1 SD)", fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved: {output_path}")


def plot_schedule_heatmap(
    df_sched: pd.DataFrame,
    scenario_name: str,
    output_path: str,
):
    """Plots side-by-side heatmaps of representative schedules for Proportional vs. Fuzzy."""
    sc_sched = df_sched[df_sched["scenario"] == scenario_name]
    if sc_sched.empty:
        sc_sched = df_sched

    algorithms = ["PSO_proportional", "PSO_fuzzy_urgency"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300, sharey=True)

    alg_titles = {
        "PSO_proportional": "PSO + Proportional Repair",
        "PSO_fuzzy_urgency": "PSO + Fuzzy Urgency Repair",
    }

    # Find global max power for common colormap scaling
    vmax = max(1.0, float(sc_sched["charging_power_kw"].max()))

    for idx, alg in enumerate(algorithms):
        ax = axes[idx]
        alg_data = sc_sched[sc_sched["algorithm"] == alg]

        pivot_table = alg_data.pivot(index="ev_id", columns="time_hours", values="charging_power_kw")
        # Ensure row ordering
        pivot_table = pivot_table.sort_index()

        im = ax.imshow(
            pivot_table.values,
            aspect="auto",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=vmax,
            origin="lower",
        )

        ax.set_title(alg_titles[alg], fontweight="bold")
        ax.set_xlabel("Time (Hours)")
        if idx == 0:
            ax.set_ylabel("EV Identifier")
            ax.set_yticks(np.arange(len(pivot_table.index)))
            ax.set_yticklabels(pivot_table.index, fontsize=8)

        # Set x-ticks to hours
        times = pivot_table.columns.values
        tick_indices = np.arange(0, len(times), 4)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f"{times[i]:.1f}h" for i in tick_indices], fontsize=8)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.03)
    cbar.set_label("Assigned Charging Power (kW)", fontweight="bold")

    plt.suptitle(f"Representative Charging Schedule Heatmap ({scenario_name.replace('_', ' ').title()})", fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved: {output_path}")


def plot_station_power_profile(
    df_sched: pd.DataFrame,
    station_capacity_kw: float,
    scenario_name: str,
    output_path: str,
):
    """Plots aggregate station charging power profile over time vs. station capacity."""
    sc_sched = df_sched[df_sched["scenario"] == scenario_name]
    if sc_sched.empty:
        sc_sched = df_sched

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    alg_labels = {
        "PSO_proportional": "PSO + Proportional Repair",
        "PSO_fuzzy_urgency": "PSO + Fuzzy Urgency Repair",
    }
    colors = {"PSO_proportional": "#4575b4", "PSO_fuzzy_urgency": "#d73027"}

    for alg in ["PSO_proportional", "PSO_fuzzy_urgency"]:
        alg_data = sc_sched[sc_sched["algorithm"] == alg]
        slot_power = alg_data.groupby("time_hours")["charging_power_kw"].sum()
        times = slot_power.index.values
        powers = slot_power.values

        ax.step(
            times,
            powers,
            where="post",
            label=alg_labels[alg],
            color=colors[alg],
            linewidth=2.0,
        )

    # Plot station capacity limit line
    ax.axhline(
        y=station_capacity_kw,
        color="#2b83ba",
        linestyle="--",
        linewidth=2.2,
        label=f"Station Capacity Limit ({station_capacity_kw:.0f} kW)",
    )

    ax.set_xlabel("Time (Hours)", fontweight="bold")
    ax.set_ylabel("Total Station Power (kW)", fontweight="bold")
    ax.set_title(
        f"Station Power Utilization Over Time ({scenario_name.replace('_', ' ').title()})",
        fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved: {output_path}")


def generate_all_plots(results_dir: str = "results", station_capacity_kw: float = 50.0):
    """Generates all 7 required publication-ready plots using actual saved CSV data."""
    set_plotting_style()
    raw_csv = os.path.join(results_dir, "raw", "run_metrics.csv")
    conv_csv = os.path.join(results_dir, "raw", "convergence_histories.csv")
    sched_csv = os.path.join(results_dir, "raw", "representative_schedules.csv")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Missing {raw_csv}. Run experiments first.")

    df_raw = pd.read_csv(raw_csv)
    df_conv = pd.read_csv(conv_csv) if os.path.exists(conv_csv) else None
    df_sched = pd.read_csv(sched_csv) if os.path.exists(sched_csv) else None

    # 1. Deadline satisfaction
    plot_metric_bar(
        df_raw=df_raw,
        metric_name="deadline_satisfaction_pct",
        metric_label="Deadline Satisfaction (%)",
        output_path=os.path.join(plots_dir, "deadline_satisfaction.png"),
        title="EV Deadline Satisfaction Percentage by Scenario and Method",
        higher_is_better=True,
    )

    # 2. Unmet energy
    plot_metric_bar(
        df_raw=df_raw,
        metric_name="total_unmet_energy_kwh",
        metric_label="Total Unmet Energy (kWh)",
        output_path=os.path.join(plots_dir, "unmet_energy.png"),
        title="Total Unmet Energy by Scenario and Method (Lower is Better)",
        higher_is_better=False,
    )

    # 3. Electricity cost
    plot_metric_bar(
        df_raw=df_raw,
        metric_name="electricity_cost",
        metric_label="Electricity Cost (Currency Units)",
        output_path=os.path.join(plots_dir, "electricity_cost.png"),
        title="Total Electricity Cost by Scenario and Method",
        higher_is_better=False,
    )

    # 4. Runtime
    plot_metric_bar(
        df_raw=df_raw,
        metric_name="runtime_seconds",
        metric_label="Optimization Runtime (seconds)",
        output_path=os.path.join(plots_dir, "runtime_comparison.png"),
        title="Optimization Runtime Comparison (Excluding Generation & Plotting)",
        higher_is_better=False,
    )

    # 5. Convergence curves
    if df_conv is not None:
        plot_convergence_curves(
            df_conv=df_conv,
            output_path=os.path.join(plots_dir, "pso_convergence.png"),
        )

    # 6. Schedule heatmap (high_demand scenario)
    if df_sched is not None:
        rep_scenario = "high_demand" if "high_demand" in df_sched["scenario"].values else df_sched["scenario"].iloc[0]
        plot_schedule_heatmap(
            df_sched=df_sched,
            scenario_name=rep_scenario,
            output_path=os.path.join(plots_dir, "representative_schedule_heatmap.png"),
        )

        # 7. Station power utilization
        plot_station_power_profile(
            df_sched=df_sched,
            station_capacity_kw=station_capacity_kw,
            scenario_name=rep_scenario,
            output_path=os.path.join(plots_dir, "station_power_profile.png"),
        )

    print("\nAll 7 visualization charts generated successfully in results/plots/.")
