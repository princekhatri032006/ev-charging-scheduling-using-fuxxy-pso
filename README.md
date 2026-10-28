# Deadline-Aware EV Charging Scheduling Using PSO with Fuzzy Urgency-Guided Power Allocation

An undergraduate Soft Computing research project investigating whether a **Mamdani Fuzzy Urgency-Guided Power Allocation** repair mechanism improves electric vehicle (EV) charging schedules compared to conventional **Proportional Power-Scaling Repair** within a continuous Particle Swarm Optimization (PSO) framework.

---

## 1. Project Objective

In public and workplace electric vehicle charging stations, aggregate charging requests often exceed transformer capacity ratings during peak hours. In a PSO-based charging scheduler, candidate solutions (particles) proposing power allocations that exceed station capacity must be repaired into feasible solutions.

This project implements, tests, and empirically evaluates two capacity repair strategies on identical paired benchmark scenarios:
- **Baseline (Method A)**: Continuous PSO with Proportional Capacity Repair.
- **Proposed (Method B)**: Continuous PSO with Mamdani Fuzzy Urgency-Guided Iterative Capped Repair.

Both algorithms share:
- Identical synthetic EV fleets and arrival/departure profiles.
- Identical station capacity limits and simulation time grids (15-min slots).
- Identical PSO optimizer implementation, population sizes, and iteration budgets.
- Identical electricity pricing profiles and objective functions.
- Paired initial swarm positions using synchronized random seeds.

---

## 2. Mathematical Model

### 2.1 Parameters & Notation
- **Simulation Horizon**: $H = 12.0\text{ hours}$, slot duration $\Delta t = 0.25\text{ hours}$ (15 minutes), yielding $T = 48$ discrete time slots.
- **EV Fleet**: $N = 20$ vehicles, each defined by:
  - Arrival time $a_i \in [0, H)$ hours.
  - Departure deadline $d_i \in (a_i, H]$ hours.
  - Required energy $E_i^\text{req} \in \mathbb{R}^+$ kWh.
  - Maximum charging power $P_i^\text{max} \in \mathbb{R}^+$ kW.
- **Station**:
  - Capacity $C(t) = C_\text{station}$ kW.
  - TOU electricity price vector $p(t)$ ($/kWh).
- **Decision Matrix**: $P \in \mathbb{R}^{N \times T}$ where $P[i, t]$ is power assigned to EV $i$ at slot $t$ in kW.

### 2.2 Constraints
1. **Nonnegativity**: $P[i, t] \ge 0 \quad \forall i, t$.
2. **Individual Peak Power**: $P[i, t] \le P_i^\text{max} \quad \forall i, t$.
3. **Availability Window**: $P[i, t] = 0$ if slot $t$ is outside $[a_i, d_i]$.
4. **Station Capacity**: $\sum_{i=1}^N P[i, t] \le C(t) \quad \forall t \in \{0, \dots, T-1\}$.
5. **Demand Saturation**: Charging stops once $E_i^\text{del}(t) = E_i^\text{req}$. Excess power is removed before evaluation.

### 2.3 Objective Function
$$J(P) = J_\text{cost} + \lambda_\text{unmet} J_\text{unmet}$$
$$J_\text{cost} = \sum_{t=0}^{T-1} \left( \sum_{i=1}^N P[i, t] \right) \cdot p(t) \cdot \Delta t$$
$$J_\text{unmet} = \sum_{i=1}^N \max\left(0, E_i^\text{req} - E_i^\text{del}(T)\right)$$
where $\lambda_\text{unmet} = 50.0\text{ \$/kWh}$ prioritizes deadline fulfillment.

---

## 3. Fuzzy Urgency Controller

### 3.1 Input Variables & Membership Functions
1. **Remaining Demand ($E^\text{rem}$ in kWh)**:
   - *Low*: Trapezoidal $[0, 0, 8, 16]$
   - *Medium*: Triangular $[10, 20, 30]$
   - *High*: Trapezoidal $[24, 32, 60, 60]$

2. **Charging Slack ($S$ in hours)**:
   $$S_i(t) = (d_i - t\Delta t) - \frac{E_i^\text{rem}(t)}{P_i^\text{max}}$$
   - *Critical*: Trapezoidal $[-5, -5, 0.25, 0.75]$
   - *Limited*: Triangular $[0.5, 1.5, 2.5]$
   - *Comfortable*: Trapezoidal $[2.0, 3.5, 15, 15]$

3. **Urgency Output ($U \in [0, 1]$)**:
   - *Low*: Trapezoidal $[0, 0, 0.15, 0.35]$
   - *Medium*: Triangular $[0.25, 0.45, 0.65]$
   - *High*: Triangular $[0.55, 0.72, 0.88]$
   - *Very High*: Trapezoidal $[0.78, 0.90, 1.0, 1.0]$

### 3.2 9-Rule Mamdani Inference Base
```
1. IF Demand is Low    AND Slack is Critical    THEN Urgency is High
2. IF Demand is Medium AND Slack is Critical    THEN Urgency is Very High
3. IF Demand is High   AND Slack is Critical    THEN Urgency is Very High
4. IF Demand is Low    AND Slack is Limited     THEN Urgency is Medium
5. IF Demand is Medium AND Slack is Limited     THEN Urgency is High
6. IF Demand is High   AND Slack is Limited     THEN Urgency is High
7. IF Demand is Low    AND Slack is Comfortable THEN Urgency is Low
8. IF Demand is Medium AND Slack is Comfortable THEN Urgency is Low
9. IF Demand is High   AND Slack is Comfortable THEN Urgency is Medium
```
Defuzzification uses standard centroid (center of gravity) defuzzification.

---

## 4. Repair Operators Pseudocode

### 4.1 Proportional Repair Pseudocode
```python
def repair_proportional(candidate_power, evs, capacity, remaining_energy, t, delta_t):
    for i in active_evs:
        clip candidate_power[i] to min(ev.max_power, remaining_energy[i] / delta_t)
    total_requested = sum(candidate_power)
    if total_requested <= capacity:
        return candidate_power
    scale = capacity / total_requested
    return candidate_power * scale
```

### 4.2 Fuzzy Urgency-Guided Iterative Capped Repair Pseudocode
```python
def repair_fuzzy_urgency(candidate_power, evs, capacity, remaining_energy, t, delta_t):
    for i in active_evs:
        cap[i] = min(ev.max_power, remaining_energy[i] / delta_t)
        urgency[i] = fuzzy_controller.evaluate(remaining_energy[i], slack[i])
        weight[i] = (urgency[i] + epsilon) * (candidate_power[i] + delta)

    C_rem = capacity
    allocated = zeros(N)
    active_set = set(active_evs)

    while active_set and C_rem > 0:
        tentative_add = C_rem * (weight / sum(active_weights))
        capped_evs = [i for i in active_set if allocated[i] + tentative_add[i] >= cap[i]]
        if capped_evs:
            for i in capped_evs:
                headroom = cap[i] - allocated[i]
                allocated[i] = cap[i]
                C_rem -= headroom
                active_set.remove(i)
            # Redistribute leftover capacity among remaining uncapped EVs
        else:
            for i in active_set:
                allocated[i] += tentative_add[i]
            C_rem = 0
            break
    return allocated
```

---

## 5. Installation Instructions

Requires Python 3.11+.

```bash
# Clone or navigate to the project directory
cd ev-charging-pso-fuzzy

# Install dependencies
pip install -r requirements.txt
```

---

## 6. Execution Commands

### Run Automated Unit Tests (21 Tests)
```bash
pytest -v
```

### Run Fast Single-Seed Demonstration
```bash
python main.py --mode demo
```

### Run Full 10-Seed Paired Experiments Across 4 Scenarios
```bash
python main.py --mode experiments --seeds 10
```

### Re-Generate Publication Plots from Saved Results
```bash
python main.py --mode plot
```

---

## 7. Directory Structure & Results

```
ev-charging-pso-fuzzy/
├── config/
│   └── default_config.yaml         # Simulation, pricing, and PSO configuration
├── src/
│   ├── models.py                   # EV, StationConfig, SimulationMetrics dataclasses
│   ├── generator.py                # Synthetic EV fleet generator & TOU profile
│   ├── fuzzy_controller.py         # Pure Mamdani fuzzy controller & LUT engine
│   ├── repair.py                   # Proportional & iterative capped repair
│   ├── evaluator.py                # Chronological simulator & objective evaluation
│   ├── pso.py                      # Continuous PSO optimizer
│   ├── experiment.py               # Paired experiment runner & CSV aggregator
│   └── visualization.py            # Matplotlib plotting routines
├── tests/                          # 21 automated pytest unit tests
├── results/
│   ├── raw/
│   │   ├── run_metrics.csv         # Raw metrics for all runs
│   │   ├── convergence_histories.csv
│   │   └── representative_schedules.csv
│   ├── summaries/
│   │   └── scenario_summary.csv    # Means and standard deviations
│   └── plots/                      # Publication figures (PNG, 300 DPI)
├── report/
│   ├── methodology.md              # Mathematical modeling & algorithm details
│   └── results_template.md         # Measured results tables & findings
├── main.py                         # CLI entry point
├── requirements.txt
└── README.md
```

---

## 8. Known Limitations

1. **Unidirectional Power Flow (G2V)**: Vehicle-to-Grid (V2G) discharging is not modeled.
2. **Fixed Converter Efficiency**: Charging efficiency is assumed constant ($100\%$ nominal) without battery non-linear CC-CV tapering effects above $80\%$ SOC.
3. **Deterministic Arrivals**: Arrival and departure times are known upon connection, rather than stochastic departure updates during parking.
4. **Computational Trade-off**: Fuzzy urgency evaluation requires additional floating-point operations per slot, resulting in slightly higher runtime (~2x) compared to simple proportional scaling.
