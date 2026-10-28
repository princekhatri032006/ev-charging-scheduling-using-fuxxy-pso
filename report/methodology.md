# Computational Methodology: Deadline-Aware EV Charging Scheduling Using PSO with Fuzzy Urgency-Guided Power Allocation

## 1. Problem Formulation & System Model

Electric Vehicle (EV) charging coordination is formulated as a constrained continuous optimization problem over a finite simulation horizon $H$ partitioned into $T$ uniform discrete time slots of duration $\Delta t = 0.25\text{ hours}$ (15 minutes).

### 1.1 Decision Variables & Schedule Matrix
Let $N$ denote the total number of electric vehicles serviced during the horizon. The primary decision variable is the continuous power schedule matrix:
$$P \in \mathbb{R}^{N \times T}$$
where $P[i, t]$ denotes the charging power (kW) assigned to vehicle $i \in \{1, \dots, N\}$ during time interval $[t\Delta t, (t+1)\Delta t)$.

### 1.2 Physical and Operational Constraints
Every admissible schedule $P$ must satisfy five fundamental constraints:

1. **Non-negativity**: No EV may discharge or inject power into the station grid:
   $$P[i, t] \ge 0 \quad \forall i \in \{1, \dots, N\}, \, \forall t \in \{0, \dots, T-1\}$$

2. **EV Physical Converter Limits**: Charging power cannot exceed vehicle $i$'s onboard charger rating $P_i^\text{max}$:
   $$P[i, t] \le P_i^\text{max} \quad \forall i, t$$

3. **Temporal Availability Window**: Charging can only occur when vehicle $i$ is connected to the charger:
   $$P[i, t] = 0 \quad \text{if } t\Delta t < a_i - \epsilon_\text{tol} \quad \text{or} \quad (t+1)\Delta t > d_i + \epsilon_\text{tol}$$
   where $a_i$ is arrival time and $d_i$ is departure deadline.

4. **Station Grid Transformer Limit**: The aggregate power drawn by all concurrently charging vehicles at time slot $t$ must not exceed station transformer capacity $C(t)$:
   $$\sum_{i=1}^N P[i, t] \le C(t) \quad \forall t \in \{0, \dots, T-1\}$$

5. **No Overcharging (Demand Saturation)**: Cumulative energy delivered cannot exceed the battery demand $E_i^\text{req}$:
   $$E_i^\text{del}(T) = \sum_{t=0}^{T-1} P[i, t] \Delta t \le E_i^\text{req}$$
   Charging stops immediately when $E_i^\text{del}(t) = E_i^\text{req}$.

### 1.3 Objective Function
Both optimization methods evaluate candidate schedules using the identical cost-penalty objective:
$$J(P) = J_\text{cost} + \lambda_\text{unmet} J_\text{unmet}$$
where:
$$J_\text{cost} = \sum_{t=0}^{T-1} \left( \sum_{i=1}^N P[i, t] \right) \cdot p(t) \cdot \Delta t$$
$$J_\text{unmet} = \sum_{i=1}^N \max\left(0, E_i^\text{req} - E_i^\text{del}(T)\right)$$
$p(t)$ is the Time-of-Use (TOU) electricity tariff (currency units/kWh) and $\lambda_\text{unmet} = 50.0$ currency units/kWh is the penalty parameter, ensuring that meeting vehicle charging deadlines dominates marginal electricity cost savings.

---

## 2. Optimization Algorithm: Continuous Particle Swarm Optimization (PSO)

Continuous Particle Swarm Optimization operates on a swarm of $S$ particles exploring the flattened $D$-dimensional decision space, where $D = N \times T$.

### 2.1 State Representation & Velocity Updates
At iteration $k$, particle $p$ possesses position $x_p^{(k)} \in \mathbb{R}^D$ and velocity $v_p^{(k)} \in \mathbb{R}^D$.
$$\begin{aligned}
v_p^{(k+1)} &= w \cdot v_p^{(k)} + c_1 \cdot r_1 \odot \left(p_{\text{best}, p} - x_p^{(k)}\right) + c_2 \cdot r_2 \odot \left(g_{\text{best}} - x_p^{(k)}\right) \\
x_p^{(k+1)} &= x_p^{(k)} + v_p^{(k+1)}
\end{aligned}$$
where:
- $w = 0.7$ (inertia weight)
- $c_1 = 1.5$ (cognitive acceleration coefficient)
- $c_2 = 1.5$ (social acceleration coefficient)
- $r_1, r_2 \sim \mathcal{U}(0, 1)^D$ (stochastic vectors)
- $v_p^{(k+1)}$ is clamped to $[-v_\text{clamp}, v_\text{clamp}]$ with $v_\text{clamp} = 5.0\text{ kW}$.
- $x_p^{(k+1)}$ is clipped to $[0, P^\text{max}]$ and masked by connection availability.

---

## 3. Capacity Repair Mechanisms

When candidate power schedules violate station capacity $\sum_i P[i, t] > C(t)$, a repair operator maps infeasible proposals to feasible allocations in strict chronological order.

### 3.1 Baseline: Proportional Power-Scaling Repair
In each time slot $t$:
1. Zero out requests for unavailable or fully charged EVs.
2. Clip candidate power to physical converter rating $P_i^\text{max}$ and remaining energy head $\frac{E_i^\text{rem}(t)}{\Delta t}$.
3. If total request $P_\text{tot} \le C(t)$, retain candidate requests.
4. If $P_\text{tot} > C(t)$, apply uniform scaling:
   $$\alpha = \frac{C(t)}{P_\text{tot}}$$
   $$P_\text{alloc}[i, t] = P_\text{req}[i, t] \cdot \alpha$$

*Limitation*: Proportional scaling penalizes all vehicles equally regardless of urgency, starving vehicles with imminent departure deadlines.

### 3.2 Method B: Mamdani Fuzzy Urgency-Guided Iterative Capped Repair
To prioritize vehicles with tight deadlines and high energy needs, power allocation is guided by a Mamdani Fuzzy Logic Controller.

#### Fuzzy Inputs:
1. **Remaining Energy Demand**: $E_i^\text{rem}(t) = \max(0, E_i^\text{req} - E_i^\text{del}(t))$ [kWh]
   - *Low*: Trapezoidal $[0, 0, 8, 16]$
   - *Medium*: Triangular $[10, 20, 30]$
   - *High*: Trapezoidal $[24, 32, 60, 60]$
2. **Charging Slack**: $S_i(t) = (d_i - t\Delta t) - \frac{E_i^\text{rem}(t)}{P_i^\text{max}}$ [hours]
   - *Critical*: Trapezoidal $[-5, -5, 0.25, 0.75]$
   - *Limited*: Triangular $[0.5, 1.5, 2.5]$
   - *Comfortable*: Trapezoidal $[2.0, 3.5, 15, 15]$

#### Fuzzy Output:
- **Urgency**: $U \in [0, 1]$
  - *Low*: Trapezoidal $[0, 0, 0.15, 0.35]$
  - *Medium*: Triangular $[0.25, 0.45, 0.65]$
  - *High*: Triangular $[0.55, 0.72, 0.88]$
  - *Very High*: Trapezoidal $[0.78, 0.90, 1.0, 1.0]$

#### 9-Rule Mamdani Inference:
| Rule | Remaining Demand | Charging Slack | Urgency Output |
|:---:|:---:|:---:|:---:|
| 1 | Low | Critical | High |
| 2 | Medium | Critical | Very High |
| 3 | High | Critical | Very High |
| 4 | Low | Limited | Medium |
| 5 | Medium | Limited | High |
| 6 | High | Limited | High |
| 7 | Low | Comfortable | Low |
| 8 | Medium | Comfortable | Low |
| 9 | High | Comfortable | Medium |

Implication is performed via minimum $t$-norm, rule aggregation via maximum $s$-norm, and defuzzification via the centroid (Center of Gravity) method:
$$u_i(t) = \frac{\int_0^1 u \cdot \mu_\text{agg}(u) du}{\int_0^1 \mu_\text{agg}(u) du}$$

#### Iterative Capped Allocation Algorithm:
At time slot $t$, when $P_\text{tot} > C(t)$:
1. Calculate allocation weight for each active EV:
   $$w_i = (u_i(t) + \epsilon) \cdot (P_\text{req}[i, t] + \delta)$$
   where $\epsilon = 0.01$ and $\delta = 0.1$.
2. Define per-EV capacity cap:
   $$\text{cap}_i = \min\left(P_i^\text{max}, \frac{E_i^\text{rem}(t)}{\Delta t}\right)$$
3. Initialize unallocated station capacity $C_\text{rem} = C(t)$, active set $\mathcal{A} = \{i \mid \text{cap}_i > 0\}$, and allocated power $P_\text{alloc}[i] = 0$.
4. **Iterative redistribution loop**:
   - Compute tentative allocation:
     $$\Delta P_i = C_\text{rem} \cdot \frac{w_i}{\sum_{j \in \mathcal{A}} w_j}$$
   - Identify EVs that reach or exceed their cap: $\mathcal{C} = \{i \in \mathcal{A} \mid P_\text{alloc}[i] + \Delta P_i \ge \text{cap}_i\}$.
   - If $\mathcal{C} \ne \emptyset$:
     For each $i \in \mathcal{C}$:
     - Headroom = $\text{cap}_i - P_\text{alloc}[i]$
     - $P_\text{alloc}[i] = \text{cap}_i$
     - $C_\text{rem} \leftarrow C_\text{rem} - \text{Headroom}$
     - Remove $i$ from $\mathcal{A}$
     - Repeat iteration to redistribute leftover capacity among remaining uncapped EVs.
   - If $\mathcal{C} = \emptyset$:
     For all $i \in \mathcal{A}$: $P_\text{alloc}[i] \leftarrow P_\text{alloc}[i] + \Delta P_i$.
     $C_\text{rem} = 0$, break loop.

This guarantees $100\%$ capacity utilization without stranded unallocated power.

---

## 4. Experimental Scenarios

To stress-test both algorithms under diverse operating conditions, four benchmark scenarios are defined:

1. **Scenario A (Normal Demand)**: $C = 60\text{ kW}$, 20 EVs, arrivals spread across $[0, 6]\text{h}$, moderate energy demands ($10-25\text{ kWh}$), dwell times $4-8\text{h}$.
2. **Scenario B (High Demand)**: $C = 50\text{ kW}$, 20 EVs, concentrated arrivals $[1, 5]\text{h}$, high required energy ($25-40\text{ kWh}$), dwell times $4-7\text{h}$.
3. **Scenario C (Tight Deadlines)**: $C = 50\text{ kW}$, 20 EVs, short dwell times ($1.5-3.0\text{h}$), tight or negative charging slacks.
4. **Scenario D (Limited Station Capacity)**: $C = 25\text{ kW}$, 20 EVs, severely constrained grid connection creating persistent transformer bottlenecks.
