# OTiS benchmark

Generated 2026-11-07 21:34:28 by `python main.py --benchmark`. Raw numbers: `otis/outputs/benchmark_results.json`.

Model: **large**, 12,723,300 parameters, best step 30000 of 30000, trained 174 s on `cuda:0`.
Problem: 50 steps × 0.08 s, |u| ≤ 1.0, λ = 0.01, start/goal in ±2.0, initial velocity in ±0.5.
Test set: 5 disjoint folds × 1000 problems drawn from PRNG streams the trainer never touches. Every method, including the convex solver, runs on every problem; Clarabel matches the exact optimum to 4.5e-08 in objective.

Every method is scored by rolling its thrust plan through the same RK4 simulator with the same objective J = ‖p_T − goal‖² + λ Σ‖u‖².

## Head-to-head (means over all test problems)

| Method | Objective J | Miss (m) | p99 miss (m) | Control effort Σu² | Δv (m/s) |
| --- | --- | --- | --- | --- | --- |
| OTiS (neural net) | 0.04658 | 0.0149 | 0.0329 | 4.630 | 0.946 |
| Convex optimization (CVXPY + Clarabel) | 0.04658 | 0.0148 | 0.0325 | 4.631 | 0.946 |
| Exact QP optimum (KKT) | 0.04658 | 0.0148 | 0.0325 | 4.631 | 0.946 |
| Apollo E-guidance (ZEM law) | 0.04686 | 0.0000 | 0.0000 | 4.686 | 0.942 |

## Numerical deltas

ΔJ % is a ratio of means: (mean J_a − mean J_b) / mean J_b.

| Comparison | ΔJ | ΔJ % (of mean) | Δ miss (m) | Δ control effort | Δ control effort % | Δ Δv % |
| --- | --- | --- | --- | --- | --- | --- |
| Neural − convex optimization (status quo) | +8.28e-07 | +0.002% | +1.54e-04 | -3.79e-04 | -0.008% | -0.007% |
| Neural − exact optimum (optimality gap) | +8.28e-07 | +0.002% | +1.54e-04 | -3.79e-04 | -0.008% | -0.007% |
| Neural − Apollo E-guidance | -2.86e-04 | -0.610% | +1.49e-02 | -5.62e-02 | -1.199% | +0.354% |
| E-guidance − exact optimum | +2.87e-04 | +0.615% | -1.48e-02 | +5.58e-02 | +1.205% | -0.360% |

Per problem, the network scores a lower objective than E-guidance on **99.4%** of test cases.
Mean per-mission optimality gap of the network (each mission's (J_nn − J*)/J*, then averaged): 0.0184% ± 0.00704% across folds, bootstrap 95% CI [0.0145%, 0.0238%], median 0.00164%, p95 0.0416%, p99 0.323%.

KKT referee vs training: same discretization (dt = 0.08 s, N = 50, checkpoint plant matches); the KKT closed-form terminal map agrees with the training RK4 simulator to 1.4e-06 m on 1000 KKT plans.

## Compute time per plan

| Method | ms per problem |
| --- | --- |
| Neural net, single problem, GPU | 1.270 |
| Neural net, single problem, CPU (Ryzen) | 1.125 |
| Neural net, batched | 0.00823 |
| Convex optimization (Clarabel, median) | 0.960 |
| E-guidance, batched closed loop | 0.00313 |
| Exact KKT, vectorized NumPy | 0.01164 |

Network vs convex solver: 0.85× for one plan at a time (best of GPU/CPU), 117× when many plans are batched on the GPU. For this small linear problem a single convex solve is already about a millisecond, so the honest claim is batch throughput and constant cost, not single-plan latency.

## Cross-validation

| Check | Result |
| --- | --- |
| RK4 (float32) vs closed form, 200 random plans | max abs error 1.11e-06 |
| RK4 (float32) vs SciPy DOP853 (rtol 1e-12) | max abs error 1.11e-06 |
| Closed form vs DOP853 (both float64) | max abs error 4.00e-15 |
| Autodiff vs finite differences, 15 entries | max abs error 1.76e-04 |
| Clarabel vs exact KKT, 100 problems | max Δu 1.2e-07, max ΔJ 1.5e-08 |
| OSQP vs exact KKT, 100 problems | max Δu 8.5e-11, max ΔJ 0.0e+00 |
| Optimality gap per fold | 0.020%, 0.030%, 0.015%, 0.014%, 0.013% |

## Out-of-distribution stress test

Initial velocity drawn from ±1.0 (2× the training range). Network optimality gap 1.99% (in distribution: 0.02%), mean miss 0.0366 vs optimum 0.0216. E-guidance gap on the same set: 0.62%.

## How to read this honestly

- Two averages appear above. "ΔJ %" compares mean objectives across all missions; the optimality gap averages each mission's own relative gap, which weights short, cheap missions more. Both are reported so neither hides the other.
- The exact KKT solver exists only because this plant is linear. It is the referee, not a competitor: it proves where the true optimum is so the network's gap is measured, not estimated.
- The status-quo comparison is the convex solver (how flight systems such as G-FOLD and Falcon 9 landing guidance solve this class of problem) and Apollo-heritage E-guidance (the classic closed-form law). See SOURCES.md.
- The network's advantage is a fixed, tiny inference cost that does not grow with constraints or nonlinearity, which is where per-problem solvers get expensive. Its cost is a small optimality gap and no hard guarantee.
- Timing was measured on the machine listed in the JSON, so the numbers change across hardware.
