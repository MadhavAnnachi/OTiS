# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary audiences are competition and demo judges evaluating the research claim in a short session, and researchers or engineers exploring neural versus classical guidance. Both open the same interactive demo.

## Product Purpose

OTiS (Optimized Trajectory Simulator) turns a start state, velocity, and goal into a full control plan. A neural network outputs the horizon of thrust commands in one forward pass; training differentiates through an RK4 physics rollout so miss and fuel gradients update the weights without a dataset or reinforcement learning.

Success means high accuracy and high fidelity to the plant, with higher planning efficiency than the status-quo per-problem solvers, so more people with less specialized compute can develop and explore this class of technology in robotics.

## Positioning

The mechanism is end-to-end differentiable planning: random missions are flown through a differentiable simulator and the exact effect of every thrust command on miss and fuel flows into the network. Neighboring products that only train with RL, imitate labeled trajectories, or solve each mission with a fresh convex program cannot truthfully claim the same training loop or the measured near-optimum gap with batch throughput on the same referee.

## Operating Context

Local interactive demo at `http://localhost:8765` (plain HTML/CSS/JS in `otis/web/`, served by Python). Training and benchmarks are CLI-driven (`python main.py`, often via WSL2 for GPU). Judges and researchers use the Simulator to drag missions, Training to watch learning, and Benchmark to read scored comparisons against convex optimization and Apollo-heritage E-guidance. Headless modes write checkpoints, `BENCHMARK.md`, and summary figures under `otis/`.

## Capabilities and Constraints

Confirmed today: interactive simulator with vehicle tabs for rocket, UAV, and USV, each with user-editable SI parameters (mass, thrust, drag, gravity, mission time, workspace, fuel weight, start and goal), derived readouts with warnings, and JSON import/export; convex transcription (CVXPY + Clarabel) and gravity/drag-compensated E-guidance on any thrust-plant parameters; one-click training of a network on the user's thrust-plant parameters, with networks matched to parameters by plant signature; training UI with model-size and step controls; benchmark dashboard backed by `outputs/benchmark_results.json`; physics and gradient self-checks; synthetic start/goal benchmark set with a provable KKT optimum; a static `docs/` copy of the UI for GitHub Pages (classical planners only).

Open / undecided: a public GitHub Pages URL for the Devpost demo link; faster training on stiff plants (high thrust-to-weight with a small fuel weight).

## Brand Commitments

Product name: OTiS (Optimized Trajectory Simulator), stylized with a capital T. The repository folder is `OTiS` and the Python package is `otis`. Visual identity: charcoal `#333333` background, gold `#f0bc5a` accent, square boxes with thin 1px outlines and no shadows (inspired by Wolfram|Alpha), Newsreader and Source Sans 3.

## Evidence on Hand

Real measured results and talking points live in the repo: [BENCHMARK.md](BENCHMARK.md), [RESEARCH.md](RESEARCH.md), [SOURCES.md](SOURCES.md), [README.md](README.md), `otis/outputs/` (including demo figures), and checkpoints under `otis/checkpoints/`. Future work must not invent testimonials, customers, or unpublished benchmarks.

## Product Principles

1. Accuracy and fidelity to the physics and referee come before presentation.
2. Efficiency that lowers the compute bar expands who can explore the technology.
3. Claims stay tied to the same simulator, objective, and baselines used in the repo.
4. The interactive demo exists to make the research claim inspectable, not to replace the numbers.
5. A network is shown only for the exact physics and objective it was trained on. Never display a network on parameters it was not trained for; offer to train instead.
