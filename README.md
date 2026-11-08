# OTiS

**Optimized Trajectory Simulator.** The project folder is `OTiS`; the Python package is `otis`.

You give it a vehicle's starting position, its velocity, and a target. A neural network reads that and outputs a complete plan: 50 thrust commands telling the vehicle how to fire its engines at each moment. The plan is flown through a physics simulator (F = m·a, integrated with RK4), and because the simulator is differentiable, the exact effect of every thrust command on the miss and on the fuel flows back into the network's weights. There is no dataset and no reinforcement learning. After training, any new mission is planned in a single forward pass.

**You define the vehicle.** Enter mass, thrust, drag, gravity, mission time, workspace size and fuel weight (or start from the Rocket, UAV or USV preset). The classical planners re-solve instantly on the plant your numbers define, and one click trains a network on that plant. What OTiS takes as input is physics plus an objective, not a dataset: the simulator generates and grades its own practice missions.

- **[BENCHMARK.md](BENCHMARK.md)**: results against the industry status quo, with numerical deltas and cross-validation.
- **[SOURCES.md](SOURCES.md)**: every source used, and where the training data comes from.
- **[RESEARCH.md](RESEARCH.md)**: short talking points for judges.

## Headline numbers (Rocket preset, 5 × 1,000 unseen missions, ~13 M-parameter network)

| | OTiS | Convex optimization (status quo) | Apollo E-guidance |
| --- | --- | --- | --- |
| Mean objective J (ratio-of-means ΔJ vs optimum +0.0018%) | 0.04658 | 0.04658 | 0.04686 |
| Mean per-mission gap to provable optimum | **0.0184%** (95% CI 0.0145–0.0238%) | ~0% (it *is* the optimizer) | 0.62% |
| p99 per-mission gap | 0.323% | ~0% | — |
| p99 terminal miss | 0.0329 m | 0.0325 m | 0.00001 m |
| One plan | ~1.0 ms (CPU) / 1.3 ms (GPU) | ~0.96 ms | — |
| Plans/sec, batched on GPU | ~75,000 | ~1,040 (one at a time) | — |

The full tables, including Δ miss, Δ fuel, Δ Δv, the out-of-distribution stress test and every cross-check, are in BENCHMARK.md. Regenerate them with `python main.py --benchmark`.

## Setup on this PC (RTX 5070 Ti, Windows 11)

JAX only uses NVIDIA GPUs on Linux, so the GPU path runs in **WSL2 Ubuntu**. Windows-native Python still works, but on CPU only.

One-time setup inside WSL (already done on this machine; the environment is `~/nh-venv`):

```bash
python3 -m venv --without-pip ~/nh-venv
curl -sSL https://bootstrap.pypa.io/get-pip.py | ~/nh-venv/bin/python
~/nh-venv/bin/pip install -r /mnt/c/Users/annac/Desktop/Building/OTiS/requirements.txt
```

## Run

Start the interactive simulator (open http://localhost:8765 in your Windows browser):

```bash
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/annac/Desktop/Building/OTiS/otis && ~/nh-venv/bin/python main.py"
```

Train headless and write the checkpoint used by the UI and the benchmark (about 3 minutes on the 5070 Ti):

```bash
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/annac/Desktop/Building/OTiS/otis && ~/nh-venv/bin/python main.py --no-viz --steps 30000 --lr 4e-4"
```

Benchmark against the industry baselines and rewrite BENCHMARK.md:

```bash
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/annac/Desktop/Building/OTiS/otis && ~/nh-venv/bin/python main.py --benchmark"
```

Self-checks (physics, gradients, short train, checkpoint round-trip):

```bash
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/annac/Desktop/Building/OTiS/otis && MPLBACKEND=Agg ~/nh-venv/bin/python main.py --check"
```

## The UI

- **Simulator.** Pick a vehicle (Rocket, UAV or USV) from the bar under the logo, which stays visible on the Training tab; then set its parameters in the left panel. Type the start, velocity and goal, or drag them on the map; double-click to drop a goal. Every edit re-plans immediately, and miss, fuel and solve time sit side by side.
  - **What these numbers mean** shows the time step, max acceleration, thrust-to-weight, top speed, reach in the mission time and the stiffness κ. It warns when the thrust cannot hold the vehicle up, when goals are out of reach, or when the problem will be slow to learn.
  - **Network** says whether a network trained on exactly these physics exists. If not, **Train on these parameters** trains one.
  - **Reset** restores the preset; **Export** / **Import** save and load a mission as JSON. Edits are remembered per vehicle in your browser.
  - Use the play bar to fly the mission; press `R` for a random mission and `Space` to play.
- **Training.** Shows the plant being trained (taken from the Simulator's parameters). Pick a model size (0.6 M to 51 M parameters) and the number of steps, then press Train. Watch the plan go from a large miss at step 0 to a clean burn, with the loss curve on a log scale. Pause, stop, or replay the whole evolution with the scrubber. The **Training log** table below fills in live with one row every 100, 250, 500 or 1,000 steps (time, train and validation loss, change in validation loss, mean miss, fuel, gradient norm, learning rate) and exports to CSV. A run on the Rocket preset saves to `checkpoints/ui_last_run.pkl` and never overwrites the benchmark checkpoint; runs on other parameters save to `checkpoints/models/<vehicle>-<id>.pkl` and reload automatically next time.
- **About.** How OTiS works: the training loop, the physics, the network, what it is compared with, and its limits.

**Benchmark for slides.** The benchmark is not on the website. Every `python main.py --benchmark` run also writes `exports/benchmark.html`, a standalone page (no JavaScript, numbers as plain text) to screenshot or edit for slides. `python main.py --export-benchmark` rebuilds it from the last run without re-benchmarking; re-exporting overwrites the file, so copy it before editing.

The interface uses a charcoal background (`#333333`), a gold accent (`#f0bc5a`) and square boxes with thin outlines and no shadows, inspired by Wolfram|Alpha, with Newsreader titles and Source Sans 3 for UI type. It is plain HTML, CSS and JS in `otis/web/`, served by Python's standard library, so there is no build step. The same files also live in `docs/` for a static GitHub Pages demo (classical planners only; the neural rocket planner needs the local Python server).

## Physics and parameters

Every plan is 50 commands spread evenly over the mission time (time step = mission time ÷ 50), integrated with RK4. Units are SI.

**Thrust plants (Rocket, UAV, USV).** State `[x, y, vx, vy]`, throttle `[ux, uy]` in `[-1, 1]` per axis:

    v̇ = (F_max / m)·u − (c / m)·v − [0, g]

The plant stays linear for any parameter choice, so the convex program and its exact KKT optimum apply everywhere, and the network can be trained on it through the differentiable JAX simulator. A network is valid only for the physics and objective it was trained on (thrust/mass, gravity, drag/mass, mission time, λ); OTiS matches networks to parameters automatically.

| Preset | Key parameters | λ |
| --- | --- | --- |
| Rocket | 1 kg, 1 N, vacuum, 4 s, ±2 m (the published network's plant) | 0.01 |
| UAV | 1.5 kg, 30 N, drag 0.3 N·s/m, g = 9.81 m/s², 4 s, ±10 m | 0.1 |
| USV | 20 kg, 40 N, drag 15 N·s/m, top-down (no gravity), 25 s, ±25 m | 0.1 |

All presets, limits and help text live in one file, [`otis/web/vehicles.json`](otis/web/vehicles.json), read by both Python and the browser.

**Stiffness κ.** κ = 1 + ‖b‖²/λ is the condition number of the planning problem, where b is how much each command moves the endpoint. It predicts how fast the network can learn the fuel/accuracy trade-off. Measured at 3,000 training steps: Rocket (κ ≈ 170) reaches a 3.6% mean gap to the optimum; the UAV at λ = 0.01 (κ ≈ 3.9×10⁴) is at 495%, at λ = 0.1 (κ ≈ 3.9×10³) 4.2%, and at λ = 1 0.02%. High thrust-to-weight with a small λ is the hard case; raise λ or train longer.

## Files

| File | Role |
| --- | --- |
| `simulator.py` | RK4 step for any thrust plant `(F/m, g, c/m)`, `lax.scan` rollout, closed-form terminal map |
| `network.py` | Residual MLP (pre-LayerNorm, GELU), physics-informed input features, tanh-bounded thrust |
| `loss.py` | J = ‖p_T − goal‖² + λ‖U‖², batched with `vmap` |
| `train.py` | AdamW + warmup-cosine + clipping, pause/stop hooks, best checkpoint |
| `baselines.py` | Convex guidance (CVXPY + Clarabel/OSQP), exact KKT optimum, Apollo E-guidance |
| `vehicles.py` | Parameters → plant spec, derived readouts, classical planners and scoring for every vehicle |
| `web/vehicles.json` | Presets, editable parameters, limits and help text (shared by Python and the UI) |
| `evaluate.py` | Benchmark, deltas, k-fold generalization, cross-validation, writes BENCHMARK.md |
| `export_benchmark.py` | Writes `exports/benchmark.html`, the standalone editable benchmark page for slides |
| `server.py` + `web/` | Interactive UI; keeps one trained network per plant and trains new ones on request |
| `main.py` | CLI entry point |
| `visualize.py` | Legacy Matplotlib window (`--legacy-window`) and the headless report figure |

## Useful flags

These train and benchmark the Rocket preset from the command line. For other parameters, use the UI.

```text
--size {small,medium,large,xl}   network preset (default large, ~13M params)
--steps 6000                     optimizer updates
--batch-size 1024                missions per update
--lr 6e-4                        peak learning rate
--lambda-fuel 0.01               fuel weight
--velocity-limit 0.5             initial-velocity range (0 = start at rest)
--position-limit 2               half-width of the start/goal square
--no-viz | --benchmark | --export-benchmark | --check | --write-benchmark | --legacy-window
--port 8765  --autotrain
```
