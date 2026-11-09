# OTiS

**Optimized Trajectory Simulator.** The project folder is `OTiS`; the Python package is `otis`.

Live demo: https://otis-site-mu.vercel.app/ (a static copy of the interface; see [The hosted demo](#the-hosted-demo) for what runs there).

OTiS plans simulated maneuvers for a rocket, a drone (UAV) or a surface boat (USV). You give it a starting position, a starting velocity and a goal. A neural network reads those six numbers and outputs a complete plan in one forward pass: 50 two-axis thrust commands, each held for an equal slice of the mission time. The plan is flown through a differentiable physics simulator, so training gets the exact gradient of the score with respect to every command and uses it to update the network. There is no dataset of expert plans and no reinforcement learning; the simulator generates and grades its own practice missions.

The score balances two things: the squared distance between the final position and the goal, and control effort (the sum of squared thrust commands). Effort is a fuel proxy, not a model of propellant burned. The planner aims for a fly-through, not a landing: there is no requirement on final velocity. Because the dynamics are linear and the commands are box-bounded, the same problem has a provable optimum, which OTiS computes with a convex solver and an exact KKT solver. The neural plan is an approximation, and every plan is checked against that optimum.

**You define the vehicle.** Enter mass, thrust, drag, mission time, workspace size and fuel weight, or start from the Rocket, UAV or USV preset. The classical planners re-solve instantly on the plant your numbers define, and one click trains a network on that plant.

- **[BENCHMARK.md](BENCHMARK.md)**: results against the industry status quo, with numerical deltas and cross-validation.
- **[SOURCES.md](SOURCES.md)**: every source used, and where the training data comes from.
- **[RESEARCH.md](RESEARCH.md)**: short talking points for judges.

## Headline numbers (Rocket preset, 5 × 1,000 unseen missions, ~13 M-parameter network)

These come from the latest benchmark run (2026-11-07), stored in `otis/outputs/benchmark_results.json` and shown in `exports/benchmark.html`.

| | OTiS | Convex optimization (status quo) | Apollo E-guidance |
| --- | --- | --- | --- |
| Mean objective J | 0.04658 | 0.04658 | 0.04686 |
| Mean objective vs exact optimum (ratio of means) | +0.0018% | ~0% (it *is* the optimizer) | +0.62% |
| Mean per-mission gap to the exact optimum | **0.0184%** (95% CI 0.0145–0.0238%) | ~0% | - |
| p99 per-mission gap | 0.323% | ~0% | - |
| p99 terminal miss | 0.0329 m | 0.0325 m | < 0.00001 m |
| One plan | 1.1 ms (CPU) / 1.3 ms (GPU) | 0.96 ms | - |
| Plans per second, batched on the GPU | ~120,000 (117× convex) | ~1,040 (one at a time) | - |

E-guidance steers for zero position miss, so it spends about 1.2% more control effort than OTiS on these missions. For one plan at a time the convex solver is about as fast as the network; the network's advantage is batch throughput and constant cost per plan.

The full tables, including Δ miss, Δ effort, Δ Δv, the out-of-distribution stress test and every cross-check, are in BENCHMARK.md. Regenerate them with `python main.py --benchmark`. This run was recorded before the simulator switched from RK4 to the exact transition described below, and its generated text still says RK4. The Rocket preset has no drag, where RK4 and the exact transition agree to round-off, so the numbers still apply.

## Setup on this PC (RTX 5070 Ti, Windows 11)

JAX only uses NVIDIA GPUs on Linux, so the GPU path runs in **WSL2 Ubuntu**. Windows-native Python still works, but on CPU only (install `jax` instead of `jax[cuda12]`; see the comment in `requirements.txt`).

One-time setup inside WSL (already done on this machine; the environment is `~/nh-venv`):

```bash
python3 -m venv --without-pip ~/nh-venv
curl -sSL https://bootstrap.pypa.io/get-pip.py | ~/nh-venv/bin/python
~/nh-venv/bin/pip install -r /mnt/c/Users/annac/Desktop/Building/OTiS/requirements.txt
```

## Run

Start the interactive simulator, then open http://localhost:8765 in your Windows browser:

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

Regression tests for the planners, API validation, About copy and the `docs/` mirror (plain asserts, no pytest needed): run `python test_regressions.py` from `otis/`.

## The UI

- **Simulator.** Pick a vehicle (Rocket, UAV or USV) from the bar under the logo, which stays visible on the Training tab; then set its parameters in the left panel. Type the start, velocity and goal, or drag them on the map; double-click to drop a goal. Every edit re-plans immediately, and miss, effort and solve time sit side by side.
  - **What these numbers mean** shows the time step, max acceleration, thrust-to-weight, top speed, reach in the mission time and the stiffness κ. It warns when the thrust cannot hold the vehicle up, when goals are out of reach, or when the problem will be slow to learn.
  - **Network** says whether a network trained on exactly these physics exists. If not, **Train on these parameters** trains one.
  - **Reset** restores the preset; **Export** / **Import** save and load a mission as JSON. Edits are remembered per vehicle in your browser.
  - Use the play bar to fly the mission; press `R` for a random mission and `Space` to play.
- **Training.** Shows the plant being trained (taken from the Simulator's parameters). Pick a model size (0.6 M to 51 M parameters) and the number of steps, then press Train. Watch the plan go from a large miss at step 0 to a clean burn, with the loss curve on a log scale. Pause, stop, or replay the whole evolution with the scrubber. The **Training log** table below fills in live with one row every 100, 250, 500 or 1,000 steps (time, train and validation loss, change in validation loss, mean miss, effort, gradient norm, learning rate) and exports to CSV. A run on the Rocket preset saves to `checkpoints/ui_last_run.pkl` and never overwrites the benchmark checkpoint; runs on other parameters save to `checkpoints/models/<vehicle>-<id>.pkl` and reload automatically next time.
- **About** (the page is titled "How it works"). What OTiS does, what to try, how the network turns a mission into 50 commands, what fly-through and effort mean, and the limits of E-guidance. Equations and implementation detail sit in a collapsed **Technical details** section. The About content is the same locally and on the hosted demo.

**Benchmark for slides.** The benchmark is not on the website. Every `python main.py --benchmark` run also writes `exports/benchmark.html`, a standalone page (no JavaScript, numbers as plain text) to screenshot or edit for slides. `python main.py --export-benchmark` rebuilds it from the last run without re-benchmarking; re-exporting overwrites the file, so copy it before editing.

The interface uses a charcoal background (`#333333`), a gold accent (`#f0bc5a`) and square boxes with thin outlines and no shadows, inspired by Wolfram|Alpha, with Newsreader titles and Source Sans 3 for UI type. It is plain HTML, CSS and JS in `otis/web/`, served by Python's standard library, so there is no build step.

## The hosted demo

Edit the UI only in `otis/web/`. Running `python build_site.py` from `otis/` copies it to `docs/`, an exact mirror, and to `../OTiS-site/`, a separate repository that Vercel deploys to https://otis-site-mu.vercel.app/. (`docs/` is not currently published with GitHub Pages.) The Vercel copy adds `<meta name="otis-deploy" content="static">` and the `otis/web/demo/` folder. With that tag, `app.js` sends its API calls to `web/demo.js` instead of Python, so the hosted site needs no server:

- **Classical planners:** the exact KKT optimum and E-guidance run in the browser from `plant.js`, for any parameters.
- **Network:** each preset vehicle's trained network (small, 584K parameters) runs in the browser from its exported weights. `check_web_demo.py` (or `node check_web_demo.js`) confirms the browser forward pass matches Flax; the current weights agree to within 2×10⁻⁶.
- **Training:** the Training tab is labelled "Simulated training on an NVIDIA RTX 5070 Ti". Pressing Train replays the real GPU run recorded for that preset, at its recorded speed: loss curves, log rows, evolution and the showcase plan.
- **Edited parameters:** if a visitor moves the parameters away from a preset, only the classical planners run; the network and training need the local server.
- **Re-recording:** `python record_web_demo.py --steps 20000` (from `otis/`) re-records the runs. It trains in a temporary folder and needs the GPU, so stop the local server first to free GPU memory. Then run `build_site.py` and commit `OTiS-site` to redeploy.

## Physics and parameters

Every plan is 50 commands spread evenly over the mission time (time step h = mission time ÷ 50). Units are SI.

**Thrust plants (Rocket, UAV, USV).** All three presets are two-dimensional point masses: state `[x, y, vx, vy]`, throttle `[ux, uy]` in `[-1, 1]` per axis, with no attitude, obstacles or detailed aerodynamics:

    dp/dt = v
    dv/dt = (F_max / m)·u − (c / m)·v − [0, g]

**Exact transition.** Python (`simulator.py`, `vehicles.py`) and the browser (`plant.js`) step this model with an exact linear zero-order-hold transition: thrust is held constant over each step and the linear equation is solved in closed form, gravity and drag included. With k = c/m, a = (F_max/m)·u − [0, g] and e = exp(−k·h):

    p_next = p + τ·v + q·a,   v_next = e·v + τ·a,   τ = (1 − e)/k,   q = (h − τ)/k

The code uses `expm1` and a small-drag series so these stay accurate when drag is tiny; at zero drag τ = h and q = h²/2. This replaced the earlier RK4 integrator and its sub-steps. "Exact" refers to this simplified model, up to floating-point precision, not to real flight.

**Stiffness guard.** The old RK4 admission rule is kept as the supported parameter range: settings with drag/mass × time step > 32 (what would have needed more than 64 RK4 sub-steps) are rejected with an explanation. The server, the in-browser planners and the training simulator use the same rule.

The plant stays linear for any parameter choice, so the convex program and its exact KKT optimum apply everywhere, and the network can be trained on it through the differentiable JAX simulator. A network is valid only for the physics and objective it was trained on (thrust/mass, gravity, drag/mass, mission time, λ); OTiS matches networks to parameters automatically.

| Preset | Key parameters | λ |
| --- | --- | --- |
| Rocket | 1 kg, 1 N, vacuum, 4 s, ±2 m (the published network's plant) | 0.01 |
| UAV | 1.5 kg, 30 N, drag 0.3 N·s/m, g = 9.81 m/s², 4 s, ±10 m | 0.1 |
| USV | 20 kg, 40 N, drag 15 N·s/m, top-down (no gravity), 25 s, ±25 m | 0.1 |

All presets, limits and help text live in one file, [`otis/web/vehicles.json`](otis/web/vehicles.json), read by both Python and the browser.

**Objective.** J = ‖p_50 − goal‖² + λ·Σ‖u_k‖². There is no terminal-velocity term, and a miss is penalized rather than forbidden, so even the optimum trades a little accuracy for less effort.

**E-guidance.** The Apollo-heritage baseline is a bounded steering law that keeps aiming at the goal within the thrust limit. It is not a reachability guarantee: saturation and drag compensation can make it miss even reachable goals, and it does not match a final velocity either.

**Stiffness κ.** κ = 1 + ‖b‖²/λ is the condition number of the planning problem, where b is how much each command moves the endpoint. It predicts how fast the network can learn the effort/accuracy trade-off. An earlier measurement on the UAV preset at 3,000 training steps (recorded in SOURCES.md and RESEARCH.md, not part of the current benchmark run): at λ = 0.01 (κ ≈ 3.9×10⁴) the mean gap to the optimum was 495%, at λ = 0.1 4.2%, and at λ = 1 0.02%. High thrust-to-weight with a small λ is the hard case; raise λ or train longer.

## Files

All paths are in `otis/`.

| File | Role |
| --- | --- |
| `config.py` | Shared constants: 50 steps, default time step, model sizes, training and benchmark settings |
| `simulator.py` | Exact zero-order-hold step for any thrust plant `(F/m, g, c/m)`, `lax.scan` rollout, closed-form terminal map |
| `network.py` | Residual MLP (pre-LayerNorm, GELU), physics-informed input features, tanh-bounded thrust |
| `loss.py` | J = ‖p_T − goal‖² + λ‖U‖², batched with `vmap` |
| `train.py` | AdamW + warmup-cosine + clipping, pause/stop hooks, best checkpoint |
| `baselines.py` | Convex guidance (CVXPY + Clarabel/OSQP), exact KKT optimum, Apollo E-guidance |
| `vehicles.py` | Parameters → plant spec, derived readouts, classical planners and scoring for every vehicle |
| `evaluate.py` | Benchmark, deltas, k-fold generalization, cross-validation, writes BENCHMARK.md |
| `export_benchmark.py` | Writes `exports/benchmark.html`, the standalone editable benchmark page for slides |
| `benchmarks.py` | Writes the fixed benchmark missions with their optimal objectives (`--write-benchmark`) |
| `server.py` + `web/` | Interactive UI; keeps one trained network per plant and trains new ones on request |
| `web/plant.js` | In-browser plant, KKT optimum and E-guidance (same physics as Python) |
| `web/demo.js` | Hosted-demo backend: in-browser networks and recorded training replays |
| `web/vehicles.json` | Presets, editable parameters, limits and help text (shared by Python and the UI) |
| `build_site.py` | Copies `web/` to `docs/` and to the `OTiS-site` repository for Vercel |
| `record_web_demo.py` | Trains and records the small preset networks used by the hosted demo (GPU) |
| `check_web_demo.py` / `.js` | Checks that `demo.js` reproduces the Flax network |
| `test_regressions.py` | Regression tests for physics, API validation, copy and mirrors |
| `main.py` | CLI entry point |
| `visualize.py` | Legacy Matplotlib window (`--legacy-window`) and the headless report figure |

## Useful flags

These train and benchmark the Rocket preset from the command line. For other parameters, use the UI.

```text
--size {small,medium,large,xl}   network preset (default large, ~13M params)
--steps 6000                     optimizer updates
--batch-size 1024                missions per update
--lr 6e-4                        peak learning rate
--lambda-fuel 0.01               effort (fuel-proxy) weight
--velocity-limit 0.5             initial-velocity range (0 = start at rest)
--position-limit 2               half-width of the start/goal square
--no-viz | --benchmark | --export-benchmark | --check | --write-benchmark | --legacy-window
--port 8765  --host 127.0.0.1  --autotrain
```


## Contributors

- [@ahanJaiswal](https://github.com/ahanJaiswal) — originated the hackathon idea, led the mathematical and physical reasoning behind the simulator and neural network, and contributed substantially to UI design.
- [@MadhavAnnachi](https://github.com/MadhavAnnachi) — carried out neural network training as part of the team’s division of work.

Development included in-person collaboration on shared machines, so commit authorship does not fully reflect individual contributions. These descriptions highlight the division of work rather than exhaustively listing each person’s contributions.
