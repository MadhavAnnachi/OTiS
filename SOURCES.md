# OTiS — Sources and Training Data

This file tracks every source the project relies on: where the training data comes from, the industry baselines the results are compared against, the theory and numerical methods behind the simulator, the software used, and the design references for the UI.

Reference numbers are stable IDs, grouped by topic, so they are not in order on the page. All references were checked on 2026-11-07 against publisher pages, Crossref, arXiv, proceedings sites, or official repositories. Section 1 also covers the references inherited from the earlier Cursor version of the project.

Related files:
- [BENCHMARK.md](BENCHMARK.md): the numerical results and deltas.
- `otis/outputs/benchmark_results.json`: the raw benchmark numbers.
- `otis/data/benchmark_problems.json`: the fixed test set, with the optimal objective for each case.

---

## 1. Training data: where it comes from

**There is no external dataset.** OTiS is trained by self-supervision through a physics simulator. Nothing is downloaded, labeled, or imitated.

| Item | What it is |
| --- | --- |
| Generator | `otis/train.py::sample_problems`, called once per training step |
| Distribution | Start position uniform in [−2, 2]², start velocity uniform in [−0.5, 0.5]², goal uniform in [−2, 2]² (`config.py`) |
| Batch | 1,024 fresh problems per step. 30,000 steps = **30.7 million** distinct training missions |
| Labels | **None.** The loss J = ‖p_T − goal‖² + λ Σ‖u‖² is computed by flying the network's plan through the RK4 simulator. Gradients come from JAX autodiff through the simulator (exact, verified against finite differences) |
| Randomness | `jax.random.PRNGKey(seed)`, split every step and never reused. Default seed 0, so any run is reproducible |
| Physics "ground truth" | Newton's second law for a unit-mass body, F = m·a, with piecewise-constant thrust. The closed form is in `simulator.py::terminal_map`. RK4 reproduces it to 1e-6 in float32, and SciPy's DOP853 reproduces it to 4e-15 (see BENCHMARK.md) |
| Validation set | 2,048 problems from a separate PRNG split, used only to pick the best checkpoint |
| Test set | 5 folds × 1,000 problems from `fold_in(PRNGKey(0), 10000 + i)`, streams the trainer never draws from. Fold 0 is saved in `data/benchmark_problems.json` with the provably optimal objective for each case |

**Why there is no public dataset.** No public labeled corpus matches this exact problem: a 2D unit-mass double integrator, a 4 s horizon, a box thrust limit, and a quadratic fuel penalty. Prior neural-guidance work (refs 30–31) generated its own training data by solving thousands of optimal-control problems offline and then imitating them. OTiS skips that step. It needs no solver to make labels because the simulator itself supplies the gradient. The solver is used only afterwards, to grade the result.

**What changed from the Cursor version.** The earlier version sampled starts at rest (velocity = 0) on an 80 × 0.05 s horizon. It now samples non-zero initial velocities and uses 50 × 0.08 s steps, which matches the "50 timesteps" in the project pitch while keeping the 4 s horizon and the 8.0-unit reachable range.

---

## 2. Industry baseline: convex optimization guidance

The status quo for fuel-optimal rocket guidance is to transcribe the problem into a convex program and solve it on board with an interior-point solver. Our "convex" baseline does exactly that for the same objective, using CVXPY + Clarabel.

6. **Açıkmeşe, B., & Ploen, S. R.** "Convex Programming Approach to Powered Descent Guidance for Mars Landing." *Journal of Guidance, Control, and Dynamics*, 30(5), 1353–1366, 2007. https://doi.org/10.2514/1.27553
   This is the founding lossless-convexification paper behind G-FOLD and the origin of our per-problem convex direct-transcription baseline.

7. **Blackmore, L., Açıkmeşe, B., & Scharf, D. P.** "Minimum-Landing-Error Powered-Descent Guidance for Mars Landing Using Convex Optimization." *JGCD*, 33(4), 1161–1171, 2010. https://doi.org/10.2514/1.47202
   It poses minimum terminal-position error as a convex program, the same terminal-miss term we use.

8. **Açıkmeşe, B., Carson, J. M., & Blackmore, L.** "Lossless Convexification of Nonconvex Control Bound and Pointing Constraints of the Soft Landing Optimal Control Problem." *IEEE Transactions on Control Systems Technology*, 21(6), 2104–2113, 2013. https://doi.org/10.1109/TCST.2012.2237346
   It justifies treating thrust bounds convexly. Our box |u_i| ≤ 1 is already convex, so the QP baseline is exact.

9. **Malyuta, D., Reynolds, T. P., Szmuk, M., Lew, T., Bonalli, R., Pavone, M., & Açıkmeşe, B.** "Convex Optimization for Trajectory Generation: A Tutorial on Generating Dynamically Feasible Trajectories Reliably and Efficiently." *IEEE Control Systems Magazine*, 42(5), 40–113, 2022. https://doi.org/10.1109/MCS.2022.3187542 (arXiv:2106.09125)
   A modern tutorial on convex trajectory generation, used to frame the baseline.

10. **JPL / NASA.** "JPL, Masten Testing New Precision Landing Software" (2013). https://www.jpl.nasa.gov/news/jpl-masten-testing-new-precision-landing-software/ — and NASA, "Masten's Xombie Tests JPL's G-FOLD Precision Landing Software." https://www.nasa.gov/centers-and-facilities/armstrong/mastens-xombie-tests-jpls-g-fold-precision-landing-software
    The public record that G-FOLD flew on Masten's Xombie in 2013. It establishes convex guidance as flight-proven.

11. **Blackmore, L.** "Autonomous Precision Landing of Space Rockets." *Frontiers of Engineering: Reports on Leading-Edge Engineering from the 2016 Symposium*, National Academies Press, 2017, pp. 33–42 (also in NAE *The Bridge*, Winter 2016). https://www.nationalacademies.org/read/23659/chapter/10
    A public SpaceX source stating that Falcon 9 landing uses CVXGEN-generated onboard convex optimization. This is why we call convex optimization the industry baseline.

## 3. Industry baseline: closed-form guidance laws (Apollo heritage)

Our "E-guidance" baseline is the energy-optimal ZEM feedback law a = 3·ZEM / t_go², with ZEM = r_goal − (r + v·t_go). It runs in closed loop at every step and is clipped to the thrust box.

12. **Cherry, G. W.** "A General, Explicit, Optimizing Guidance Law for Rocket-Propelled Spaceflight." AIAA Paper 64-638, 1964. https://doi.org/10.2514/6.1964-638
    The origin of Apollo E-guidance, the family our closed-form baseline belongs to. Metadata was confirmed via Crossref; the full text was not accessed.

13. **Klumpp, A. R.** "Apollo Lunar Descent Guidance." *Automatica*, 10(2), 133–146, 1974. https://doi.org/10.1016/0005-1098(74)90019-3 (free copy: https://ntrs.nasa.gov/citations/19740044219)
    The flown Apollo LM descent guidance, which is the flight heritage of the closed-form law.

14. **D'Souza, C.** "An Optimal Guidance Law for Planetary Landing." AIAA GNC Conference, 1997, AIAA Paper 97-3709. https://doi.org/10.2514/6.1997-3709
    Shows that E-guidance-type laws minimize integrated squared acceleration, the continuous analogue of our λΣu² term.

15. **Ebrahimi, B., Bahrami, M., & Roshanian, J.** "Optimal Sliding-Mode Guidance with Terminal Velocity Constraint for Fixed-Interval Propulsive Maneuvers." *Acta Astronautica*, 62(10–11), 556–562, 2008. https://doi.org/10.1016/j.actaastro.2008.02.002
    An original ZEM/ZEV formulation. Do not confuse it with the authors' 2006 IEEE CCA paper, which has a different title.

16. **Guo, Y., Hawkins, M., & Wie, B.** "Applications of Generalized Zero-Effort-Miss/Zero-Effort-Velocity Feedback Guidance Algorithm." *JGCD*, 36(3), 810–820, 2013. https://doi.org/10.2514/1.58099
    The standard generalized ZEM/ZEV reference. The law is optimal in uniform gravity, so this baseline is a true optimum of its own problem, not a strawman.

## 4. Optimal control theory

17. **Lawden, D. F.** *Optimal Trajectories for Space Navigation*. Butterworths, London, 1963. No DOI. Confirmed via Longuski, Guzmán & Prussing, *Optimal Control with Aerospace Applications*, Springer 2014: https://rd.springer.com/chapter/10.1007/978-1-4614-8945-0_10
    Primer-vector theory, cited in the earlier notes for bang-bang fuel-optimal control.

18. **Bryson, A. E., Jr., & Ho, Y.-C.** *Applied Optimal Control*. Hemisphere, 1975; reissued by Taylor & Francis. https://doi.org/10.1201/9781315137667 (the DOI is for the reissue)
    The textbook minimum-energy control of linear systems. Our exact KKT referee solves the discrete version of this problem.

19. **Betts, J. T.** "Survey of Numerical Methods for Trajectory Optimization." *JGCD*, 21(2), 193–207, 1998. https://doi.org/10.2514/2.4231
    The taxonomy of direct and indirect methods. Our QP baseline is direct transcription in Betts's sense.

## 5. Numerical integration and cross-validation

20. **Dormand, J. R., & Prince, P. J.** "A Family of Embedded Runge-Kutta Formulae." *Journal of Computational and Applied Mathematics*, 6(1), 19–26, 1980. https://doi.org/10.1016/0771-050X(80)90013-3
    The origin of the Dormand–Prince pairs. This paper gives RK5(4); DOP853 comes from ref 21.

21. **Hairer, E., Nørsett, S. P., & Wanner, G.** *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed., Springer, 1993. https://doi.org/10.1007/978-3-540-78862-1
    The source of DOP853, the independent 8th-order integrator we use to cross-validate our RK4.

22. **Virtanen, P., et al. (SciPy 1.0 Contributors).** "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python." *Nature Methods*, 17, 261–272, 2020. https://doi.org/10.1038/s41592-019-0686-2
    `scipy.integrate.solve_ivp(method="DOP853")`.

## 6. Differentiable simulation and learning (the method)

1. **Amos, B., Jimenez, I., Sacks, J., Boots, B., & Kolter, J. Z.** "Differentiable MPC for End-to-end Planning and Control." *NeurIPS 2018*. https://proceedings.neurips.cc/paper/2018/hash/ba6d843eb4251a4526ce65d1807a9309-Abstract.html (arXiv:1810.13400)
   The closest prior work that backpropagates through an optimal-control solver. We differentiate the rollout directly instead.

2. **de Avila Belbute-Peres, F., Smith, K., Allen, K., Tenenbaum, J., & Kolter, J. Z.** "End-to-End Differentiable Physics for Learning and Control." *NeurIPS 2018*, pp. 7178–7189. https://proceedings.neurips.cc/paper/2018/hash/842424a1d0595b76ec4fa03c46e8d755-Abstract.html
   Analytic gradients through a physics engine for control, the core idea behind our training loop.

3. **Hu, Y., Anderson, L., Li, T.-M., Sun, Q., Carr, N., Ragan-Kelley, J., & Durand, F.** "DiffTaichi: Differentiable Programming for Physical Simulation." *ICLR 2020*. https://arxiv.org/abs/1910.00935
   Controllers trained through differentiable simulators converge fast, which supports our no-RL design.

4. **Freeman, C. D., Frey, E., Raichuk, A., Girgin, S., Mordatch, I., & Bachem, O.** "Brax – A Differentiable Physics Engine for Large Scale Rigid Body Simulation." *NeurIPS 2021 Datasets & Benchmarks Track*. https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/d1f491a404d6854880943e5c3cd9ca25-Abstract-round1.html (arXiv:2106.13281)
   A JAX-native differentiable simulator, the large-scale counterpart to ours.

5. **Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D.** "Neural Ordinary Differential Equations." *NeurIPS 2018*. https://arxiv.org/abs/1806.07366
   The standard reference for differentiating through ODE solvers. We backpropagate through fixed-step RK4 rather than using the adjoint method.

## 7. Prior work on neural-network guidance

30. **Sánchez-Sánchez, C., & Izzo, D.** "Real-Time Optimal Control via Deep Neural Networks: Study on Landing Problems." *JGCD*, 41(5), 1122–1135, 2018. https://doi.org/10.2514/1.G002357 (arXiv:1610.08668)
    The closest prior work. Networks trained by imitating precomputed optimal trajectories. OTiS trains through the simulator instead and needs no optimal-trajectory dataset.

31. **Izzo, D., & Öztürk, E.** "Real-Time Guidance for Low-Thrust Transfers Using Deep Neural Networks." *JGCD*, 44(2), 315–327, 2021. https://doi.org/10.2514/1.G005254 (preprint: arXiv:2002.09063)
    Neural imitation of optimal interplanetary guidance, another supervised contrast to our approach.

## 8. Software

23. **Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Katariya, Y., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q.** "JAX: composable transformations of Python+NumPy programs." 2018. https://github.com/jax-ml/jax (used: 0.12.0.1, CUDA 12 build under WSL2)
24. **Heek, J., Levskaya, A., Oliver, A., Ritter, M., Rondepierre, B., Steiner, A., & van Zee, M.** "Flax: A neural network library and ecosystem for JAX." 2020–. https://github.com/google/flax (used: 0.12.12)
25. **DeepMind et al.** "The DeepMind JAX Ecosystem" (Optax). 2020. https://github.com/google-deepmind/optax (used: 0.2.8; AdamW, warmup-cosine schedule, global-norm clipping)
26. **Kingma, D. P., & Ba, J.** "Adam: A Method for Stochastic Optimization." *ICLR 2015*. https://arxiv.org/abs/1412.6980
27. **Stellato, B., Banjac, G., Goulart, P., Bemporad, A., & Boyd, S.** "OSQP: An Operator Splitting Solver for Quadratic Programs." *Mathematical Programming Computation*, 12(4), 637–672, 2020. https://doi.org/10.1007/s12532-020-00179-2 (used: 1.1.3, as the second cross-check solver)
28. **Diamond, S., & Boyd, S.** "CVXPY: A Python-Embedded Modeling Language for Convex Optimization." *JMLR*, 17(83), 1–5, 2016. https://jmlr.org/papers/v17/15-408.html (used: 1.9.3)
29. **Goulart, P. J., & Chen, Y.** "Clarabel: An Interior-Point Solver for Conic Programs with Quadratic Objectives." arXiv:2405.12762, 2024. https://arxiv.org/abs/2405.12762 (used: 0.11.1; the primary convex baseline solver, in the same interior-point class as flight convex guidance)

Not cited as papers, but used: NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2 (for the headless report figure).

## 9. UI design references

32. **Apple Inc.** *Human Interface Guidelines*. https://developer.apple.com/design/human-interface-guidelines (accessed 2026-11-07). The overall visual and interaction style: grouped inset lists, segmented controls, switches, system colors, light and dark appearance.
33. **Apple Inc.** "Materials." *Human Interface Guidelines*. https://developer.apple.com/design/human-interface-guidelines/materials. The translucent, blurred navigation bar and floating controls.
34. **Apple Inc.** "Fonts" (SF Pro). https://developer.apple.com/fonts/. The UI uses the system font stack: SF Pro on Apple devices, with Inter (Google Fonts, SIL Open Font License) as the fallback on Windows. No Apple font files are bundled.

---

## 10. User-defined plants (editable parameters)

The Simulator now takes your vehicle's parameters (mass, thrust, drag, gravity, mission time, workspace, fuel weight). The thrust plants stay linear, so refs 1–11 and 17–19 still apply to every parameter choice. The car and the conditioning readout rely on the references below. These citations come from prior knowledge and were not re-fetched in this session; verify them before submission.

35. **Coulter, R. C.** "Implementation of the Pure Pursuit Path Tracking Algorithm." Carnegie Mellon University Robotics Institute, Technical Report CMU-RI-TR-92-01, 1992.
    The steering law δ = atan(2L·sin α / d) behind the Car's planner, with the goal as the look-ahead point.

36. **Rajamani, R.** *Vehicle Dynamics and Control*, 2nd ed., Springer, 2012. https://doi.org/10.1007/978-1-4614-1433-9
    The kinematic bicycle model (no lateral slip) used for the Car plant.

37. **Nocedal, J., & Wright, S. J.** *Numerical Optimization*, 2nd ed., Springer, 2006. https://doi.org/10.1007/978-0-387-40065-5
    Gradient methods slow down in proportion to the condition number. This is the basis for the "Stiffness κ = 1 + ‖b‖²/λ" readout and its warning. Measured in this repo at 3,000 training steps on the UAV preset: 495% gap to the optimum at λ = 0.01 (κ ≈ 3.9×10⁴), 4.2% at λ = 0.1, and 0.02% at λ = 1.

## Notes on the earlier (Cursor) references

The earlier `RESEARCH.md` listed 7 references, now items 1–5, 19 and 23 above. **All seven are real, and their core details were correct.** Refinements:

- Amos et al. and Chen et al.: added the arXiv IDs and full author lists.
- DiffTaichi: added the full title and all seven authors.
- Brax: it was *accepted* to the NeurIPS 2021 Datasets & Benchmarks proceedings (Round 1), not just submitted.
- Betts: added pages 193–207.
- JAX: the repository moved from `google/jax` to `jax-ml/jax`. The official author list has 12 names, including Yash Katariya, whom some older copies omit.
- Lawden (primer-vector theory) was mentioned without a citation. It is now item 17.

No other sources were found in the codebase; the code itself contained no citations.
