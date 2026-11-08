"""Export the benchmark as one standalone, hand-editable HTML file.

    python main.py --export-benchmark            # -> exports/benchmark.html
    python export_benchmark.py [results.json] [out.html]

The page is plain HTML and CSS with every number written in as text: no
JavaScript, no server, no external files except the Google Fonts link (it
falls back to system fonts offline). Open it in a browser to screenshot
sections for slides, or edit the text, colors (the :root block) and layout
directly. Re-running the export overwrites the file, so copy it before
editing if you want to keep your changes.
"""

from __future__ import annotations

import html
import json
import math
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_IN = PACKAGE_ROOT / "outputs" / "benchmark_results.json"
DEFAULT_OUT = PACKAGE_ROOT.parent / "exports" / "benchmark.html"


def sig3(v):
    if v is None or not math.isfinite(v):
        return "—"
    if abs(v) >= 1000:
        return f"{round(v):,}"
    s = f"{v:.3g}"
    if "e" in s:  # expand small numbers instead of exponent notation
        s = f"{v:.{max(0, 2 - int(math.floor(math.log10(abs(v)))))}f}"
    return s


def signed_pct(v, d=2):
    return f"{'+' if v >= 0 else '−'}{abs(v):.{d}f}%"


def delta_cell(v, d, pct=False):
    cls = "good" if v < 0 else "bad" if v > 0 else ""
    sign = "+" if v >= 0 else "−"
    if pct:
        txt = f"{sign}{abs(v):.3f}%"
    elif abs(v) < 1e-2:
        txt = f"{sign}{abs(v):.2e}"
    else:
        txt = f"{sign}{abs(v):.{d}f}"
    return f'<td class="{cls}">{txt}</td>'


CSS = """
:root {
  /* Edit these to restyle the whole page. */
  --bg: #333333; --panel: #333333; --line: #5e5e5e; --sep: #474747;
  --text: #ececec; --text-2: #b4b4b4; --text-3: #8c8c8c;
  --accent: #f0bc5a; --green: #52c08a; --red: #e65b55; --coral: #ee7f5c; --purple: #a898f2;
  --serif: "Newsreader", "Iowan Old Style", Palatino, serif;
  --sans: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--text); font-family: var(--sans); -webkit-font-smoothing: antialiased; }
main { max-width: 1600px; margin: 0 auto; padding: 48px 32px 64px; }
header.page { text-align: center; margin-bottom: 36px; }
header.page h1 { font-family: var(--serif); font-weight: 500; font-size: 56px; letter-spacing: -0.02em; margin: 0 0 12px; }
header.page p { font-family: var(--serif); font-style: italic; font-size: 22px; color: var(--text-2); max-width: 900px; margin: 0 auto; line-height: 1.45; }
section { border: 1px solid var(--line); background: var(--panel); padding: 24px 28px; margin-bottom: 22px; }
section > h2 { margin: 0 0 16px; font-size: 24px; font-weight: 600; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 22px; }
.tile { border: 1px solid var(--line); padding: 26px 26px 22px; }
.t-lab { font-size: 16px; font-weight: 600; color: var(--text-2); }
.t-val { font-size: 50px; font-weight: 700; letter-spacing: -0.03em; line-height: 1.1; margin: 10px 0 6px; font-variant-numeric: tabular-nums; }
.t-sub { font-size: 15.5px; color: var(--text-2); line-height: 1.4; }
.c-gold { color: var(--accent); } .c-green { color: var(--green); } .c-purple { color: var(--purple); } .c-coral { color: var(--coral); }
table { width: 100%; border-collapse: collapse; font-size: 16px; }
th { text-align: right; font-size: 13px; font-weight: 600; color: var(--text-2); padding: 9px 12px; border-bottom: 1px solid var(--sep); white-space: nowrap; }
td { text-align: right; padding: 12px; border-bottom: 1px solid var(--sep); font-family: var(--mono); font-size: 15px; font-variant-numeric: tabular-nums; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; font-family: var(--sans); font-size: 16px; }
td:first-child { font-weight: 500; }
tr:last-child td { border-bottom: 0; }
td.good { color: var(--green); } td.bad { color: var(--red); }
dl.defs { margin: 0; border-top: 1px solid var(--sep); }
dl.defs > div { display: grid; grid-template-columns: 220px 1fr; gap: 16px; padding: 11px 0; border-bottom: 1px solid var(--sep); }
dl.defs > div:last-child { border-bottom: 0; }
dt { color: var(--accent); font-weight: 600; font-size: 16px; }
dd { margin: 0; font-size: 16.5px; line-height: 1.45; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 24px; margin-top: 14px; font-size: 14.5px; color: var(--text-2); }
.legend i { display: inline-block; width: 12px; height: 12px; margin-right: 7px; vertical-align: -1px; }
.legend .g { background: var(--green); } .legend .r { background: var(--red); }
.legend .note { color: var(--text-3); }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
.two > section { margin-bottom: 22px; }
ul.checks { list-style: none; margin: 0; padding: 0; }
ul.checks li { display: flex; gap: 14px; padding: 12px 0; border-bottom: 1px solid var(--sep); }
ul.checks li:last-child { border-bottom: 0; }
.ok { width: 26px; height: 26px; background: var(--green); color: #2b2b2b; display: grid; place-items: center; flex: none; font-weight: 700; }
.c-t { font-weight: 500; font-size: 16px; }
.c-v { font-family: var(--mono); font-size: 14px; color: var(--text-2); margin-top: 2px; }
.bars { display: flex; flex-direction: column; gap: 17px; }
.bar-lab { display: flex; justify-content: space-between; font-size: 16px; margin-bottom: 6px; }
.bar-lab span:last-child { font-family: var(--mono); font-size: 14px; color: var(--text-2); }
.bar-track { height: 12px; background: #2a2a2a; }
.bar-fill { height: 100%; }
small, .foot { color: var(--text-3); font-size: 14px; }
.foot { text-align: center; margin-top: 10px; }
.tw { overflow-x: auto; }
@media (max-width: 1100px) { .tiles { grid-template-columns: repeat(2, 1fr); } .two { grid-template-columns: 1fr; } }
@media (max-width: 700px) {
  main { padding: 28px 16px 40px; }
  header.page h1 { font-size: 38px; } header.page p { font-size: 18px; }
  .tiles { grid-template-columns: 1fr; }
  section { padding: 18px 16px; }
  dl.defs > div { grid-template-columns: 1fr; gap: 4px; }
}
"""


def render(r: dict) -> str:
    M, D, T, CV = r["methods"], r["deltas"], r["timing_ms_per_problem"], r["cross_validation"]
    G = CV["generalization"]
    P = r.get("problem", {})
    n_test = P.get("test_missions", G["folds"] * G["problems_per_fold"])
    eg = D["neural_minus_e_guidance"]
    dv = eg["delta_v"]["pct"]
    e = html.escape

    tiles = [
        ("c-gold", "Mean per-mission gap", f"{sig3(G['gap_pct_mean'])}%",
         f"95% CI {sig3(G['gap_pct_bootstrap_95ci'][0])}–{sig3(G['gap_pct_bootstrap_95ci'][1])}% vs exact optimum"),
        ("c-green", f"Batched throughput vs convex · batch {G['problems_per_fold']:,}", f"{r['speedup_neural_vs_convex']['batched']:.0f}×",
         f"single plan: {T['neural_single_best_latency']:.2f} ms vs {T['convex_clarabel_median']:.2f} ms solver"),
        ("c-purple", "p99 optimality gap", f"{sig3(G.get('gap_pct_p99'))}%",
         f"99% of {n_test:,} test missions are within this of the exact optimum"),
        ("c-coral", "Control effort vs Apollo E-guidance", signed_pct(eg["fuel"]["pct"]),
         f"for {M['neural']['miss']['mean']:.4f} m terminal miss · Apollo steers for zero position miss and uses "
         f"{abs(dv):.2f}% {'less' if dv >= 0 else 'more'} Δv"),
    ]
    tiles_html = "\n".join(
        f'    <div class="tile"><div class="t-lab">{e(l)}</div><div class="t-val {c}">{e(v)}</div><div class="t-sub">{e(s)}</div></div>'
        for c, l, v, s in tiles)

    dt, N, lam = P.get("dt"), P.get("n_steps"), P.get("lambda_fuel")
    horizon = P.get("horizon", (N or 0) * (dt or 0))
    defs = [
        ("Dynamics", "Planar point mass (double integrator), state [x, y, v<sub>x</sub>, v<sub>y</sub>]: v̇ = (F/m)·u with F/m = 1 m/s², no gravity or drag. Thrust is held constant over each step and integrated with RK4."),
        ("Time step", f"dt = {dt} s"),
        ("Horizon", f"N = {N} steps (T = {horizon:.2f} s)"),
        ("Objective", f"J = ‖p<sub>N</sub> − goal‖² + λ·Σ<sub>k</sub>‖u<sub>k</sub>‖², λ = {lam}; |u<sub>x</sub>|, |u<sub>y</sub>| ≤ {P.get('max_thrust', 1)} per step"),
        ("Terminal condition", "Soft penalty on the terminal position; terminal velocity is free (interception, not rendezvous or landing). OTiS, convex and the KKT optimum trade a small miss for less control effort. E-guidance steers for zero terminal position error when the goal is reachable within the thrust limit, and misses when it is not."),
        ("Test missions", f"{n_test:,} ({G['folds']} disjoint folds × {G['problems_per_fold']:,}), never seen in training"),
        ("Sampling", f"Start and goal positions uniform in [−{P.get('position_limit')}, {P.get('position_limit')}]² m; initial velocity uniform in [−{P.get('velocity_limit')}, {P.get('velocity_limit')}]² m/s; drawn from random-number streams separate from training's"),
    ]
    defs_html = "\n".join(f"      <div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in defs)

    names = {"neural": "OTiS", "convex": "Convex optimization (Clarabel)", "optimal": "Exact optimum (KKT)", "e_guidance": "Apollo E-guidance"}
    h2h = "\n".join(
        f"      <tr><td>{n}</td><td>{M[k]['loss']['mean']:.5f}</td><td>{M[k]['miss']['mean']:.4f}</td>"
        f"<td>{M[k]['miss'].get('p99', float('nan')):.4f}</td><td>{M[k]['fuel']['mean']:.3f}</td><td>{M[k]['delta_v']['mean']:.3f}</td></tr>"
        for k, n in names.items())

    labs = {"neural_minus_convex": "Network − convex optimization", "neural_minus_optimal": "Network − exact optimum",
            "neural_minus_e_guidance": "Network − E-guidance", "e_guidance_minus_optimal": "E-guidance − exact optimum"}
    deltas = "\n".join(
        f"      <tr><td>{l}</td>{delta_cell(D[k]['loss']['abs'], 5)}{delta_cell(D[k]['loss']['pct'], 2, True)}"
        f"{delta_cell(D[k]['miss']['abs'], 4)}{delta_cell(D[k]['fuel']['abs'], 3)}{delta_cell(D[k]['fuel']['pct'], 2, True)}"
        f"{delta_cell(D[k]['delta_v']['pct'], 2, True)}</tr>"
        for k, l in labs.items() if k in D)

    KC = CV.get("kkt_consistency")
    checks = []
    if KC:
        checks.append(("KKT referee = training problem",
                       f"same dt {KC['kkt_dt']} s and N {KC['kkt_n_steps']}; KKT terminal map vs RK4 {KC['max_abs_terminal_pos_kkt_map_vs_rk4']:.1e} m on {KC['cases']} plans"
                       if KC["consistent"] else "MISMATCH"))
    checks += [
        ("RK4 vs closed-form physics", f"max error {CV['integrator']['max_abs_rk4_vs_closed_form']:.1e} over {CV['integrator']['cases']} random plans"),
        ("RK4 vs SciPy DOP853", f"max error {CV['integrator']['max_abs_rk4_vs_scipy_dop853']:.1e} (independent 8th-order integrator)"),
        ("Autodiff vs finite differences", f"max error {CV['gradient']['max_abs_autodiff_vs_finite_difference']:.1e} over {CV['gradient']['checks']} gradient entries"),
        ("Clarabel vs exact optimum", f"max ΔJ {CV['solvers']['clarabel']['max_abs_loss_diff_vs_exact']:.1e} on {CV['solvers']['cases']} problems"),
        ("OSQP vs exact optimum", f"max ΔJ {CV['solvers']['osqp']['max_abs_loss_diff_vs_exact']:.1e} (second, independent solver)"),
        (f"{G['folds']}-fold generalization", "mean per-mission gap " + " · ".join(f"{sig3(v)}%" for v in G["gap_pct_mean_per_fold"])),
        ("Out-of-distribution stress", f"2× velocity range: mean per-mission gap {sig3(r['stress_out_of_distribution']['neural_gap_pct_mean'])}%"),
    ]
    checks_html = "\n".join(f'      <li><span class="ok">✓</span><div><div class="c-t">{e(t)}</div><div class="c-v">{e(v)}</div></div></li>' for t, v in checks)

    bars = [
        ("Network · single plan (GPU)", T["neural_single_gpu_latency"], "var(--accent)"),
        ("Network · single plan (CPU)", T["neural_single_cpu_latency"], "var(--accent)"),
        (f"Network · batched ({G['problems_per_fold']:,} per batch)", T["neural_batched_gpu"], "var(--accent)"),
        ("Convex optimization (Clarabel)", T["convex_clarabel_median"], "var(--green)"),
        ("E-guidance · batched", T["e_guidance_batched"], "var(--coral)"),
    ]
    hi = math.log10(max(b[1] for b in bars)) + 0.3
    lo = math.log10(min(b[1] for b in bars)) - 0.3
    bars_html = "\n".join(
        f'      <div><div class="bar-lab"><span>{e(l)}</span><span>{(f"{v:.1e}" if v < 0.01 else f"{v:.3f}")} ms</span></div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{100 * (math.log10(v) - lo) / (hi - lo):.1f}%;background:{c}"></div></div></div>'
        for l, v, c in bars)

    model = r.get("model", {})
    foot = (f"{model.get('size')} model · {model.get('n_params', 0):,} parameters · {G['folds']} × {G['problems_per_fold']:,} unseen missions · "
            f"benchmark run {r.get('generated')} on {r.get('environment', {}).get('device')} · exported {time.strftime('%Y-%m-%d %H:%M')}")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OTiS benchmark</title>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,300..500&family=Source+Sans+3:wght@400..700&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<!-- Generated by otis/export_benchmark.py from otis/outputs/benchmark_results.json.
     Every number below is plain text: edit freely. Re-exporting overwrites this file. -->
<main>
  <header class="page">
    <h1>Measured against the status quo.</h1>
    <p>Thousands of unseen missions, scored by the same simulator. The convex solver is how flight systems solve this class of problem. The exact optimum is the referee.</p>
  </header>

  <!-- ===== Headline tiles ===== -->
  <div class="tiles">
{tiles_html}
  </div>

  <!-- ===== Problem definition ===== -->
  <section>
    <h2>Problem definition</h2>
    <dl class="defs">
{defs_html}
    </dl>
  </section>

  <!-- ===== Head-to-head ===== -->
  <section>
    <h2>Head-to-head</h2>
    <div class="tw"><table>
      <thead><tr><th>Method</th><th>Objective J</th><th>Miss (m)</th><th>p99 miss (m)</th><th>Control effort Σu²</th><th>Δv (m/s)</th></tr></thead>
      <tbody>
{h2h}
      </tbody>
    </table></div>
  </section>

  <!-- ===== Numerical deltas ===== -->
  <section>
    <h2>Numerical deltas</h2>
    <div class="tw"><table>
      <thead><tr><th>Comparison</th><th>ΔJ</th><th>ΔJ % (of mean)</th><th>Δ miss (m)</th><th>Δ control effort</th><th>Δ control effort %</th><th>Δ Δv %</th></tr></thead>
      <tbody>
{deltas}
      </tbody>
    </table></div>
    <div class="legend"><span><i class="g"></i>green = better for the first method</span><span><i class="r"></i>red = worse for the first method</span><span class="note">ΔJ % is a ratio of means: (mean J<sub>a</sub> − mean J<sub>b</sub>) / mean J<sub>b</sub>. Lower is better in every column.</span></div>
  </section>

  <div class="two">
    <!-- ===== Cross-validation ===== -->
    <section>
      <h2>Cross-validation</h2>
      <ul class="checks">
{checks_html}
      </ul>
    </section>
    <!-- ===== Compute per plan ===== -->
    <section>
      <h2>Compute per plan</h2>
      <div class="bars">
{bars_html}
      </div>
      <p><small>Log scale. Measured on {e(str(r.get('environment', {}).get('device')))}.</small></p>
    </section>
  </div>

  <p class="foot">{e(foot)}</p>
</main>
</body>
</html>
"""


def export(results_path=DEFAULT_IN, out_path=DEFAULT_OUT) -> Path:
    results_path, out_path = Path(results_path), Path(out_path)
    if not results_path.exists():
        raise SystemExit(f"No benchmark results at {results_path}. Run: python main.py --benchmark")
    r = json.loads(results_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(r), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    export(*(sys.argv[1:3]))
