"""Regression tests for the issues found in outputs/stress-test/REPORT.md.

    python test_regressions.py          (plain asserts, no pytest needed; numpy only)

Covers the numerical items (stiff drag, long horizons, precision of tiny
controls), the API validation rules and the copy that must not promise
behaviour the objective does not implement.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vehicles as V  # noqa: E402


def exact_point_mass(controls, x0, spec):
    """Independent closed-form propagation of v' = a u - k v - [0, g] (zero-order hold)."""

    k, a, g, dt = spec["drag"], spec["gain"], spec["gravity"], spec["dt"]
    p, v = np.array(x0[:2], float), np.array(x0[2:], float)
    for u in np.clip(np.asarray(controls, float), -1, 1):
        acc = a * u - np.array([0.0, g])
        if k > 1e-12:
            e = math.exp(-k * dt)
            p = p + v * (1 - e) / k + acc / k * (dt - (1 - e) / k)
            v = v * e + acc / k * (1 - e)
        else:
            p = p + v * dt + 0.5 * acc * dt * dt
            v = v + acc * dt
    return np.r_[p, v]


def test_presets_unchanged():
    for vid in ("rocket", "uav", "usv"):
        spec = V.resolve(vid)
        assert spec["substeps"] == 1 and spec["error"] is None, vid


def test_stiff_plants_match_exact_integration():
    rng = np.random.default_rng(0)
    for vid, over in (("usv", {"horizon": 200}), ("uav", {"drag": 100}), ("uav", {"horizon": 200}), ("uav", {"horizon": 600}), ("usv", {}), ("rocket", {})):
        spec = V.resolve(vid, over)
        u = rng.uniform(-1, 1, (V.N, 2))
        x0 = np.array([0.3, -0.2, 0.5, 0.1])
        ours = V.simulate_point_mass(u, x0, spec)[-1]
        ref = exact_point_mass(u, x0, spec)
        scale = max(1.0, np.abs(ref).max())
        assert np.abs(ours - ref).max() / scale < 2e-10, (vid, over, ours, ref)


def test_long_horizon_usv_plan_is_real():
    # Report: KKT controls ~6e-8 "reached" the goal only because RK4 added energy.
    spec = V.resolve("usv", {"horizon": 200})
    out = V.plan_vehicle(spec, [0, 0, 0, 0], [1, 1], methods=["convex"])
    plan = out["plans"]["convex"]
    end = exact_point_mass(plan["controls"], [0, 0, 0, 0], spec)
    assert np.linalg.norm(end[:2] - np.array(plan["path"][-1][:2])) < 1e-5
    assert max(abs(c) for row in plan["controls"] for c in row) > 1e-4  # controls survive serialization


def test_heavy_drag_is_stable():
    spec = V.resolve("uav", {"drag": 100})
    out = V.plan_vehicle(spec, [-5, -3, 1, 0], [5, 4], methods=["convex", "e_guidance"])
    for name, plan in out["plans"].items():
        assert np.isfinite(plan["miss"]) and np.abs(np.array(plan["path"])).max() < 50, name


def test_absurd_stiffness_rejected():
    spec = V.resolve("uav", {"drag": 100000, "mass": 0.01})
    assert spec["substeps"] == 0 and "too stiff" in spec["error"]


def test_near_stationary_classical():
    for vid in ("rocket", "uav", "usv"):
        spec = V.resolve(vid)
        out = V.plan_vehicle(spec, [0, 0, 0, 0], [1e-5, 0], methods=["convex"])
        plan = out["plans"]["convex"]
        assert abs(plan["loss"] - out["optimal_loss"]) < 1e-6, vid  # it is the optimum
        if spec["gravity"] == 0:  # with gravity, hovering costs effort, so the optimum sags slightly
            assert plan["miss"] < 1e-3, vid


def test_api_validation():
    try:
        import server  # needs JAX; skipped on machines without it
    except Exception as exc:  # pragma: no cover
        print(f"  (skipping API validation: {exc.__class__.__name__})")
        return
    ok = {"x0": [0, 0, 0, 0], "goal": [1, 1], "vehicle": "rocket"}
    server.validate_plan_request(ok)
    for bad in ({**ok, "vehicle": "car"}, {**ok, "x0": [1, 2]}, {**ok, "x0": None}, {"goal": [1, 1]},
                {**ok, "vehicle": []}, {**ok, "vehicle": {}}, {**ok, "methods": [{}]},
                {**ok, "methods": [[]]}, {**ok, "methods": ["magic"]}, {**ok, "params": {"mass": "heavy"}}, {**ok, "x0": [0, 0, float("inf"), 0]}):
        try:
            server.validate_plan_request(bad)
        except server.BadRequest:
            continue
        raise AssertionError(f"accepted invalid request {bad}")


def test_about_copy_is_honest():
    html = (Path(__file__).resolve().parent / "web" / "index.html").read_text(encoding="utf-8")
    assert "exact rendezvous" not in html and "hits exactly" not in html
    assert "even on reachable missions" in html and "not a reachability guarantee" in html


def test_long_horizon_uav_continuous_miss():
    for horizon in (200, 600):
        spec = V.resolve("uav", {"horizon": horizon})
        plan = V.plan_vehicle(spec, [0, 0, 0, 0], [1, 1], methods=["convex"])["plans"]["convex"]
        end = exact_point_mass(plan["controls"], [0, 0, 0, 0], spec)
        assert abs(np.linalg.norm(end[:2] - [1, 1]) - plan["miss"]) < 1e-6, (horizon, end, plan["miss"])


def test_rejected_plant_current_diagnostics():
    spec = V.resolve("uav", {"drag": 100000})
    assert spec["error"] and abs(spec["derived"]["top_speed"] - 0.0003) < 1e-12
    assert np.isfinite(spec["derived"]["stiffness"])
    assert 0 < spec["derived"]["reach"] < 0.01


def test_malformed_types_http_400():
    import server
    # Exercise the actual handler/catch path without a socket, server or Session.
    class NoPlanning:
        def plan(self, *args):
            raise AssertionError("invalid request reached planning")
    handler = server.make_handler(NoPlanning())
    ok = {"x0": [0, 0, 0, 0], "goal": [1, 1], "vehicle": "rocket"}
    class Request:
        path = "/api/plan"
        def _body(self):
            return self.body
        def _json(self, payload, code=200):
            self.code, self.payload = code, payload
    for extra in ({"vehicle": []}, {"vehicle": {}}, {"methods": [{}]}):
        req = Request()
        req.body = {**ok, **extra}
        handler.do_POST(req)
        assert req.code == 400 and req.payload["kind"] == "invalid", req.payload
    server.validate_plan_request({**ok, "methods": ["convex", "e_guidance", "neural"]})


def test_jax_exact_transition_and_gradient():
    import jax
    import jax.numpy as jnp
    from simulator import simulate
    rng = np.random.default_rng(42)
    # Float64 checks isolate integrator error from JAX's default float32 roundoff.
    with jax.enable_x64():
        for vid, over in (("uav", {"horizon": 600}), ("usv", {"horizon": 200}),
                          ("uav", {"drag": 100}), ("rocket", {})):
            spec = V.resolve(vid, over)
            u = rng.uniform(-1, 1, (V.N, 2))
            x0 = np.array([0.3, -0.2, 0.5, 0.1])
            end, _ = simulate(jnp.array(u), jnp.array(x0), spec["dt"], V.plant_tuple(spec))
            assert np.max(np.abs(np.asarray(end) - exact_point_mass(u, x0, spec))) < 1e-8
        spec = V.resolve("usv")
        gradient = jax.grad(lambda u: simulate(u, jnp.zeros(4), spec["dt"], V.plant_tuple(spec))[0][0])(jnp.zeros((V.N, 2)))
        assert np.allclose(np.asarray(gradient)[:, 0], V.influence_weights(spec), atol=1e-10)


def test_frontend_mirrors():
    root = Path(__file__).resolve().parent.parent
    for name in ("app.js", "plant.js", "index.html"):
        assert (root / "otis" / "web" / name).read_bytes() == (root / "docs" / name).read_bytes(), name


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} regression tests passed")
