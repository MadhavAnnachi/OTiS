"""Local web server for the interactive simulator UI.

    python main.py            # serves http://localhost:8765
    python main.py --port 9000

Standard library only (``http.server``). Training runs on a background
thread; the browser polls ``/api/state`` and posts commands. JAX work for
interactive planning happens on the request thread with the newest weights.

Endpoints
---------
GET  /                  the UI (web/index.html)
GET  /api/state         training phase, step, loss history, showcase plan
GET  /api/evolution     snapshots of the showcase plan over training
GET  /api/benchmark     outputs/benchmark_results.json, if present
POST /api/plan          {x0, goal, vehicle, params} -> every applicable planner,
                        scored in the plant those parameters define
POST /api/train         {size, steps, vehicle, params, x0, goal} -> trains a
                        network on that plant (thrust-driven vehicles)
POST /api/pause | /api/resume | /api/stop
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

try:
    from . import config
    from .baselines import ConvexGuidance
    from .network import apply_policy, count_params
    from .train import load_checkpoint, train
    from .vehicles import (VEHICLES, catalog_public, influence_weights, plan_vehicle, plant_tuple,
                           resolve, signature_slug, vehicle_public)
except ImportError:
    import config
    from baselines import ConvexGuidance
    from network import apply_policy, count_params
    from train import load_checkpoint, train
    from vehicles import (VEHICLES, catalog_public, influence_weights, plan_vehicle, plant_tuple,
                          resolve, signature_slug, vehicle_public)

PACKAGE_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PACKAGE_ROOT / "web"
MAX_SNAPSHOTS = 160


def _tolist(a):
    return np.asarray(a, dtype=np.float64).round(6).tolist()


class Session:
    """Owns the training thread and every trained network, keyed by plant.

    A network is only valid for the physics and objective it was trained on,
    so each checkpoint is registered under ``vehicles.signature(spec)``.
    ``/api/plan`` shows the network only when the user's parameters match
    one; ``/api/train`` trains a new one on the parameters in the request.
    """

    def __init__(self, checkpoint: Path, defaults: dict):
        self.lock = threading.Lock()
        self.checkpoint = checkpoint
        self.models_dir = checkpoint.parent / "models"
        self.defaults = defaults
        self.thread = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.registry = {}  # signature -> {model, params, meta, info, fast}
        self.convex_cache = {}  # signature -> ConvexGuidance
        self.training = None  # {signature, spec, params, rollout} while training
        self.model_info = {}
        self.state = {"phase": "idle", "step": 0, "total_steps": 0, "history": {}, "message": ""}
        self.snapshots = []
        self._next_snap = 0
        self.scenario_box = None
        self.version = 0
        self._load_registry()

    # ---------------- weights ----------------
    def _register(self, path: Path, source: str):
        model, params, meta = load_checkpoint(path)
        vehicle_id = meta.get("vehicle", "rocket")
        if "params" in meta:
            spec = resolve(vehicle_id, meta["params"])
        else:  # checkpoints from before parameters were editable: unit plant
            spec = resolve("rocket", {"lambda_fuel": meta.get("lambda_fuel", config.LAMBDA_FUEL)})
        info = {
            "signature": spec["signature"],
            "vehicle": vehicle_id,
            "size": meta.get("size"),
            "n_params": meta.get("n_params") or count_params(params),
            "step": meta.get("step"),
            "workspace": meta.get("position_limit"),
            "start_speed": meta.get("velocity_limit"),
            "val_distance": meta.get("val_distance"),
            "file": path.name,
            "source": source,
        }
        self.registry.setdefault(spec["signature"], {"model": model, "params": params, "meta": meta, "info": info, "fast": None})
        return info

    def _load_registry(self):
        paths = [self.checkpoint] + sorted(self.models_dir.glob("*.pkl"))
        for path in paths:
            if not path.exists() or not path.with_suffix(".json").exists():
                continue
            try:
                info = self._register(path, "checkpoint")
                print(f"Loaded {path.name}: {info['n_params']:,} params for {info['signature']}", flush=True)
            except Exception as exc:  # stale checkpoint from an older architecture
                print(f"Could not load {path}: {exc}", flush=True)
        if self.registry:
            first = next(iter(self.registry.values()))
            self.model_info = dict(first["info"])
            self.state["phase"] = "pretrained"
            self.state["message"] = f"{len(self.registry)} trained network(s) loaded"
            hist_path = self.checkpoint.parent / "history.json"
            if hist_path.exists():
                self.state["history"] = json.loads(hist_path.read_text())
        else:
            self.state["message"] = "No trained weights yet. Press Train."

    def models_public(self):
        with self.lock:
            return [dict(entry["info"]) for entry in self.registry.values()]

    # ---------------- training ----------------
    def on_update(self, **kw):
        with self.lock:
            params = kw.pop("params", None)
            rollout = kw.pop("rollout", None)
            if params is not None and self.training is not None:
                self.training["params"] = params
                self.training["rollout"] = rollout
                self.model_info = {"size": kw.get("size"), "n_params": kw.get("n_params"), "source": "training",
                                   "step": kw.get("step"), "signature": self.training["signature"]}
            st = self.state
            for k in ("phase", "step", "total_steps", "current_loss", "distance", "fuel", "rate", "n_params", "size"):
                if k in kw:
                    st[k] = kw[k]
            if "history" in kw:
                st["history"] = {k: list(v) for k, v in kw["history"].items()}
            if "trajectory" in kw:
                st["trajectory"] = _tolist(kw["trajectory"])
                st["controls"] = _tolist(kw["controls"])
                st["x0"] = _tolist(kw["x0"])
                st["goal"] = _tolist(kw["goal"])
                step = int(kw.get("step", 0))
                total = max(int(kw.get("total_steps", 1)), 1)
                stride = max(1, total // MAX_SNAPSHOTS)
                if kw.get("phase") == "training" and (step <= 40 or step >= self._next_snap):
                    self._next_snap = step + stride
                    self.snapshots.append({
                        "step": step,
                        "path": st["trajectory"],
                        "controls": st["controls"],
                        "loss": kw.get("current_loss"),
                        "distance": kw.get("distance"),
                        "x0": st["x0"],
                        "goal": st["goal"],
                    })
            self.version += 1

    def start_training(self, opts: dict):
        if self.thread is not None and self.thread.is_alive():
            return False, "Training already running"
        spec = resolve(opts.get("vehicle", "rocket"), opts.get("params"))
        if spec.get("error"):
            return False, spec["error"]
        if spec["kind"] != "point_mass":
            return False, f"Network training needs a thrust-driven plant (Rocket, UAV, USV), not the {spec['name']}."
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.snapshots = []
        self._next_snap = 0
        self.state = {
            "phase": "compiling", "step": 0, "total_steps": int(opts["steps"]), "history": {}, "message": "",
            "plant": {"vehicle": spec["id"], "name": spec["name"], "params": spec["params"], "signature": spec["signature"]},
        }
        lim = spec["workspace"]
        x0 = np.asarray(opts.get("x0") or [-0.8 * lim, -0.6 * lim, 0, 0], np.float32)
        goal = np.asarray(opts.get("goal") or [0.75 * lim, 0.65 * lim], np.float32)
        self.scenario_box = {"x0": x0, "goal": goal, "signature": spec["signature"]}
        self.training = {"signature": spec["signature"], "spec": spec, "params": None, "rollout": None}
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if spec["signature"] == resolve("rocket")["signature"]:
            # UI runs never overwrite the benchmark checkpoint.
            ckpt = self.checkpoint.with_name("ui_last_run.pkl")
        else:
            ckpt = self.models_dir / f"{spec['id']}-{signature_slug(spec['signature'])}.pkl"
        box = self.scenario_box

        def worker():
            try:
                result = train(
                    n_steps=int(opts["steps"]),
                    batch_size=int(opts["batch_size"]),
                    learning_rate=float(opts["lr"]),
                    lambda_fuel=spec["lambda_fuel"],
                    seed=int(opts["seed"]),
                    size=opts["size"],
                    position_limit=spec["workspace"],
                    velocity_limit=spec["start_speed"],
                    checkpoint_path=ckpt if opts.get("save", True) else None,
                    on_update=self.on_update,
                    stop_event=self.stop_event,
                    pause_event=self.pause_event,
                    scenario_box=box,
                    plant=plant_tuple(spec),
                    horizon=spec["horizon"],
                    plant_meta={"vehicle": spec["id"], "params": spec["params"], "signature": spec["signature"]},
                )
                with self.lock:
                    meta = {"size": result.size, "n_params": result.n_params, "step": result.best_step,
                            "position_limit": spec["workspace"], "velocity_limit": spec["start_speed"],
                            "val_distance": result.best_val_distance, "vehicle": spec["id"], "params": spec["params"]}
                    info = {"signature": spec["signature"], "vehicle": spec["id"], "size": result.size,
                            "n_params": result.n_params, "step": result.best_step, "workspace": spec["workspace"],
                            "start_speed": spec["start_speed"], "val_distance": result.best_val_distance,
                            "file": ckpt.name, "source": "trained this session"}
                    # The newest run wins for this plant.
                    self.registry[spec["signature"]] = {"model": result.model, "params": result.best_params,
                                                        "meta": meta, "info": info, "fast": None}
                    self.model_info = dict(info)
                    self.training = None
                    self.state["phase"] = "done"
                    self.state["best_step"] = result.best_step
                    self.state["train_seconds"] = result.elapsed_sec
                    self.state["message"] = f"Best weights from step {result.best_step}"
                    self.version += 1
            except Exception as exc:
                traceback.print_exc()
                with self.lock:
                    self.training = None
                    self.state["phase"] = "error"
                    self.state["message"] = str(exc)

        self.thread = threading.Thread(target=worker, name="otis-train", daemon=True)
        self.thread.start()
        return True, "started"

    # ---------------- planning ----------------
    def _convex_for(self, spec):
        sig = spec["signature"]
        solver = self.convex_cache.get(sig)
        if solver is None:
            solver = ConvexGuidance(lambda_fuel=spec["lambda_fuel"], b=influence_weights(spec))
            self.convex_cache[sig] = solver
        return solver

    def _neural_for(self, spec, x0, goal):
        """``(neural_fn or None, status dict)`` for this plant."""

        sig = spec["signature"]
        if sig is None:
            return None, {"available": False, "trainable": False,
                          "reason": "The network trains through a differentiable thrust plant; the car is planned classically for now."}
        with self.lock:
            training = self.training if self.training and self.training["signature"] == sig else None
            entry = self.registry.get(sig)
        if training is not None and training["params"] is not None:
            params, rollout = training["params"], training["rollout"]

            def live(a, b):
                a, b = jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)
                return lambda: rollout(params, a, b)[0].block_until_ready()

            return live, {"available": True, "trainable": True, "live": True, "reason": "Live weights from the run in progress."}
        if entry is None:
            return None, {"available": False, "trainable": True,
                          "reason": "No network has been trained for these parameters yet."}
        if entry["fast"] is None:
            # Frozen weights compiled in as constants, on the CPU: one plan is
            # ~1 ms there, while each GPU call pays a few ms of WSL sync.
            cpu = jax.devices("cpu")[0]
            p_cpu = jax.device_put(entry["params"], cpu)
            model = entry["model"]
            jfn = jax.jit(lambda a, b, _p=p_cpu: apply_policy(model, _p, a, b))

            def fast(a, b):
                a = jax.device_put(jnp.asarray(a, jnp.float32), cpu)
                b = jax.device_put(jnp.asarray(b, jnp.float32), cpu)
                return lambda: jfn(a, b).block_until_ready()

            entry["fast"] = fast
        info = entry["info"]
        status = {"available": True, "trainable": True, "model": info, "reason": ""}
        ws, vs = info.get("workspace"), info.get("start_speed")
        if ws and (np.max(np.abs(np.r_[x0[:2], goal])) > ws * 1.001):
            status["warning"] = f"Outside the network's training workspace (±{ws:g} m): expect extrapolation error."
        elif vs is not None and np.max(np.abs(x0[2:])) > vs * 1.001:
            status["warning"] = f"Start velocity beyond the training range (±{vs:g} m/s): expect extrapolation error."
        return entry["fast"], status

    def plan(self, x0, goal, methods=None, vehicle="rocket", params=None):
        spec = resolve(vehicle, params)
        if spec.get("error"):
            raise BadRequest(spec["error"], 422)
        lim = spec["workspace"]
        x0 = np.asarray(x0, np.float64)
        goal = np.clip(np.asarray(goal, np.float64), -lim, lim)
        x0[:2] = np.clip(x0[:2], -lim, lim)
        box = self.scenario_box
        if box is not None and box.get("signature") == spec["signature"]:
            box["x0"], box["goal"] = x0.astype(np.float32), goal.astype(np.float32)
        neural_fn, neural_status = self._neural_for(spec, x0, goal)
        convex = self._convex_for(spec) if spec["kind"] == "point_mass" else None
        out = plan_vehicle(spec, x0, goal, methods, neural_fn=neural_fn, convex=convex)
        out["vehicle"] = vehicle_public(VEHICLES[spec["id"]])
        out["spec"] = {k: spec[k] for k in ("params", "horizon", "dt", "workspace", "start_speed",
                                             "lambda_fuel", "signature", "derived", "clamped")}
        out["neural_status"] = neural_status
        return out


class BadRequest(ValueError):
    """A client error: answered with HTTP 400 (or ``code``) and a readable message."""

    def __init__(self, message, code=400):
        super().__init__(message)
        self.code = code


KNOWN_METHODS = {"neural", "convex", "e_guidance", "pursuit"}
MAX_ABS_INPUT = 1e6  # positions in m, velocities in m/s


def _finite_vector(value, n, name):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise BadRequest(f"'{name}' must be a list of {n} numbers")
    try:
        out = np.asarray([float(v) for v in value], np.float64)
    except (TypeError, ValueError):
        raise BadRequest(f"'{name}' must contain only numbers") from None
    if not np.all(np.isfinite(out)):
        raise BadRequest(f"'{name}' must be finite")
    if np.max(np.abs(out)) > MAX_ABS_INPUT:
        raise BadRequest(f"'{name}' values must be within ±{MAX_ABS_INPUT:g}")
    return out


def validate_plan_request(body):
    """Return (x0, goal, methods, vehicle, params) or raise BadRequest."""

    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    vehicle = body.get("vehicle", "rocket")
    if not isinstance(vehicle, str) or vehicle not in VEHICLES:
        raise BadRequest(f"unknown vehicle '{vehicle}'; expected one of {sorted(VEHICLES)}")
    x0 = _finite_vector(body.get("x0"), 4, "x0")
    goal = _finite_vector(body.get("goal"), 2, "goal")
    methods = body.get("methods")
    if methods is not None:
        if not isinstance(methods, list) or not methods:
            raise BadRequest("'methods' must be a non-empty list")
        unknown = [m for m in methods if not isinstance(m, str) or m not in KNOWN_METHODS]
        if unknown:
            raise BadRequest(f"unknown method(s) {unknown}; expected some of {sorted(KNOWN_METHODS)}")
        if not set(methods) & set(VEHICLES[vehicle]["methods"]):
            raise BadRequest(f"none of {methods} apply to the {VEHICLES[vehicle]['name']}")
    params = validate_params(body.get("params"))
    return x0, goal, methods, vehicle, params


def validate_params(params):
    if params is None:
        return None
    if not isinstance(params, dict):
        raise BadRequest("'params' must be an object of numbers")
    for k, v in params.items():
        try:
            ok = np.isfinite(float(v))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise BadRequest(f"parameter '{k}' must be a finite number")
    return params


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, payload, code=200):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/state":
                with session.lock:
                    st = dict(session.state)
                    st["model"] = dict(session.model_info)
                    st["has_model"] = bool(session.registry) or session.training is not None
                    st["snapshots"] = len(session.snapshots)
                    st["version"] = session.version
                st["config"] = {**session.defaults, "n_steps": config.N_STEPS, "dt": config.DT,
                                "max_thrust": config.MAX_THRUST, "sizes": config.MODEL_SIZES,
                                "device": str(jax.devices()[0]),
                                "app_name": "OTiS",
                                "app_full_name": "Optimized Trajectory Simulator",
                                "catalog": catalog_public()}
                st["models"] = session.models_public()
                return self._json(st)
            if path == "/api/evolution":
                with session.lock:
                    return self._json({"snapshots": list(session.snapshots)})
            if path == "/api/benchmark":
                f = PACKAGE_ROOT / "outputs" / "benchmark_results.json"
                return self._json(json.loads(f.read_text()) if f.exists() else {"missing": True})
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (WEB_ROOT / rel).resolve()
            if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve() or not target.is_file():
                self.send_error(404)
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                body = self._body()
                if path == "/api/plan":
                    return self._json(session.plan(*validate_plan_request(body)))
                if path == "/api/train":
                    if body.get("vehicle", "rocket") not in VEHICLES:
                        raise BadRequest(f"unknown vehicle '{body.get('vehicle')}'")
                    validate_params(body.get("params"))
                    opts = {**session.defaults, **{k: v for k, v in body.items() if v is not None}}
                    ok, msg = session.start_training(opts)
                    return self._json({"ok": ok, "message": msg}, 200 if ok else 409)
                if path == "/api/pause":
                    session.pause_event.set()
                    return self._json({"ok": True})
                if path == "/api/resume":
                    session.pause_event.clear()
                    return self._json({"ok": True})
                if path == "/api/stop":
                    session.stop_event.set()
                    session.pause_event.clear()
                    return self._json({"ok": True})
                self.send_error(404)
            except BadRequest as exc:
                self._json({"error": str(exc), "kind": "invalid"}, exc.code)
            except json.JSONDecodeError:
                self._json({"error": "request body is not valid JSON", "kind": "invalid"}, 400)
            except Exception as exc:
                traceback.print_exc()
                self._json({"error": f"planner failure: {exc}", "kind": "server"}, 500)

    return Handler


def serve(checkpoint: Path, defaults: dict, host: str = "127.0.0.1", port: int = 8765, autotrain: bool = False):
    session = Session(checkpoint, defaults)
    httpd = ThreadingHTTPServer((host, port), make_handler(session))
    shown = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    print(f"\nOTiS is running at  http://{shown}:{port}\n  device: {jax.devices()[0]}   (Ctrl+C to quit)\n", flush=True)
    if autotrain:
        session.start_training(defaults)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…", flush=True)
    finally:
        session.stop_event.set()
        httpd.server_close()
