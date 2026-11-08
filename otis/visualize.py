"""Live training figure.

Two panels share one Matplotlib figure and refresh together on a
``FuncAnimation`` timer (200 ms):

- Left: the thrust plan the current weights produce for a fixed start and
  goal. The path is a line, thrust commands are arrows, the start is a blue
  dot, and the goal is a red star. A marker walks the path so the panel
  keeps moving between weight updates. The axes title is the loss of the
  plan on screen.
- Right: validation loss and the terminal-error term on a log scale.

Training itself runs on another thread and only publishes numpy snapshots.
This module never calls JAX.
"""

from __future__ import annotations

import threading

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

try:
    from .config import N_STEPS
except ImportError:
    from config import N_STEPS

ANIMATION_INTERVAL_MS = 200
DEMO_FRAMES = 22  # ~4.4 s at 200 ms, one demo case at a time

BG = "#121418"
AX_BG = "#181c24"
GRID = "#2c3444"
SPINE = "#3a4456"
TEXT = "#ece7dc"
MUTED = "#8e97a3"
START = "#5aa2ff"
GOAL = "#ff5a6a"
PATH = "#3ecf8e"
PATH_DIM = "#1d6b4a"
THRUST = "#e6a15c"
VEHICLE = "#f7f4ee"
TERMINAL = "#e07a5f"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": AX_BG,
            "axes.edgecolor": SPINE,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.facecolor": AX_BG,
            "legend.edgecolor": SPINE,
            "legend.labelcolor": TEXT,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def _finite_positive(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, np.nan)
    return arr


class TrainingMonitor:
    """Thread-safe snapshot plus the FuncAnimation that draws it."""

    def __init__(self, total_steps: int, position_limit: float, on_set_goal=None, on_set_start=None, on_random=None, on_replay=None):
        self.total_steps = int(total_steps)
        self.position_limit = float(position_limit)
        self.on_set_goal = on_set_goal
        self.on_set_start = on_set_start
        self.on_random = on_random
        self.on_replay = on_replay
        self._lock = threading.Lock()
        n = int(N_STEPS)
        self._state = {
            "phase": "compiling",
            "step": 0,
            "total_steps": self.total_steps,
            "scenario_id": "train",
            "loss_steps": np.zeros((0,), dtype=np.float64),
            "loss_totals": np.zeros((0,), dtype=np.float64),
            "loss_terminals": np.zeros((0,), dtype=np.float64),
            "trajectory": np.zeros((n + 1, 4), dtype=np.float32),
            "controls": np.zeros((n, 2), dtype=np.float32),
            "x0": np.zeros((4,), dtype=np.float32),
            "goal": np.array([0.5, 0.5], dtype=np.float32),
            "current_loss": float("nan"),
            "distance": float("nan"),
            "fuel": float("nan"),
            "message": "Compiling differentiated trainer",
            "demos": [],
            "error": "",
        }
        self.fig = None
        self._anim = None
        self._demo_frame0 = None
        self._view_lo = None
        self._view_hi = None
        self._view_scenario = None
        self._artists_ready = False

    def publish(self, **kwargs) -> None:
        """Replace snapshot fields. Arrays are copied under the lock."""

        with self._lock:
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    value = np.array(value, copy=True)
                self._state[key] = value

    def snapshot(self) -> dict:
        with self._lock:
            snap = dict(self._state)
            for key in (
                "loss_steps",
                "loss_totals",
                "loss_terminals",
                "trajectory",
                "controls",
                "x0",
                "goal",
            ):
                snap[key] = np.array(snap[key], copy=True)
            demos = []
            for demo in snap["demos"]:
                copied = dict(demo)
                for key in ("x0", "goal", "controls", "trajectory"):
                    if key in copied:
                        copied[key] = np.array(copied[key], copy=True)
                demos.append(copied)
            snap["demos"] = demos
            return snap

    def _build(self) -> None:
        if self.fig is not None:
            return
        _apply_style()
        self.fig = plt.figure(figsize=(12.6, 5.8), dpi=110)
        try:
            self.fig.canvas.manager.set_window_title("OTiS")
        except Exception:
            pass
        grid = self.fig.add_gridspec(
            1,
            2,
            width_ratios=[1.0, 1.12],
            left=0.06,
            right=0.975,
            top=0.78,
            bottom=0.16,
            wspace=0.28,
        )
        self.ax_traj = self.fig.add_subplot(grid[0, 0])
        self.ax_loss = self.fig.add_subplot(grid[0, 1])
        self._brand = self.fig.text(
            0.06,
            0.93,
            "OTiS",
            fontsize=16,
            fontweight="bold",
            color=TEXT,
            ha="left",
            va="center",
        )
        self._status = self.fig.text(
            0.975,
            0.93,
            "compiling",
            fontsize=10,
            color=MUTED,
            ha="right",
            va="center",
            fontfamily="monospace",
        )
        self._tagline = self.fig.text(
            0.06,
            0.875,
            "differentiable thrust planning",
            fontsize=9,
            color=MUTED,
            ha="left",
            va="center",
        )
        self._hint = self.fig.text(
            0.06,
            0.835,
            "Left-click the map to place a new goal     Right-click to move the start     Random / Replay below",
            fontsize=8,
            color=MUTED,
            ha="left",
            va="center",
        )

        limit = self.position_limit
        self._region = Rectangle(
            (-limit, -limit),
            2 * limit,
            2 * limit,
            fill=False,
            linestyle=(0, (3, 3)),
            linewidth=0.9,
            edgecolor=SPINE,
            zorder=1,
        )
        self.ax_traj.add_patch(self._region)
        (self._plan_line,) = self.ax_traj.plot(
            [], [], color=PATH_DIM, lw=1.4, alpha=0.9, zorder=2
        )
        (self._trail_line,) = self.ax_traj.plot([], [], color=PATH, lw=2.2, zorder=3)
        self._start = self.ax_traj.scatter([0.0], [0.0], s=42, c=START, zorder=4)
        self._goal = self.ax_traj.scatter(
            [0.5], [0.5], s=220, c=GOAL, marker="*", zorder=4
        )
        self._start_label = self.ax_traj.text(
            0.0, 0.0, "start", color=START, fontsize=8, ha="right", va="bottom", zorder=6
        )
        self._goal_label = self.ax_traj.text(
            0.0, 0.0, "goal", color=GOAL, fontsize=8, ha="left", va="bottom", zorder=6
        )
        zeros = np.zeros((int(N_STEPS), 2))
        self._quiver = self._make_quiver(zeros[:, 0], zeros[:, 1], zeros[:, 0], zeros[:, 1])
        self._vehicle = self.ax_traj.scatter(
            [0.0],
            [0.0],
            s=70,
            c=VEHICLE,
            edgecolors=PATH,
            linewidths=1.3,
            zorder=5,
        )
        self._info = self.ax_traj.text(
            0.02,
            0.02,
            "",
            transform=self.ax_traj.transAxes,
            fontsize=8,
            color=MUTED,
            fontfamily="monospace",
            va="bottom",
            ha="left",
        )
        self.ax_traj.set_title("loss —", loc="left", fontfamily="monospace", color=TEXT)
        self.ax_traj.set_xlabel("x")
        self.ax_traj.set_ylabel("y")
        self.ax_traj.set_aspect("equal", adjustable="box")
        self.ax_traj.grid(True, alpha=0.55)

        (self._loss_line,) = self.ax_loss.plot(
            [], [], color=PATH, lw=1.8, label="total loss"
        )
        (self._term_line,) = self.ax_loss.plot(
            [], [], color=TERMINAL, lw=1.6, label="terminal error"
        )
        self.ax_loss.set_yscale("log")
        self.ax_loss.set_xlabel("training step")
        self.ax_loss.set_ylabel("loss")
        self.ax_loss.set_title("Training curves", loc="left")
        self.ax_loss.grid(True, which="both", alpha=0.55)
        self.ax_loss.legend(loc="upper right", fontsize=8, framealpha=0.92)
        self.ax_loss.set_xlim(0, max(self.total_steps, 1))
        self.ax_loss.set_ylim(1e-3, 10)
        self._ax_random = self.fig.add_axes([0.06, 0.03, 0.13, 0.055])
        self._ax_replay = self.fig.add_axes([0.21, 0.03, 0.13, 0.055])
        self._btn_random = Button(self._ax_random, "Random", color="#2a3344", hovercolor="#3a4a62")
        self._btn_replay = Button(self._ax_replay, "Replay", color="#2a3344", hovercolor="#3a4a62")
        self._btn_random.label.set_color(TEXT)
        self._btn_replay.label.set_color(TEXT)
        self._btn_random.on_clicked(lambda _evt: self.on_random() if self.on_random else None)
        self._btn_replay.on_clicked(lambda _evt: self.on_replay() if self.on_replay else None)
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse)
        self._artists_ready = True

    def _on_mouse(self, event) -> None:
        """Left-click sets a goal; right-click sets a start. Ignore button axes."""

        if getattr(event, "inaxes", None) is not getattr(self, "ax_traj", None):
            return
        if event.xdata is None or event.ydata is None:
            return
        xy = (float(event.xdata), float(event.ydata))
        button = int(getattr(event, "button", 1))
        if button == 3:
            if self.on_set_start:
                self.on_set_start(xy)
            return
        if button == 1 and self.on_set_goal:
            self.on_set_goal(xy)

    def _make_quiver(self, x, y, u, v):
        return self.ax_traj.quiver(
            x,
            y,
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=5.5,
            color=THRUST,
            width=0.0045,
            pivot="tail",
            zorder=3,
        )

    def _set_quiver(self, origins, uv) -> None:
        """Replace thrust arrows. Recreate the quiver if the count changes."""

        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 2)
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        current = np.asarray(self._quiver.get_offsets())
        if current.ndim != 2 or len(current) != len(origins):
            self._quiver.remove()
            self._quiver = self._make_quiver(origins[:, 0], origins[:, 1], uv[:, 0], uv[:, 1])
            return
        self._quiver.set_offsets(origins)
        self._quiver.set_UVC(uv[:, 0], uv[:, 1])

    def _frame_window(self, scenario_id: str, x0, goal, trajectory) -> None:
        pts = [np.asarray(x0, dtype=np.float64)[:2], np.asarray(goal, dtype=np.float64)[:2]]
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim == 2 and traj.shape[0] > 0 and traj.shape[1] >= 2:
            pts.append(traj[:, :2])
        cloud = np.vstack(pts)
        pad = 0.45
        lo = cloud.min(axis=0) - pad
        hi = cloud.max(axis=0) + pad
        span = np.maximum(hi - lo, 1.7)
        center = 0.5 * (lo + hi)
        # Keep the viewport square so equal aspect does not crop a side.
        side = float(span.max())
        half = side / 2.0
        new_lo = center - half
        new_hi = center + half
        if self._view_scenario != scenario_id or self._view_lo is None:
            self._view_scenario = scenario_id
            self._view_lo = new_lo
            self._view_hi = new_hi
        else:
            alpha = 0.18
            self._view_lo = (1.0 - alpha) * self._view_lo + alpha * new_lo
            self._view_hi = (1.0 - alpha) * self._view_hi + alpha * new_hi
        self.ax_traj.set_xlim(float(self._view_lo[0]), float(self._view_hi[0]))
        self.ax_traj.set_ylim(float(self._view_lo[1]), float(self._view_hi[1]))

    def _fmt_loss(self, value: float) -> str:
        if not np.isfinite(value):
            return "—"
        if value >= 0.01:
            return f"{value:.4f}"
        return f"{value:.3e}"

    def _draw_case(
        self,
        *,
        trajectory,
        controls,
        x0,
        goal,
        loss,
        distance,
        fuel,
        head: int,
        scenario_id: str,
        info: str,
    ) -> None:
        traj = np.asarray(trajectory, dtype=np.float64)
        ctrls = np.asarray(controls, dtype=np.float64)
        x0 = np.asarray(x0, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[0] == 0:
            traj = np.zeros((1, 4), dtype=np.float64)
        head = int(np.clip(head, 0, traj.shape[0] - 1))
        self._plan_line.set_data(traj[:, 0], traj[:, 1])
        self._trail_line.set_data(traj[: head + 1, 0], traj[: head + 1, 1])
        self._start.set_offsets(np.asarray(x0[:2], dtype=np.float64).reshape(1, 2))
        self._goal.set_offsets(np.asarray(goal[:2], dtype=np.float64).reshape(1, 2))
        self._vehicle.set_offsets(np.asarray(traj[head, :2], dtype=np.float64).reshape(1, 2))

        n_ctrl = ctrls.shape[0] if ctrls.ndim == 2 else 0
        n_pos = max(traj.shape[0] - 1, 0)
        n = min(n_ctrl, n_pos)
        if n > 0:
            self._set_quiver(traj[:n, :2], ctrls[:n])
        else:
            self._set_quiver(np.zeros((1, 2)), np.zeros((1, 2)))

        self._frame_window(scenario_id, x0, goal, traj)
        span = 1.0
        if self._view_lo is not None:
            span = float(self._view_hi[0] - self._view_lo[0])
        offset = 0.04 * span
        self._start_label.set_position((float(x0[0]) - offset, float(x0[1]) + offset * 0.35))
        self._goal_label.set_position((float(goal[0]) + offset, float(goal[1]) + offset * 0.35))
        self.ax_traj.set_title(f"loss {self._fmt_loss(loss)}", loc="left", fontfamily="monospace")
        bits = []
        if np.isfinite(distance):
            bits.append(f"dist {distance:.3f}")
        if np.isfinite(fuel):
            bits.append(f"fuel {fuel:.2f}")
        if info:
            bits.append(info)
        self._info.set_text("   ".join(bits))

    def _draw_curves(self, snap: dict) -> None:
        steps = np.asarray(snap["loss_steps"], dtype=np.float64)
        totals = _finite_positive(snap["loss_totals"])
        terminals = _finite_positive(snap["loss_terminals"])
        if steps.size == 0:
            return
        self._loss_line.set_data(steps, totals)
        self._term_line.set_data(steps, terminals)
        finite = np.concatenate([totals[np.isfinite(totals)], terminals[np.isfinite(terminals)]])
        if finite.size:
            ymin = float(finite.min())
            ymax = float(finite.max())
            if ymax <= ymin:
                ymax = ymin * 10.0
                ymin = ymin / 10.0
            self.ax_loss.set_ylim(ymin * 0.7, ymax * 1.45)
        last = float(steps[-1])
        self.ax_loss.set_xlim(0.0, max(float(self.total_steps), last * 1.02, 1.0))

    def _select_view(self, snap: dict, frame: int):
        phase = snap["phase"]
        if phase == "demo" and snap["demos"]:
            if self._demo_frame0 is None:
                self._demo_frame0 = frame
            local = max(0, frame - self._demo_frame0)
            index = min(len(snap["demos"]) - 1, local // DEMO_FRAMES)
            within = local % DEMO_FRAMES
            demo = snap["demos"][index]
            traj = demo["trajectory"]
            head = int(round(within / max(DEMO_FRAMES - 1, 1) * (len(traj) - 1)))
            return {
                "trajectory": traj,
                "controls": demo["controls"],
                "x0": demo["x0"],
                "goal": demo["goal"],
                "loss": demo["loss"],
                "distance": demo["distance"],
                "fuel": demo["fuel"],
                "head": head,
                "scenario_id": f"demo-{index}",
                "info": f"demo {index + 1}/{len(snap['demos'])}",
                "status": f"demo {index + 1}/{len(snap['demos'])}",
            }
        self._demo_frame0 = None
        traj = snap["trajectory"]
        head = int(frame % max(len(traj), 1))
        step = int(snap["step"])
        total = int(snap.get("total_steps", self.total_steps))
        if phase == "compiling":
            status = "compiling"
            info = snap.get("message") or "compiling"
        elif phase == "error":
            status = "error"
            info = snap.get("error") or "error"
        else:
            status = f"step {step}/{total}"
            info = f"step {step}"
        return {
            "trajectory": traj,
            "controls": snap["controls"],
            "x0": snap["x0"],
            "goal": snap["goal"],
            "loss": snap["current_loss"],
            "distance": snap["distance"],
            "fuel": snap["fuel"],
            "head": head,
            "scenario_id": str(snap.get("scenario_id", "train")),
            "info": info,
            "status": status,
        }

    def _on_frame(self, frame: int):
        if not self._artists_ready:
            self._build()
        snap = self.snapshot()
        view = self._select_view(snap, int(frame))
        self._draw_case(
            trajectory=view["trajectory"],
            controls=view["controls"],
            x0=view["x0"],
            goal=view["goal"],
            loss=view["loss"],
            distance=view["distance"],
            fuel=view["fuel"],
            head=view["head"],
            scenario_id=view["scenario_id"],
            info=view["info"],
        )
        self._draw_curves(snap)
        self._status.set_text(view["status"])
        if snap["phase"] == "error" and snap.get("error"):
            self.ax_traj.set_title(str(snap["error"])[:80], loc="left", fontfamily="monospace")
        return (
            self._plan_line,
            self._trail_line,
            self._loss_line,
            self._term_line,
        )

    def _animate(self):
        self._build()
        self._anim = FuncAnimation(
            self.fig,
            self._on_frame,
            interval=ANIMATION_INTERVAL_MS,
            blit=False,
            cache_frame_data=False,
        )
        return self._anim

    def show(self, stop_event=None) -> None:
        """Block on the GUI. Closing the window sets ``stop_event``."""

        backend = matplotlib.get_backend().lower()
        if backend == "agg":
            raise RuntimeError(
                "The Agg backend cannot open a live window. "
                "Unset MPLBACKEND or pass --no-viz."
            )
        self._animate()
        try:
            mgr = self.fig.canvas.manager
            mgr.set_window_title("OTiS")
            win = getattr(mgr, "window", None)
            if win is not None:
                if hasattr(win, "lift"):
                    win.lift()
                if hasattr(win, "attributes"):
                    try:
                        win.attributes("-topmost", True)
                        win.attributes("-topmost", False)
                    except Exception:
                        pass
                if hasattr(win, "focus_force"):
                    win.focus_force()
        except Exception:
            pass

        def _on_close(_event):
            if stop_event is not None:
                stop_event.set()

        self.fig.canvas.mpl_connect("close_event", _on_close)
        plt.show()

    def render_frames(self, n: int = 3):
        """Step the same FuncAnimation callback without opening a window.

        Used by the headless self-check to prove the update function draws.
        """

        self._animate()
        for i in range(n):
            self._on_frame(i)
        self.fig.canvas.draw()
        if self._anim is not None and getattr(self._anim, "event_source", None) is not None:
            self._anim.event_source.stop()
        return self.fig


def save_report(path, history: dict, cases: list[dict], position_limit: float, note: str = "") -> None:
    """Write a static summary: the five demo trajectories and the loss curve."""

    _apply_style()
    n = max(len(cases), 1)
    fig = plt.figure(figsize=(2.5 * n + 1.2, 7.2), dpi=140)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.38, top=0.88, bottom=0.08, left=0.06, right=0.98)
    top = outer[0].subgridspec(1, n, wspace=0.28)
    ax_loss = fig.add_subplot(outer[1])
    fig.text(
        0.06,
        0.955,
        "OTiS",
        fontsize=16,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="center",
    )
    subtitle = "final demo" if not note else note
    fig.text(0.06, 0.915, subtitle, fontsize=9, color=MUTED, ha="left", va="center")

    for i, case in enumerate(cases):
        ax = fig.add_subplot(top[0, i])
        traj = np.asarray(case["trajectory"])
        ctrls = np.asarray(case["controls"])
        x0 = np.asarray(case["x0"])
        goal = np.asarray(case["goal"])
        ax.plot(traj[:, 0], traj[:, 1], color=PATH, lw=1.8, zorder=2)
        n_arrow = min(len(ctrls), max(len(traj) - 1, 0))
        if n_arrow:
            ax.quiver(
                traj[:n_arrow, 0],
                traj[:n_arrow, 1],
                ctrls[:n_arrow, 0],
                ctrls[:n_arrow, 1],
                angles="xy",
                scale_units="xy",
                scale=6.0,
                color=THRUST,
                width=0.006,
                pivot="tail",
                zorder=3,
            )
        ax.scatter([x0[0]], [x0[1]], s=28, c=START, zorder=4)
        ax.scatter([goal[0]], [goal[1]], s=140, c=GOAL, marker="*", zorder=4)
        limit = position_limit
        ax.add_patch(
            Rectangle(
                (-limit, -limit),
                2 * limit,
                2 * limit,
                fill=False,
                linestyle=(0, (3, 3)),
                linewidth=0.8,
                edgecolor=SPINE,
            )
        )
        pts = np.vstack([traj[:, :2], x0[:2], goal[:2]])
        lo = pts.min(axis=0) - 0.35
        hi = pts.max(axis=0) + 0.35
        span = np.maximum(hi - lo, 1.4)
        center = 0.5 * (lo + hi)
        half = float(span.max()) / 2.0
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.45)
        ax.set_title(
            f"case {i + 1}   d={case['distance']:.3f}",
            loc="left",
            fontsize=9,
            fontfamily="monospace",
        )
        ax.tick_params(labelsize=7)

    steps = np.asarray(history.get("step", []), dtype=np.float64)
    totals = _finite_positive(history.get("val_loss", []))
    terminals = _finite_positive(history.get("val_terminal", []))
    if steps.size:
        ax_loss.plot(steps, totals, color=PATH, lw=1.8, label="total loss")
        ax_loss.plot(steps, terminals, color=TERMINAL, lw=1.6, label="terminal error")
        finite = np.concatenate(
            [totals[np.isfinite(totals)], terminals[np.isfinite(terminals)]]
        )
        if finite.size:
            ymin = float(finite.min())
            ymax = float(finite.max())
            if ymax <= ymin:
                ymax = ymin * 10.0
                ymin = ymin / 10.0
            ax_loss.set_ylim(ymin * 0.7, ymax * 1.45)
        ax_loss.set_xlim(0, max(float(steps[-1]), 1.0))
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training curves", loc="left")
    ax_loss.grid(True, which="both", alpha=0.5)
    ax_loss.legend(loc="upper right", fontsize=8, framealpha=0.92)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
