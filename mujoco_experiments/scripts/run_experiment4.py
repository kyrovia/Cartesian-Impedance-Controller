#!/usr/bin/env python3
"""Experiment 4: hold Cartesian pose, step nullspace configuration."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cartesian_impedance_mujoco.config import load_yaml, resolve_path
from cartesian_impedance_mujoco.controller import (
    diagonal_impedance,
    nullspace_torque,
    orientation_error,
    task_torque,
)
from cartesian_impedance_mujoco.panda import set_mocap_pose, site_jacobian, site_quat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "experiment4.yaml")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--no-nullspace",
        action="store_true",
        help="Temporary: Kn=0, do not step q_d, hold Cartesian pose only",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=None,
        help="Override sim/control dt in seconds (e.g. 0.0005 for 2000 Hz)",
    )
    return parser.parse_args()


def _as_vec(value, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size:
        raise ValueError(f"expected length {size}, got {array.size}")
    return array


def run(
    config_path: Path,
    *,
    headless: bool = False,
    no_nullspace: bool = False,
    timestep: float | None = None,
) -> None:
    import mujoco

    cfg = load_yaml(config_path)
    sim = cfg["sim"]
    robot = cfg["robot"]
    impedance = cfg["impedance"]
    nullspace = cfg["nullspace"]
    experiment = cfg["experiment"]
    log_cfg = cfg["log"]

    model = mujoco.MjModel.from_xml_path(str(resolve_path(ROOT, sim["model"])))
    model.opt.timestep = float(timestep if timestep is not None else sim["timestep"])
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(robot["site"]))
    if site_id < 0:
        raise RuntimeError(f"site not found: {robot['site']}")

    home_q = _as_vec(robot["home_q"], 7)
    data.qpos[:7] = home_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    position_d = np.array(data.site_xpos[site_id])
    orientation_d = site_quat(model, data, site_id)
    q_d = home_q.copy()
    q_d_step = home_q + _as_vec(experiment["q_offset"], 7)

    stiffness, damping = diagonal_impedance(
        _as_vec(impedance["stiffness_trans"], 3),
        _as_vec(impedance["stiffness_rot"], 3),
        _as_vec(impedance["damping_zeta"], 6),
        None if impedance.get("damping_trans") is None else _as_vec(impedance["damping_trans"], 3),
        None if impedance.get("damping_rot") is None else _as_vec(impedance["damping_rot"], 3),
    )
    kn = float(nullspace["stiffness"])
    dn = float(nullspace.get("damping_zeta", 1.0)) * 2.0 * np.sqrt(max(kn, 0.0))
    if no_nullspace:
        kn = 0.0
        dn = 0.0
    duration = float(sim["duration"])
    dt = float(model.opt.timestep)
    step_time = float(experiment["step_time"])
    use_viewer = bool(sim.get("viewer", True)) and not headless
    realtime = float(sim.get("realtime", 1.0))
    sync_every = max(1, int(round(0.016 / dt)))

    set_mocap_pose(model, data, "target_marker", position_d, orientation_d)
    set_mocap_pose(model, data, "ref_marker", position_d, orientation_d)

    print("cartesian pose locked at", position_d)
    if no_nullspace:
        print("TEMP TEST: nullspace OFF, q_d stays at home, no q step")
    else:
        print("nullspace q step at t=", step_time, "offset [rad]:", q_d_step - home_q)
    print("nullspace stiffness:", kn)
    print("control/sim frequency [Hz]:", 1.0 / dt)

    viewer = None
    if use_viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.2
        viewer.cam.lookat[:] = [0.3, 0.0, 0.4]
        print("窗口：黄/绿球=锁死的末端期望。看肘部是否动、红点是否离开绿球。")

    rows: list[list[float]] = []
    t = 0.0
    step = 0
    wall_clock0 = time.perf_counter()
    try:
        while t < duration + 0.5 * dt:
            if viewer is not None and not viewer.is_running():
                break
            mujoco.mj_forward(model, data)
            if (not no_nullspace) and t >= step_time:
                q_d = q_d_step
            q = np.array(data.qpos[:7], dtype=float)
            dq = np.array(data.qvel[:7], dtype=float)
            position = np.array(data.site_xpos[site_id])
            orientation = site_quat(model, data, site_id)
            jacobian = site_jacobian(model, data, site_id)[:, :7]
            tau_task = task_torque(
                jacobian, dq, position, orientation, position_d, orientation_d, stiffness, damping
            )
            tau_null = (
                np.zeros(7)
                if no_nullspace
                else nullspace_torque(jacobian, q, dq, q_d, kn, dn)
            )
            data.ctrl[:7] = tau_task + tau_null + np.array(data.qfrc_bias[:7])
            data.qfrc_applied[:] = 0.0
            err_p = position - position_d
            err_o = orientation_error(orientation_d, orientation)
            rows.append([t, *err_p.tolist(), *err_o.tolist(), *q.tolist()])
            mujoco.mj_step(model, data)
            t = float(data.time)
            step += 1
            if viewer is not None and step % sync_every == 0:
                viewer.sync()
                if realtime > 0.0:
                    lag = t / realtime - (time.perf_counter() - wall_clock0)
                    if lag > 0.0:
                        time.sleep(lag)
    finally:
        if viewer is not None:
            viewer.close()

    table = np.asarray(rows, dtype=float)
    times = table[:, 0]
    err_p = table[:, 1:4]
    err_o = table[:, 4:7]
    qs = table[:, 7:14]
    after = times >= step_time
    peak_ee = np.max(np.abs(err_p[after]), axis=0) if np.any(after) else np.max(np.abs(err_p), axis=0)
    peak_ee_all = np.max(np.abs(err_p), axis=0)
    rms_ee = np.sqrt(np.mean(err_p**2, axis=0))
    peak_ori = np.max(np.abs(err_o[after]), axis=0) if np.any(after) else np.max(np.abs(err_o), axis=0)
    dq_peak = np.max(np.abs(qs[after] - home_q), axis=0) if np.any(after) else np.zeros(7)
    settle = times >= max(float(times[-1]) - 1.0, step_time + 2.0)
    if not np.any(settle):
        settle = times >= times[-1] - 1.0
    ss_p = err_p[settle].mean(axis=0)
    ss_o = err_o[settle].mean(axis=0)
    ss_p_abs = np.mean(np.abs(err_p[settle]), axis=0)
    ss_o_abs = np.mean(np.abs(err_o[settle]), axis=0)
    print("steady-state ee position error [m]:", ss_p, "|e|=", float(np.linalg.norm(ss_p)))
    print("steady-state |ee position error| [m]:", ss_p_abs)
    print("steady-state ee orientation error [rad]:", ss_o, "|e|=", float(np.linalg.norm(ss_o)))
    print("steady-state |ee orientation error| [rad]:", ss_o_abs)

    try:
        import matplotlib

        if headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skip plot")
        return

    if no_nullspace:
        ee_path = resolve_path(ROOT, "logs/experiment4_hold_only_ee.png")
        q_path = resolve_path(ROOT, "logs/experiment4_hold_only_q.png")
    else:
        ee_path = resolve_path(ROOT, log_cfg.get("plot_ee", "logs/experiment4_ee.png"))
        q_path = resolve_path(ROOT, log_cfg.get("plot_q", "logs/experiment4_q.png"))
    ee_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(8, 10))
    labels_p = ["ex [m]", "ey [m]", "ez [m]"]
    labels_o = ["erx [rad]", "ery [rad]", "erz [rad]"]
    for i in range(3):
        axes[i].plot(times, err_p[:, i])
        axes[i].axvline(step_time, color="0.5", linestyle=":")
        axes[i].set_ylabel(labels_p[i])
        axes[i].grid(True, alpha=0.3)
    for i in range(3):
        axes[i + 3].plot(times, err_o[:, i])
        axes[i + 3].axvline(step_time, color="0.5", linestyle=":")
        axes[i + 3].set_ylabel(labels_o[i])
        axes[i + 3].grid(True, alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Experiment 4: EE error while nullspace q steps")
    fig.tight_layout()
    fig.savefig(ee_path, dpi=120)
    print("ee plot:", ee_path)

    fig_q, axes_q = plt.subplots(7, 1, sharex=True, figsize=(8, 10))
    for i, axis in enumerate(axes_q):
        axis.plot(times, qs[:, i])
        axis.axvline(step_time, color="0.5", linestyle=":")
        axis.axhline(q_d_step[i], color="C1", linestyle="--", alpha=0.6)
        axis.set_ylabel(f"q{i + 1}\n[rad]", rotation=0, labelpad=18, va="center")
        axis.grid(True, alpha=0.3)
    axes_q[-1].set_xlabel("t [s]")
    fig_q.suptitle("Experiment 4: joint angles (dashed = nullspace target)")
    fig_q.tight_layout()
    fig_q.savefig(q_path, dpi=120)
    print("q plot:", q_path)
    if not headless:
        plt.show()


def main() -> None:
    args = parse_args()
    run(
        args.config.resolve(),
        headless=args.headless,
        no_nullspace=args.no_nullspace,
        timestep=args.timestep,
    )


if __name__ == "__main__":
    main()
