#!/usr/bin/env python3
"""Experiment 1: hold a Cartesian pose and apply a known end-effector force."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cartesian_impedance_mujoco.config import load_yaml, resolve_path
from cartesian_impedance_mujoco.controller import diagonal_impedance, orientation_error, task_torque
from cartesian_impedance_mujoco.panda import (
    apply_site_wrench,
    geom_in_contact,
    place_front_wall,
    site_jacobian,
    site_quat,
)
from cartesian_impedance_mujoco.visualize import draw_force_arrow, draw_torque_arrow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "experiment1.yaml",
        help="YAML with robot, impedance, and disturbance parameters",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open the MuJoCo window (csv/plot only)",
    )
    return parser.parse_args()


def _as_vec(value, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size:
        raise ValueError(f"expected length {size}, got {array.size}")
    return array


def run(config_path: Path, *, headless: bool = False) -> None:
    import mujoco

    cfg = load_yaml(config_path)
    sim = cfg["sim"]
    robot = cfg["robot"]
    impedance = cfg["impedance"]
    experiment = cfg["experiment"]
    log_cfg = cfg["log"]

    model_path = resolve_path(ROOT, sim["model"])
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = float(sim["timestep"])
    data = mujoco.MjData(model)

    site_name = str(robot["site"])
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise RuntimeError(f"site not found: {site_name}")

    home_q = _as_vec(robot["home_q"], 7)
    data.qpos[:7] = home_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    position_d = data.site_xpos[site_id].copy()
    orientation_d = site_quat(model, data, site_id)

    stiffness, damping = diagonal_impedance(
        _as_vec(impedance["stiffness_trans"], 3),
        _as_vec(impedance["stiffness_rot"], 3),
        _as_vec(impedance["damping_zeta"], 6),
        None if impedance.get("damping_trans") is None else _as_vec(impedance["damping_trans"], 3),
        None if impedance.get("damping_rot") is None else _as_vec(impedance["damping_rot"], 3),
    )
    impulse = _as_vec(experiment["impulse"], 3)
    angular_impulse = _as_vec(experiment.get("angular_impulse", [0.0, 0.0, 0.0]), 3)
    impulse_time = float(experiment["impulse_time"])
    wall_cfg = cfg.get("wall") or {}
    wall_gap = float(wall_cfg.get("gap", 0.03))
    wall_size = _as_vec(wall_cfg.get("size", [0.03, 0.45, 0.45]), 3)
    wall_inner_x = place_front_wall(model, data, "wall", position_d, wall_gap, wall_size)
    duration = float(sim["duration"])
    dt = float(model.opt.timestep)
    use_viewer = bool(sim.get("viewer", True)) and not headless
    realtime = float(sim.get("realtime", 1.0))
    sync_every = max(1, int(round(0.016 / dt)))

    ref_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ref_marker")
    if ref_body >= 0:
        mocap_id = int(model.body_mocapid[ref_body])
        if mocap_id >= 0:
            data.mocap_pos[mocap_id] = position_d
            data.mocap_quat[mocap_id] = orientation_d

    viewer = None
    if use_viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.2
        viewer.cam.lookat[:] = [0.3, 0.0, 0.4]
        print("窗口：绿球=参考，红点=末端，灰墙=固定刚体，红箭=力冲量，蓝箭=力矩冲量。")

    rows: list[list[float]] = []
    t = 0.0
    step = 0
    wall_clock0 = time.perf_counter()
    impulse_done = False
    arrow_until = -1.0
    hit_wall = False
    try:
        while t < duration + 0.5 * dt:
            if viewer is not None and not viewer.is_running():
                break
            mujoco.mj_forward(model, data)
            dq = np.array(data.qvel[:7])
            position = np.array(data.site_xpos[site_id])
            orientation = site_quat(model, data, site_id)
            jacobian = site_jacobian(model, data, site_id)[:, :7]

            tau_task = task_torque(
                jacobian, dq, position, orientation, position_d, orientation_d, stiffness, damping
            )
            data.ctrl[:7] = tau_task + np.array(data.qfrc_bias[:7])

            applying = (not impulse_done) and t >= impulse_time
            if applying:
                apply_site_wrench(model, data, site_id, impulse / dt, angular_impulse / dt)
                impulse_done = True
                arrow_until = t + 0.2
            else:
                data.qfrc_applied[:] = 0.0

            error = position - position_d
            ori_error = orientation_error(orientation_d, orientation)
            tau_cmd = np.array(data.ctrl[:7], dtype=float)
            mujoco.mj_step(model, data)
            contacting = geom_in_contact(model, data, "wall")
            hit_wall = hit_wall or contacting
            rows.append(
                [
                    t,
                    position[0],
                    position[1],
                    position[2],
                    error[0],
                    error[1],
                    error[2],
                    ori_error[0],
                    ori_error[1],
                    ori_error[2],
                    float(applying),
                    float(contacting),
                    *tau_cmd.tolist(),
                ]
            )
            t = float(data.time)
            step += 1

            if viewer is not None and step % sync_every == 0:
                viewer.user_scn.ngeom = 0
                if t <= arrow_until:
                    draw_force_arrow(viewer.user_scn, position, impulse / dt)
                    draw_torque_arrow(viewer.user_scn, position, angular_impulse / dt)
                viewer.sync()
                if realtime > 0.0:
                    target = t / realtime
                    lag = target - (time.perf_counter() - wall_clock0)
                    if lag > 0.0:
                        time.sleep(lag)
    finally:
        if viewer is not None:
            viewer.close()

    csv_path = resolve_path(ROOT, log_cfg["csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["t", "x", "y", "z", "ex", "ey", "ez", "eox", "eoy", "eoz", "force_on", "wall_contact"]
            + [f"tau{i}" for i in range(1, 8)]
        )
        writer.writerows(rows)

    times = np.array([row[0] for row in rows])
    errors = np.array([row[4:7] for row in rows])
    ori_errors = np.array([row[7:10] for row in rows])
    torques = np.array([row[12:19] for row in rows])
    after = times >= impulse_time + 2.0
    peak = errors[times >= impulse_time] if np.any(times >= impulse_time) else errors
    peak_abs = peak[np.argmax(np.linalg.norm(peak, axis=1))]
    ori_peak = ori_errors[times >= impulse_time] if np.any(times >= impulse_time) else ori_errors
    ori_peak_abs = ori_peak[np.argmax(np.linalg.norm(ori_peak, axis=1))]
    recovered = errors[after].mean(axis=0) if np.any(after) else errors[-1]
    ori_recovered = ori_errors[after].mean(axis=0) if np.any(after) else ori_errors[-1]

    print("reference position [m]:", position_d)
    print("wall inner face x [m]:", wall_inner_x, "(gap", wall_gap, "m)")
    print("linear impulse [N·s]:", impulse)
    print("angular impulse [N·m·s]:", angular_impulse)
    print("peak displacement after impulse [m]:", peak_abs)
    print("peak orientation error after impulse [rad]:", ori_peak_abs)
    print("hit wall:", hit_wall)
    print("mean position error after recovery [m]:", recovered)
    print("mean orientation error after recovery [rad]:", ori_recovered)
    print("csv:", csv_path)

    plot_path = resolve_path(ROOT, log_cfg["plot"])
    torque_plot_path = resolve_path(ROOT, log_cfg.get("plot_torque", "logs/experiment1_torque.png"))
    try:
        import matplotlib

        if headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skip plot")
        return

    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(8, 10))
    pos_labels = ["x", "y", "z"]
    rot_labels = ["rx", "ry", "rz"]
    for i in range(3):
        axes[i].plot(times, errors[:, i], label="measured")
        axes[i].axvline(impulse_time, color="C1", linestyle="--", label="impulse" if i == 0 else None)
        if i == 0:
            axes[i].axhline(wall_inner_x - position_d[0], color="0.4", linestyle=":", label="wall")
        axes[i].set_ylabel(f"e_{pos_labels[i]} [m]")
        axes[i].grid(True, alpha=0.3)
    for i in range(3):
        axis = axes[i + 3]
        axis.plot(times, ori_errors[:, i])
        axis.axvline(impulse_time, color="C1", linestyle="--")
        axis.set_ylabel(f"e_{rot_labels[i]} [rad]")
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Experiment 1: position and orientation error after wrench impulse")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    print("plot:", plot_path)

    fig_tau, axes_tau = plt.subplots(7, 1, sharex=True, figsize=(9, 10))
    limits = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
    for i, axis in enumerate(axes_tau):
        axis.plot(times, torques[:, i], color=f"C{i}")
        axis.axvline(impulse_time, color="C1", linestyle="--", alpha=0.7)
        axis.axhline(limits[i], color="0.6", linestyle=":", linewidth=0.8)
        axis.axhline(-limits[i], color="0.6", linestyle=":", linewidth=0.8)
        axis.set_ylabel(f"τ{i + 1}\n[N·m]", rotation=0, labelpad=18, va="center")
        axis.grid(True, alpha=0.3)
    axes_tau[-1].set_xlabel("t [s]")
    fig_tau.suptitle("Joint torques (command, includes gravity compensation)")
    fig_tau.tight_layout()
    fig_tau.savefig(torque_plot_path, dpi=120)
    print("torque plot:", torque_plot_path)

    if not headless:
        plt.show()


def main() -> None:
    args = parse_args()
    run(args.config.resolve(), headless=args.headless)


if __name__ == "__main__":
    main()
