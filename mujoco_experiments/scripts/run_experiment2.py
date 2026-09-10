#!/usr/bin/env python3
"""Experiment 2: large desired-pose step, with/without pose filtering."""

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
from cartesian_impedance_mujoco.controller import (
    PoseReference,
    diagonal_impedance,
    orientation_error,
    quat_mul,
    rpy_to_quat,
    task_torque,
)
from cartesian_impedance_mujoco.panda import set_mocap_pose, site_jacobian, site_linear_acc, site_quat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "experiment2.yaml",
        help="YAML with step size and pose filter",
    )
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def _as_vec(value, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size:
        raise ValueError(f"expected length {size}, got {array.size}")
    return array


def _simulate(
    *,
    model,
    data,
    site_id: int,
    home_q: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
    pose_filter: float,
    step_time: float,
    position_target: np.ndarray,
    orientation_target: np.ndarray,
    duration: float,
    dt: float,
    viewer,
    realtime: float,
) -> dict[str, np.ndarray]:
    import mujoco

    mujoco.mj_resetData(model, data)
    data.qpos[:7] = home_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    position_0 = np.array(data.site_xpos[site_id])
    orientation_0 = site_quat(model, data, site_id)
    frequency = 1.0 / dt
    pose_ref = PoseReference(position_0, orientation_0, frequency, pose_filter)
    stepped = False

    set_mocap_pose(model, data, "target_marker", position_0, orientation_0)
    set_mocap_pose(model, data, "ref_marker", pose_ref.position, pose_ref.orientation)

    rows: list[list[float]] = []
    t = 0.0
    step = 0
    wall_clock0 = time.perf_counter()
    sync_every = max(1, int(round(0.016 / dt)))

    while t < duration + 0.5 * dt:
        if viewer is not None and not viewer.is_running():
            break
        mujoco.mj_forward(model, data)
        if (not stepped) and t >= step_time:
            pose_ref.set_target(position_target, orientation_target)
            stepped = True

        pose_ref.step()
        set_mocap_pose(model, data, "target_marker", pose_ref.target_position, pose_ref.target_orientation)
        set_mocap_pose(model, data, "ref_marker", pose_ref.position, pose_ref.orientation)

        dq = np.array(data.qvel[:7])
        position = np.array(data.site_xpos[site_id])
        orientation = site_quat(model, data, site_id)
        jacobian = site_jacobian(model, data, site_id)[:, :7]
        tau_task = task_torque(
            jacobian,
            dq,
            position,
            orientation,
            pose_ref.position,
            pose_ref.orientation,
            stiffness,
            damping,
        )
        data.ctrl[:7] = tau_task + np.array(data.qfrc_bias[:7])
        data.qfrc_applied[:] = 0.0

        err_p = position - pose_ref.target_position
        err_o = orientation_error(pose_ref.target_orientation, orientation)
        tau_cmd = np.array(data.ctrl[:7], dtype=float)
        mujoco.mj_step(model, data)
        acc = site_linear_acc(model, data, site_id)
        rows.append(
            [t, *position.tolist(), *err_p.tolist(), *err_o.tolist(), *tau_cmd.tolist(), *acc.tolist()]
        )
        t = float(data.time)
        step += 1
        if viewer is not None and step % sync_every == 0:
            viewer.sync()
            if realtime > 0.0:
                lag = t / realtime - (time.perf_counter() - wall_clock0)
                if lag > 0.0:
                    time.sleep(lag)

    table = np.asarray(rows, dtype=float)
    return {
        "t": table[:, 0],
        "pos": table[:, 1:4],
        "err_p": table[:, 4:7],
        "err_o": table[:, 7:10],
        "tau": table[:, 10:17],
        "acc": table[:, 17:20],
    }


def run(config_path: Path, *, headless: bool = False) -> None:
    import mujoco

    cfg = load_yaml(config_path)
    sim = cfg["sim"]
    robot = cfg["robot"]
    impedance = cfg["impedance"]
    filtering = cfg["filtering"]
    experiment = cfg["experiment"]
    log_cfg = cfg["log"]

    model = mujoco.MjModel.from_xml_path(str(resolve_path(ROOT, sim["model"])))
    model.opt.timestep = float(sim["timestep"])
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(robot["site"]))
    if site_id < 0:
        raise RuntimeError(f"site not found: {robot['site']}")

    home_q = _as_vec(robot["home_q"], 7)
    data.qpos[:7] = home_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    position_0 = np.array(data.site_xpos[site_id])
    orientation_0 = site_quat(model, data, site_id)
    offset = _as_vec(experiment["position_offset"], 3)
    rpy = _as_vec(experiment["orientation_rpy"], 3)
    position_target = position_0 + offset
    orientation_target = quat_mul(rpy_to_quat(rpy), orientation_0)

    stiffness, damping = diagonal_impedance(
        _as_vec(impedance["stiffness_trans"], 3),
        _as_vec(impedance["stiffness_rot"], 3),
        _as_vec(impedance["damping_zeta"], 6),
        None if impedance.get("damping_trans") is None else _as_vec(impedance["damping_trans"], 3),
        None if impedance.get("damping_rot") is None else _as_vec(impedance["damping_rot"], 3),
    )

    pose_filter = float(filtering["pose"])
    compare = bool(filtering.get("compare", False))
    duration = float(sim["duration"])
    dt = float(model.opt.timestep)
    step_time = float(experiment["step_time"])
    use_viewer = bool(sim.get("viewer", True)) and not headless
    realtime = float(sim.get("realtime", 1.0))

    print("start pose [m]:", position_0)
    print("step pose [m]:", position_target, "delta [m]:", offset)
    print("orientation step rpy [rad]:", rpy, f"({np.degrees(rpy)} deg)")
    print("pose filter:", pose_filter, "(1.0 = off)")

    viewer = None
    if use_viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.2
        viewer.cam.lookat[:] = [0.3, 0.0, 0.4]
        print("窗口：黄球=阶跃后的期望，绿球=滤波后的参考，红点=末端。")

    try:
        filtered = _simulate(
            model=model,
            data=data,
            site_id=site_id,
            home_q=home_q,
            stiffness=stiffness,
            damping=damping,
            pose_filter=pose_filter,
            step_time=step_time,
            position_target=position_target,
            orientation_target=orientation_target,
            duration=duration,
            dt=dt,
            viewer=viewer,
            realtime=realtime,
        )
    finally:
        if viewer is not None:
            viewer.close()

    unfiltered = None
    if compare and pose_filter < 1.0:
        unfiltered = _simulate(
            model=model,
            data=data,
            site_id=site_id,
            home_q=home_q,
            stiffness=stiffness,
            damping=damping,
            pose_filter=1.0,
            step_time=step_time,
            position_target=position_target,
            orientation_target=orientation_target,
            duration=duration,
            dt=dt,
            viewer=None,
            realtime=0.0,
        )

    csv_path = resolve_path(ROOT, log_cfg["csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["t", "x", "y", "z", "ex", "ey", "ez", "eox", "eoy", "eoz"]
            + [f"tau{i}" for i in range(1, 8)]
            + ["ax", "ay", "az"]
        )
        for i in range(len(filtered["t"])):
            writer.writerow(
                [
                    filtered["t"][i],
                    *filtered["pos"][i],
                    *filtered["err_p"][i],
                    *filtered["err_o"][i],
                    *filtered["tau"][i],
                    *filtered["acc"][i],
                ]
            )

    after = filtered["t"] >= step_time + 2.5
    recovered_p = filtered["err_p"][after].mean(axis=0) if np.any(after) else filtered["err_p"][-1]
    recovered_o = filtered["err_o"][after].mean(axis=0) if np.any(after) else filtered["err_o"][-1]
    tau_peak = np.max(np.abs(filtered["tau"]), axis=0)
    print("peak |tau| with configured filter [N·m]:", tau_peak)
    print("peak |ee acc| with configured filter [m/s²]:", np.max(np.abs(filtered["acc"]), axis=0))
    print("mean pose error after settling [m / rad]:", recovered_p, recovered_o)
    if unfiltered is not None:
        after_u = unfiltered["t"] >= step_time + 2.5
        rec_p_u = unfiltered["err_p"][after_u].mean(axis=0) if np.any(after_u) else unfiltered["err_p"][-1]
        rec_o_u = unfiltered["err_o"][after_u].mean(axis=0) if np.any(after_u) else unfiltered["err_o"][-1]
        tau_peak_raw = np.max(np.abs(unfiltered["tau"]), axis=0)
        print("peak |tau| without filter [N·m]:", tau_peak_raw)
        print("peak |ee acc| without filter [m/s²]:", np.max(np.abs(unfiltered["acc"]), axis=0))
        print("mean pose error without filter [m / rad]:", rec_p_u, rec_o_u)
    print("csv:", csv_path)

    plot_path = resolve_path(ROOT, log_cfg["plot"])
    pos_plot_path = resolve_path(ROOT, log_cfg.get("plot_position", "logs/experiment2_position.png"))
    torque_plot_path = resolve_path(ROOT, log_cfg.get("plot_torque", "logs/experiment2_torque.png"))
    acc_plot_path = resolve_path(ROOT, log_cfg.get("plot_acceleration", "logs/experiment2_acc.png"))
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
        axes[i].plot(filtered["t"], filtered["err_p"][:, i], label=f"filter={pose_filter}")
        if unfiltered is not None:
            axes[i].plot(unfiltered["t"], unfiltered["err_p"][:, i], linestyle="--", label="filter=1 (off)")
        axes[i].axvline(step_time, color="0.5", linestyle=":")
        axes[i].set_ylabel(f"e_{pos_labels[i]} [m]")
        axes[i].grid(True, alpha=0.3)
    for i in range(3):
        axis = axes[i + 3]
        axis.plot(filtered["t"], filtered["err_o"][:, i], label=f"filter={pose_filter}")
        if unfiltered is not None:
            axis.plot(unfiltered["t"], unfiltered["err_o"][:, i], linestyle="--", label="filter=1 (off)")
        axis.axvline(step_time, color="0.5", linestyle=":")
        axis.set_ylabel(f"e_{rot_labels[i]} [rad]")
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Experiment 2: large pose step (error to commanded target)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    print("plot:", plot_path)

    fig_pos, axes_pos = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    pos_labels = ["x", "y", "z"]
    for i, axis in enumerate(axes_pos):
        axis.plot(filtered["t"], filtered["pos"][:, i], label=f"filter={pose_filter}")
        if unfiltered is not None:
            axis.plot(unfiltered["t"], unfiltered["pos"][:, i], linestyle="--", label="filter=1 (off)")
        target_trace = np.where(filtered["t"] >= step_time, position_target[i], position_0[i])
        axis.plot(filtered["t"], target_trace, color="0.4", linestyle=":", label="target" if i == 0 else None)
        axis.axvline(step_time, color="0.5", linestyle=":")
        axis.set_ylabel(f"{pos_labels[i]} [m]")
        axis.grid(True, alpha=0.3)
    axes_pos[0].legend(loc="upper right")
    axes_pos[-1].set_xlabel("t [s]")
    fig_pos.suptitle("Experiment 2: end-effector position")
    fig_pos.tight_layout()
    fig_pos.savefig(pos_plot_path, dpi=120)
    print("position plot:", pos_plot_path)

    fig_tau, axes_tau = plt.subplots(7, 1, sharex=True, figsize=(9, 10))
    limits = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
    for i, axis in enumerate(axes_tau):
        axis.plot(filtered["t"], filtered["tau"][:, i], label=f"filter={pose_filter}")
        if unfiltered is not None:
            axis.plot(unfiltered["t"], unfiltered["tau"][:, i], linestyle="--", label="filter=1 (off)")
        axis.axvline(step_time, color="0.5", linestyle=":")
        axis.axhline(limits[i], color="0.6", linestyle=":", linewidth=0.8)
        axis.axhline(-limits[i], color="0.6", linestyle=":", linewidth=0.8)
        axis.set_ylabel(f"τ{i + 1}\n[N·m]", rotation=0, labelpad=18, va="center")
        axis.grid(True, alpha=0.3)
    axes_tau[0].legend(loc="upper right")
    axes_tau[-1].set_xlabel("t [s]")
    fig_tau.suptitle("Experiment 2: joint torques with vs without pose filter")
    fig_tau.tight_layout()
    fig_tau.savefig(torque_plot_path, dpi=120)
    print("torque plot:", torque_plot_path)

    fig_acc, axes_acc = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    acc_labels = ["ax", "ay", "az"]
    for i, axis in enumerate(axes_acc):
        axis.plot(filtered["t"], filtered["acc"][:, i], label=f"filter={pose_filter}")
        if unfiltered is not None:
            axis.plot(unfiltered["t"], unfiltered["acc"][:, i], linestyle="--", label="filter=1 (off)")
        axis.axvline(step_time, color="0.5", linestyle=":")
        axis.set_ylabel(f"{acc_labels[i]} [m/s²]")
        axis.grid(True, alpha=0.3)
    axes_acc[0].legend(loc="upper right")
    axes_acc[-1].set_xlabel("t [s]")
    fig_acc.suptitle("Experiment 2: end-effector linear acceleration")
    fig_acc.tight_layout()
    fig_acc.savefig(acc_plot_path, dpi=120)
    print("acceleration plot:", acc_plot_path)

    if not headless:
        plt.show()


def main() -> None:
    args = parse_args()
    run(args.config.resolve(), headless=args.headless)


if __name__ == "__main__":
    main()
