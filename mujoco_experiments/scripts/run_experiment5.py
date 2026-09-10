#!/usr/bin/env python3
"""Experiment 5: wall contact with Fz feedforward and near-zero Kz."""

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
from cartesian_impedance_mujoco.controller import PoseReference, diagonal_impedance, task_command
from cartesian_impedance_mujoco.panda import (
    geom_contact_force_world,
    geom_in_contact,
    place_front_wall,
    set_mocap_pose,
    site_jacobian,
    site_quat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "experiment5.yaml")
    parser.add_argument("--headless", action="store_true")
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
    wall_cfg = cfg["wall"]
    ff = cfg["feedforward"]
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
    wall_inner_x = place_front_wall(
        model,
        data,
        "wall",
        position_0,
        float(wall_cfg["gap"]),
        _as_vec(wall_cfg["size"], 3),
    )
    position_target = position_0.copy()
    position_target[0] = wall_inner_x + float(experiment["penetrate"])

    stiffness, damping = diagonal_impedance(
        _as_vec(impedance["stiffness_trans"], 3),
        _as_vec(impedance["stiffness_rot"], 3),
        _as_vec(impedance["damping_zeta"], 6),
        None if impedance.get("damping_trans") is None else _as_vec(impedance["damping_trans"], 3),
        None if impedance.get("damping_rot") is None else _as_vec(impedance["damping_rot"], 3),
    )
    w_ff = np.zeros(6)
    w_ff[:3] = _as_vec(ff["force"], 3)
    w_ff[3:] = _as_vec(ff.get("torque", [0.0, 0.0, 0.0]), 3)
    ff_start = float(ff.get("start_time", experiment["step_time"]))

    duration = float(sim["duration"])
    dt = float(model.opt.timestep)
    step_time = float(experiment["step_time"])
    approach_duration = float(experiment.get("approach_duration", 0.0))
    ff_ramp = float(ff.get("ramp_time", 0.0))
    pose_filter = float(cfg.get("filtering", {}).get("pose", 1.0))
    pose_ref = PoseReference(position_0, orientation_0, 1.0 / dt, pose_filter)
    use_viewer = bool(sim.get("viewer", True)) and not headless
    realtime = float(sim.get("realtime", 1.0))
    sync_every = max(1, int(round(0.016 / dt)))

    print("wall inner x [m]:", wall_inner_x, "desired x [m]:", position_target[0])
    print("K_trans:", stiffness[:3], "D_trans:", damping[:3])
    print("F_ff [N]:", w_ff[:3], "from t=", ff_start, "ramp", ff_ramp)
    print("approach_duration [s]:", approach_duration)

    viewer = None
    if use_viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.2
        viewer.cam.lookat[:] = [0.35, 0.0, 0.45]
        print("窗口：墙 + 黄球在墙内。Z 向前馈 10 N，Kz≈0。")

    stepped = False
    rows: list[list[float]] = []
    t = 0.0
    step = 0
    wall_clock0 = time.perf_counter()
    last_print = -1.0
    try:
        while t < duration + 0.5 * dt:
            if viewer is not None and not viewer.is_running():
                break
            mujoco.mj_forward(model, data)
            if approach_duration > 0.0:
                alpha = float(np.clip((t - step_time) / approach_duration, 0.0, 1.0))
                pose_now = (1.0 - alpha) * position_0 + alpha * position_target
                pose_ref.set_target(pose_now, orientation_0)
            elif (not stepped) and t >= step_time:
                pose_ref.set_target(position_target, orientation_0)
                stepped = True
            pose_ref.step()
            set_mocap_pose(model, data, "target_marker", pose_ref.target_position, pose_ref.target_orientation)
            set_mocap_pose(model, data, "ref_marker", pose_ref.position, pose_ref.orientation)

            dq = np.array(data.qvel[:7])
            position = np.array(data.site_xpos[site_id])
            orientation = site_quat(model, data, site_id)
            jacobian = site_jacobian(model, data, site_id)[:, :7]
            tau_task, _wrench = task_command(
                jacobian,
                dq,
                position,
                orientation,
                pose_ref.position,
                pose_ref.orientation,
                stiffness,
                damping,
            )
            wrench_ff = np.zeros(6)
            if t >= ff_start:
                scale = 1.0 if ff_ramp <= 0.0 else float(np.clip((t - ff_start) / ff_ramp, 0.0, 1.0))
                wrench_ff = scale * w_ff
            data.ctrl[:7] = tau_task + jacobian.T @ wrench_ff + np.array(data.qfrc_bias[:7])
            data.qfrc_applied[:] = 0.0
            mujoco.mj_step(model, data)
            f_wall = geom_contact_force_world(model, data, "wall")
            f_robot = -f_wall
            contacting = geom_in_contact(model, data, "wall")
            dx = jacobian @ dq
            rows.append(
                [
                    t,
                    *f_robot.tolist(),
                    float(np.linalg.norm(f_robot)),
                    float(contacting),
                    position[2],
                    float(dx[0]),
                    float(dx[2]),
                ]
            )
            t = float(data.time)
            step += 1
            if t - last_print >= 0.5:
                last_print = t
                print(
                    f"t={t:5.2f}s  contact={contacting}  F_robot=[{f_robot[0]:6.2f}, {f_robot[1]:6.2f}, {f_robot[2]:6.2f}]  "
                    f"|F|={np.linalg.norm(f_robot):6.2f}  z={position[2]:.3f}"
                )
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
    force = table[:, 1:4]
    force_norm = table[:, 4]
    contacting = table[:, 5] > 0.5
    vx = table[:, 7]
    vz = table[:, 8]
    first_hit = np.argmax(contacting) if np.any(contacting) else None
    if first_hit is not None and contacting[first_hit]:
        print(
            f"impact t={times[first_hit]:.3f}s  vx={vx[first_hit]:.4f} m/s  vz={vz[first_hit]:.4f} m/s"
        )
    print("peak Fx, Fy, Fz, |F| [N]:", np.max(np.abs(force), axis=0), float(np.max(force_norm)))
    after = times >= max(ff_start + 2.0, (times[first_hit] + 1.5 if first_hit is not None else ff_start + 2.0))
    if np.any(after):
        print("steady |contact force on robot| [N]:", float(np.mean(force_norm[after])))
        print("steady Fxyz [N]:", force[after].mean(axis=0))

    plot_path = resolve_path(ROOT, log_cfg.get("plot_contact", "logs/experiment5_contact.png"))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        if headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skip plot")
        return

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
    labels = ["Fx [N]", "Fy [N]", "Fz [N]", "|F| [N]"]
    series = [force[:, 0], force[:, 1], force[:, 2], force_norm]
    for i, axis in enumerate(axes):
        axis.plot(times, series[i])
        axis.axvline(step_time, color="0.5", linestyle=":")
        if i == 2:
            axis.axhline(w_ff[2], color="C1", linestyle="--", label="Fz feedforward")
            axis.legend(loc="upper right")
        axis.set_ylabel(labels[i])
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Experiment 5: contact force on the robot from the wall")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    print("contact plot:", plot_path)
    if not headless:
        plt.show()


def main() -> None:
    args = parse_args()
    run(args.config.resolve(), headless=args.headless)


if __name__ == "__main__":
    main()
