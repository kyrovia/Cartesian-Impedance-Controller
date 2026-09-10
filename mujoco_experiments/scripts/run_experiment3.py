#!/usr/bin/env python3
"""Experiment 3: command into a wall; compare impedance vs position control."""

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
    geom_in_contact,
    place_front_wall,
    set_mocap_pose,
    site_jacobian,
    site_quat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "experiment3.yaml")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def _as_vec(value, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size:
        raise ValueError(f"expected length {size}, got {array.size}")
    return array


def _gains(block: dict) -> tuple[np.ndarray, np.ndarray]:
    return diagonal_impedance(
        _as_vec(block["stiffness_trans"], 3),
        _as_vec(block["stiffness_rot"], 3),
        _as_vec(block["damping_zeta"], 6),
        None if block.get("damping_trans") is None else _as_vec(block["damping_trans"], 3),
        None if block.get("damping_rot") is None else _as_vec(block["damping_rot"], 3),
    )


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
    orientation_0: np.ndarray,
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
    pose_ref = PoseReference(position_0, orientation_0, 1.0 / dt, pose_filter)
    stepped = False
    rows: list[list[float]] = []
    t = 0.0
    step = 0
    wall_clock0 = time.perf_counter()
    sync_every = max(1, int(round(0.016 / dt)))
    last_print = -1.0
    hit_wall = False

    while t < duration + 0.5 * dt:
        if viewer is not None and not viewer.is_running():
            break
        mujoco.mj_forward(model, data)
        if (not stepped) and t >= step_time:
            pose_ref.set_target(position_target, orientation_0)
            stepped = True
        pose_ref.step()
        set_mocap_pose(model, data, "target_marker", pose_ref.target_position, pose_ref.target_orientation)
        set_mocap_pose(model, data, "ref_marker", pose_ref.position, pose_ref.orientation)

        dq = np.array(data.qvel[:7])
        position = np.array(data.site_xpos[site_id])
        orientation = site_quat(model, data, site_id)
        jacobian = site_jacobian(model, data, site_id)[:, :7]
        tau_task, wrench = task_command(
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
        mujoco.mj_step(model, data)
        contacting = geom_in_contact(model, data, "wall")
        hit_wall = hit_wall or contacting
        rows.append([t, *wrench.tolist(), *np.array(data.ctrl[:7], dtype=float).tolist(), float(contacting)])
        t = float(data.time)
        step += 1
        if t - last_print >= 0.5:
            last_print = t
            print(
                f"t={t:5.2f}s  ee_x={position[0]:.4f}  desired_x={pose_ref.target_position[0]:.4f}  "
                f"mx={wrench[3]:.2f} my={wrench[4]:.2f} mz={wrench[5]:.2f}  contact={contacting}"
            )
        if viewer is not None and step % sync_every == 0:
            viewer.sync()
            if realtime > 0.0:
                lag = t / realtime - (time.perf_counter() - wall_clock0)
                if lag > 0.0:
                    time.sleep(lag)

    table = np.asarray(rows, dtype=float)
    return {
        "t": table[:, 0],
        "wrench": table[:, 1:7],
        "tau": table[:, 7:14],
        "hit_wall": hit_wall,
    }


def run(config_path: Path, *, headless: bool = False) -> None:
    import mujoco

    cfg = load_yaml(config_path)
    sim = cfg["sim"]
    robot = cfg["robot"]
    experiment = cfg["experiment"]
    wall_cfg = cfg["wall"]
    control = cfg.get("control") or {}
    mode = str(control.get("mode", "impedance")).lower()
    if mode not in {"impedance", "position"}:
        raise ValueError("control.mode must be impedance or position")
    compare = bool(control.get("compare", False))

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
    penetrate = float(experiment["penetrate"])
    position_target = position_0.copy()
    position_target[0] = wall_inner_x + penetrate

    gains = {
        "impedance": _gains(cfg["impedance"]),
        "position": _gains(cfg["position"]),
    }
    duration = float(sim["duration"])
    dt = float(model.opt.timestep)
    step_time = float(experiment["step_time"])
    pose_filter = float(cfg.get("filtering", {}).get("pose", 1.0))
    use_viewer = bool(sim.get("viewer", True)) and not headless
    realtime = float(sim.get("realtime", 1.0))

    print("ee start x [m]:", position_0[0])
    print("wall inner face x [m]:", wall_inner_x)
    print("desired x inside wall [m]:", position_target[0], "(penetrate", penetrate, "m)")
    print("control mode:", mode, "compare:", compare)

    viewer = None
    if use_viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.2
        viewer.cam.lookat[:] = [0.35, 0.0, 0.45]
        print("窗口：半透明墙，黄球=墙内期望，绿球=参考，红点=末端。")

    try:
        primary = _simulate(
            model=model,
            data=data,
            site_id=site_id,
            home_q=home_q,
            stiffness=gains[mode][0],
            damping=gains[mode][1],
            pose_filter=pose_filter,
            step_time=step_time,
            position_target=position_target,
            orientation_0=orientation_0,
            duration=duration,
            dt=dt,
            viewer=viewer,
            realtime=realtime,
        )
    finally:
        if viewer is not None:
            viewer.close()

    other_mode = "position" if mode == "impedance" else "impedance"
    other = None
    if compare:
        print(f"running {other_mode} for overlay (no viewer)")
        other = _simulate(
            model=model,
            data=data,
            site_id=site_id,
            home_q=home_q,
            stiffness=gains[other_mode][0],
            damping=gains[other_mode][1],
            pose_filter=pose_filter,
            step_time=step_time,
            position_target=position_target,
            orientation_0=orientation_0,
            duration=duration,
            dt=dt,
            viewer=None,
            realtime=0.0,
        )

    print("hit wall:", primary["hit_wall"])
    peak_m = np.max(np.abs(primary["wrench"][:, 3:]), axis=0)
    print(f"peak |ee moment| ({mode}) [N·m]:", peak_m)
    if other is not None:
        print(f"peak |ee moment| ({other_mode}) [N·m]:", np.max(np.abs(other["wrench"][:, 3:]), axis=0))

    log_cfg = cfg.get("log") or {}
    plot_path = resolve_path(ROOT, log_cfg.get("plot_wrench", "logs/experiment3_wrench.png"))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        if headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skip plot")
        return

    labels = ["fx [N]", "fy [N]", "fz [N]", "mx [N·m]", "my [N·m]", "mz [N·m]"]
    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(8, 10))
    for i, axis in enumerate(axes):
        axis.plot(primary["t"], primary["wrench"][:, i], label=mode)
        if other is not None:
            axis.plot(other["t"], other["wrench"][:, i], linestyle="--", label=other_mode)
        axis.axvline(step_time, color="0.5", linestyle=":")
        axis.set_ylabel(labels[i])
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Experiment 3: commanded end-effector wrench (into wall)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    print("wrench plot:", plot_path)
    if not headless:
        plt.show()


def main() -> None:
    args = parse_args()
    run(args.config.resolve(), headless=args.headless)


if __name__ == "__main__":
    main()
