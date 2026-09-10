"""Overlay markers in the MuJoCo viewer."""

from __future__ import annotations

import numpy as np
import mujoco


def add_capsule(
    scene: mujoco.MjvScene,
    point_a: np.ndarray,
    point_b: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    geom = scene.geoms[scene.ngeom - 1]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(point_a, dtype=np.float64),
        np.asarray(point_b, dtype=np.float64),
    )


def draw_force_arrow(
    scene: mujoco.MjvScene,
    origin: np.ndarray,
    force: np.ndarray,
    scale: float = 0.015,
) -> None:
    if float(np.linalg.norm(force)) < 1e-9:
        return
    tip = origin + scale * np.asarray(force, dtype=float)
    add_capsule(scene, origin, tip, 0.008, np.array([1.0, 0.15, 0.1, 0.95]))


def draw_torque_arrow(
    scene: mujoco.MjvScene,
    origin: np.ndarray,
    torque: np.ndarray,
    scale: float = 0.08,
) -> None:
    if float(np.linalg.norm(torque)) < 1e-9:
        return
    tip = origin + scale * np.asarray(torque, dtype=float)
    add_capsule(scene, origin, tip, 0.006, np.array([0.2, 0.35, 1.0, 0.95]))
