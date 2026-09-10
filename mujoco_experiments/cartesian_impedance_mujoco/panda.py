from __future__ import annotations

import numpy as np
import mujoco


def site_quat(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return quat


def site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack([jacp, jacr])


def apply_site_wrench(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    force_world: np.ndarray,
    torque_world: np.ndarray,
) -> None:
    body_id = int(model.site_bodyid[site_id])
    point = np.array(data.site_xpos[site_id], dtype=np.float64)
    qfrc = np.zeros(model.nv)
    mujoco.mj_applyFT(
        model,
        data,
        np.asarray(force_world, dtype=np.float64),
        np.asarray(torque_world, dtype=np.float64),
        point,
        body_id,
        qfrc,
    )
    data.qfrc_applied[:] = qfrc


def set_mocap_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    position: np.ndarray,
    orientation: np.ndarray,
) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        return
    data.mocap_pos[mocap_id] = position
    data.mocap_quat[mocap_id] = orientation


def site_linear_acc(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    """World-frame linear acceleration of a site (after mj_step / mj_forward)."""
    spatial = np.zeros(6)
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, spatial, 0)
    return spatial[3:6]


def collision_xmax(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    skip_names: set[str],
) -> float:
    """Furthest +x extent of colliding geoms (bounding sphere, conservative)."""
    xmax = -1e9
    for geom_id in range(model.ngeom):
        if int(model.geom_contype[geom_id]) == 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name in skip_names:
            continue
        xpos = float(data.geom_xpos[geom_id][0])
        radius = float(model.geom_rbound[geom_id])
        xmax = max(xmax, xpos + radius)
    return xmax


def place_front_wall(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_name: str,
    ee_pos: np.ndarray,
    gap: float,
    size: np.ndarray,
) -> float:
    """World-fixed box; inner face is `gap` past the arm along +x. Returns inner-face x."""
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise RuntimeError(f"geom not found: {geom_name}")
    model.geom_size[geom_id] = size
    arm_xmax = collision_xmax(model, data, skip_names={geom_name, "floor"})
    inner_x = max(float(ee_pos[0]), arm_xmax) + gap
    half_x = float(size[0])
    model.geom_pos[geom_id, 0] = inner_x + half_x
    model.geom_pos[geom_id, 1] = float(ee_pos[1])
    model.geom_pos[geom_id, 2] = float(ee_pos[2])
    mujoco.mj_forward(model, data)
    return inner_x


def geom_in_contact(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str) -> bool:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    for i in range(data.ncon):
        contact = data.contact[i]
        if int(contact.geom1) == geom_id or int(contact.geom2) == geom_id:
            return True
    return False


def geom_contact_force_world(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str) -> np.ndarray:
    """Contact force on this geom, world frame (normal + tangents)."""
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    total = np.zeros(3)
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 != geom_id and geom2 != geom_id:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        normal = np.array(contact.frame[0:3])
        tangent1 = np.array(contact.frame[3:6])
        tangent2 = np.array(contact.frame[6:9])
        force = wrench[0] * normal + wrench[1] * tangent1 + wrench[2] * tangent2
        # mj_contactForce is on geom1; flip if the named geom is geom2
        if geom2 == geom_id:
            force = -force
        total += force
    return total
