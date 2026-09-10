"""Task-space Cartesian impedance: tau = J^T (-K e - D J qdot)."""

from __future__ import annotations

import numpy as np


def orientation_error(orientation_d: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Axis-angle error of orientation * orientation_d^{-1}. Quaternions are (w, x, y, z)."""
    q = orientation.copy()
    if float(np.dot(orientation_d, q)) < 0.0:
        q *= -1.0
    q_d_inv = np.array([orientation_d[0], -orientation_d[1], -orientation_d[2], -orientation_d[3]])
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = q_d_inv
    err_q = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    vec = err_q[1:]
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vec_norm, float(err_q[0]))
    return (vec / vec_norm) * angle


def diagonal_impedance(
    stiffness_trans: np.ndarray,
    stiffness_rot: np.ndarray,
    damping_zeta: np.ndarray,
    damping_trans: np.ndarray | None,
    damping_rot: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    stiffness = np.concatenate([stiffness_trans, stiffness_rot]).astype(float)
    if damping_trans is None or damping_rot is None:
        damping = 2.0 * damping_zeta * np.sqrt(np.maximum(stiffness, 0.0))
    else:
        damping = np.concatenate([damping_trans, damping_rot]).astype(float)
    return stiffness, damping


def task_command(
    jacobian: np.ndarray,
    qdot: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
    position_d: np.ndarray,
    orientation_d: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (joint torque, cartesian wrench). Wrench is force then moment."""
    error = np.zeros(6)
    error[:3] = position - position_d
    error[3:] = orientation_error(orientation_d, orientation)
    dx = jacobian @ qdot
    wrench = -stiffness * error - damping * dx
    return jacobian.T @ wrench, wrench


def damped_pseudo_inverse(matrix: np.ndarray, damp: float = 0.2) -> np.ndarray:
    """Damped SVD pseudo-inverse, same idea as src/pseudo_inversion.h."""
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    inv_singular = singular / (singular * singular + damp * damp)
    return (vh.T * inv_singular) @ u.T


def nullspace_torque(
    jacobian: np.ndarray,
    q: np.ndarray,
    qdot: np.ndarray,
    q_d: np.ndarray,
    nullspace_stiffness: float,
    nullspace_damping: float,
) -> np.ndarray:
    n_joints = jacobian.shape[1]
    jacobian_t = jacobian.T
    jacobian_t_pinv = damped_pseudo_inverse(jacobian_t)
    projector = np.eye(n_joints) - jacobian_t @ jacobian_t_pinv
    return projector @ (nullspace_stiffness * (q_d - q) - nullspace_damping * qdot)


def task_torque(
    jacobian: np.ndarray,
    qdot: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
    position_d: np.ndarray,
    orientation_d: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
) -> np.ndarray:
    tau, _wrench = task_command(
        jacobian, qdot, position, orientation, position_d, orientation_d, stiffness, damping
    )
    return tau


def filter_step(update_frequency: float, filter_percentage: float) -> float:
    """Same mapping as the C++ controller: 1.0 means no filtering."""
    if filter_percentage >= 1.0:
        return 1.0
    percentage = min(float(filter_percentage), 0.999999)
    kappa = -1.0 / np.log(1.0 - percentage)
    return 1.0 / (kappa * update_frequency + 1.0)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Intrinsic xyz RPY to (w, x, y, z)."""
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    half = 0.5 * np.array([roll, pitch, yaw])
    cr, sr = np.cos(half[0]), np.sin(half[0])
    cp, sp = np.cos(half[1]), np.sin(half[1])
    cy, sy = np.cos(half[2]), np.sin(half[2])
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_slerp(q0: np.ndarray, q1: np.ndarray, amount: float) -> np.ndarray:
    start = np.asarray(q0, dtype=float)
    end = np.asarray(q1, dtype=float)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    if dot > 0.9995:
        out = start + amount * (end - start)
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * amount
    q_perp = (end - start * dot) / sin_theta_0
    return start * np.cos(theta) + q_perp * np.sin(theta)


class PoseReference:
    """Filters a Cartesian pose target the same way as updateFilteredPose()."""

    def __init__(self, position: np.ndarray, orientation: np.ndarray, frequency_hz: float, pose_filter: float):
        self.target_position = np.asarray(position, dtype=float).copy()
        self.target_orientation = np.asarray(orientation, dtype=float).copy()
        self.position = self.target_position.copy()
        self.orientation = self.target_orientation.copy()
        self.frequency_hz = float(frequency_hz)
        self.pose_filter = float(pose_filter)

    def set_target(self, position: np.ndarray, orientation: np.ndarray) -> None:
        self.target_position = np.asarray(position, dtype=float).copy()
        self.target_orientation = np.asarray(orientation, dtype=float).copy()

    def step(self) -> None:
        if self.pose_filter >= 1.0:
            self.position = self.target_position.copy()
            self.orientation = self.target_orientation.copy()
            return
        alpha = filter_step(self.frequency_hz, self.pose_filter)
        self.position = (1.0 - alpha) * self.position + alpha * self.target_position
        self.orientation = quat_slerp(self.orientation, self.target_orientation, alpha)
