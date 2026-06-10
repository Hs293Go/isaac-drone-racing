"""Pose-derived reward-shaping signals (FPV gate visibility + flight quality).

The course-independent reward terms: ``gate_bearing`` / ``visibility_reward`` keep the
target gate in the FPV camera's FOV, and ``backpedal`` penalizes tail-first flight
(together, the perception-aware shaping).

These functions are pure NumPy/scipy (no Isaac Sim) and work in NED-FRD, matching the
env's native state, for easy integration into the RL reward.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from isaacrace.config import FpvConfig

_EPS_SENTINEL = 1e-9  # Sentinel to prevent divide-by-zero; not a tuning parameter.


def visibility_reward(cos_angle: ArrayLike, fov_deg: float) -> NDArray:
    """FOV-gated visibility reward in [0, 1] (scalar or batched cos_angle).

    This function is 1 when the gate is centered on the optical axis, ramping linearly
    to 0 at the FOV edge, and 0 once the gate leaves the FOV. It implements:
    ``r = relu(cos_angle - cos(fov/2)) / (1 - cos(fov/2))``.

    Args:
        cos_angle: Scalar or (N,) array of the cosine of the angle between the FPV
          optical axis and the line of sight to the gate.
        fov_deg: FPV camera horizontal field of view in degrees.

    Returns:
        Scalar of (N,) a reward in [0, 1] that is 1 when the gate is centered and 0 when
        the gate leaves the FOV.
    """
    cos_angle = np.asarray(cos_angle, dtype=np.double)
    cos_half = float(np.cos(np.radians(fov_deg) / 2.0))
    return np.maximum(0.0, cos_angle - cos_half) / max(_EPS_SENTINEL, 1.0 - cos_half)


def _optical_axis_frd(tilt_deg: float) -> np.ndarray:
    """FPV viewing direction in the FRD body frame (forward, pitched up by tilt)."""
    t = np.radians(tilt_deg)
    return np.array([np.cos(t), 0.0, -np.sin(t)])  # FLU [c,0,s] -> FRD [c,0,-s]


def gate_bearing(
    pos_ned: ArrayLike, euler_ned: ArrayLike, gate_ned: ArrayLike, fpv: FpvConfig
) -> NDArray:
    """The cos of the angle between the FPV optical axis and the LOS to the gate.

    Operates in NED-FRD coordinates.

    Args:
        pos_ned: (,3) or (N,3) position of the drone in NED.
        euler_ned: (,3) or (N,3) Euler angles (roll,pitch,yaw)
        gate_ned: (,3) or (N,3) position of the gate in NED.
        fpv: The FPV camera config, which includes the tilt angle and mount position.

    Returns:
        Scalar or (N,) array of the cosine of the angle between the FPV optical axis and
        the LOS to the gate
    """
    rot = Rotation.from_euler("xyz", np.asarray(euler_ned))  # body(FRD)->world(NED)
    mount = np.asarray(fpv.mount, float)
    mount_frd = np.array([mount[0], -mount[1], -mount[2]])  # FLU -> FRD
    cam = np.asarray(pos_ned, float) + rot.apply(mount_frd)  # (N,3)
    los = np.asarray(gate_ned, float) - cam
    dist = np.linalg.norm(los, axis=-1, keepdims=True)
    los_body = rot.inv().apply(los) / np.maximum(dist, _EPS_SENTINEL)
    return los_body @ _optical_axis_frd(fpv.tilt_deg)  # (N,)


def backpedal(vel_ned: ArrayLike, euler_ned: ArrayLike) -> NDArray:
    """Tail-first penalty: relu(-cos) between velocity and the body nose (NED).

    0 when the nose leads the velocity (flying forward), up to 1 when fully
    tail-first. Frame-invariant, so it reads the env's native NED state directly.
    Batch-aware (scalar or leading (N,)).

    Args:
        vel_ned: The (n_batch x 3) NED velocity of the drone.
        euler_ned: The (n_batch x 3) aerospace (ZYX) euler angles of the drone.

    Returns:
        A scalar or (n_batch,) penalty in [0, 1].
    """
    vel = np.asarray(vel_ned, float)
    euler = np.asarray(euler_ned, float)
    theta, psi = euler[..., 1], euler[..., 2]
    # body x-axis in NED = first column of Rz(psi) Ry(theta) Rx(phi)
    ct = np.cos(theta)
    fwd = np.stack([ct * np.cos(psi), ct * np.sin(psi), -np.sin(theta)], axis=-1)
    speed = np.linalg.norm(vel, axis=-1)
    cos = np.sum(vel * fwd, axis=-1) / np.maximum(speed, _EPS_SENTINEL)
    return np.maximum(0.0, -cos)
