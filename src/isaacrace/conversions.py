"""Vector/quaternion conversions between ENU-FLU Isaac and NED-FRD aerospace frames.

Backend-agnostic: each function takes a NumPy array or a torch tensor and returns the
same type. The few ops that differ by name/kwarg (stack, atan2, asin, clip, cos, sin,
cross) dispatch via `_Backend`; torch is imported lazily, so the NumPy path never pulls
it in. Single source shared by the framework-free `isaacrace` core and the GPU
`isaacracelab` track. NED appears only at the observation boundary.
"""

from typing import Any

import numpy as np

# Array is a NumPy ndarray or a torch tensor. Typed `Any` — the same-type-in/-out
# contract holds by construction (each op dispatches on the input) but isn't statically
# expressible across the two backends without a heavy arithmetic Protocol.
Array = Any
SQRT_1_2 = 0.70710678118654757


class _Backend:
    """NumPy/torch op dispatch; holds the torch module only for torch inputs.

    The methods are typed `Any` (they accept either array type); the public functions
    carry the `Array` "same type in, same type out" signal.
    """

    def __init__(self, array: Any):
        """Pick torch vs NumPy from the array type (NumPy path imports no torch)."""
        self._torch = None
        if type(array).__module__.partition(".")[0] == "torch":
            import torch

            self._torch = torch

    def stack(self, arrays: list) -> Any:
        """Stack component arrays along a new trailing axis."""
        if self._torch is not None:
            return self._torch.stack(arrays, dim=-1)
        return np.stack(arrays, axis=-1)

    def atan2(self, y: Any, x: Any) -> Any:
        """Two-argument arctangent."""
        if self._torch is not None:
            return self._torch.atan2(y, x)
        return np.arctan2(y, x)

    def asin(self, x: Any) -> Any:
        """Arcsine."""
        if self._torch is not None:
            return self._torch.asin(x)
        return np.arcsin(x)

    def clip(self, x: Any, lo: float, hi: float) -> Any:
        """Clamp to [lo, hi]."""
        if self._torch is not None:
            return self._torch.clamp(x, lo, hi)
        return np.clip(x, lo, hi)

    def cos(self, x: Any) -> Any:
        """Cosine."""
        if self._torch is not None:
            return self._torch.cos(x)
        return np.cos(x)

    def sin(self, x: Any) -> Any:
        """Sine."""
        if self._torch is not None:
            return self._torch.sin(x)
        return np.sin(x)

    def cross(self, a: Any, b: Any) -> Any:
        """Cross product along the trailing axis."""
        if self._torch is not None:
            return self._torch.cross(a, b, dim=-1)
        return np.cross(a, b, axis=-1)


def vec_enu_ned(vector: Array) -> Array:
    """Convert a vector from ENU (East-North-Up) to NED (North-East-Down) frame.

    Args:
        vector: A 3D vector (..., 3), NumPy array or torch tensor.

    Returns:
        The corresponding vector in NED, same type as the input.
    """
    if vector.shape[-1] != 3:
        raise ValueError("Input vector must be a 3D vector.")
    return _Backend(vector).stack([vector[..., 1], vector[..., 0], -vector[..., 2]])


def vec_flu_frd(vector: Array) -> Array:
    """Convert a vector from FLU (Forward-Left-Up) to FRD (Forward-Right-Down) frame.

    Args:
        vector: A 3D vector (..., 3), NumPy array or torch tensor.

    Returns:
        The corresponding vector in FRD, same type as the input.
    """
    if vector.shape[-1] != 3:
        raise ValueError("Input vector must be a 3D vector.")
    return _Backend(vector).stack([vector[..., 0], -vector[..., 1], -vector[..., 2]])


def quat_aero_isaac(quaternion: Array) -> Array:
    """Swap a quaternion between aerospace (FRD->NED) and Isaac (FLU->ENU) conventions.

    The two conventions are 180° apart; applying this twice is the identity (both ways).

    Args:
        quaternion: A quaternion [x, y, z, w] (..., 4), NumPy array or torch tensor.

    Returns:
        The quaternion in the other convention [x, y, z, w], same type as the input.
    """
    if quaternion.shape[-1] != 4:
        raise ValueError("Input quaternion must have 4 components (x, y, z, w).")
    x = quaternion[..., 0]
    y = quaternion[..., 1]
    z = quaternion[..., 2]
    w = quaternion[..., 3]
    return _Backend(quaternion).stack([
        -(SQRT_1_2 * (x + y)),
        -(SQRT_1_2 * (x - y)),
        (SQRT_1_2 * (z - w)),
        -(SQRT_1_2 * (z + w)),
    ])


def quat_wxyz_to_xyzw(q_wxyz: Array) -> Array:
    """Reorder a (w, x, y, z) quaternion (Isaac articulation) to (x, y, z, w)."""
    return q_wxyz[..., [1, 2, 3, 0]]


def quat_xyzw_to_euler_xyz(q: Array) -> Array:
    """Intrinsic XYZ (roll, pitch, yaw) from an (x, y, z, w) quat; matches scipy 'xyz'.

    Standard aerospace RPY; verified vs scipy ``as_euler('xyz')`` in test_parity.py.
    """
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]
    xp = _Backend(q)
    roll = xp.atan2(2.0 * (x * w + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = xp.asin(xp.clip(2.0 * (y * w - x * z), -1.0, 1.0))
    yaw = xp.atan2(2.0 * (z * w + x * y), 1.0 - 2.0 * (y * y + z * z))
    return xp.stack([roll, pitch, yaw])


def euler_xyz_to_quat_xyzw(e: Array) -> Array:
    """Inverse of quat_xyzw_to_euler_xyz; matches scipy from_euler('xyz').as_quat()."""
    r = e[..., 0] * 0.5
    p = e[..., 1] * 0.5
    y = e[..., 2] * 0.5
    xp = _Backend(e)
    cr, sr = xp.cos(r), xp.sin(r)
    cp, sp = xp.cos(p), xp.sin(p)
    cy, sy = xp.cos(y), xp.sin(y)
    return xp.stack([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def quat_rotate_xyzw(q: Array, v: Array) -> Array:
    """Rotate vector v by quaternion q=(x, y, z, w): v' = R(q) v. Batch-aware."""
    qv = q[..., :3]
    qw = q[..., 3:4]
    xp = _Backend(q)
    t = 2.0 * xp.cross(qv, v)
    return v + qw * t + xp.cross(qv, t)
