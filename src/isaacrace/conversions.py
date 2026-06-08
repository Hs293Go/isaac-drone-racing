"""Vector/quaternion conversions between ENU-FLU Isaac and NED-FRD aerospace frames."""

import numpy as np
from numpy.typing import NDArray


def vec_enu_ned(vector: NDArray) -> NDArray:
    """Convert a vector from ENU (East-North-Up) to NED (North-East-Down) frame.

    Args:
        vector: A 3D vector in ENU coordinates.

    Returns:
        The corresponding vector in NED coordinates.
    """
    if vector.shape[-1] != 3:
        raise ValueError("Input vector must be a 3D vector.")

    # ENU to NED conversion
    return np.stack([vector[..., 1], vector[..., 0], -vector[..., 2]], axis=-1)


def vec_flu_frd(vector: NDArray) -> NDArray:
    """Convert a vector from FLU (Forward-Left-Up) to FRD (Forward-Right-Down) frame.

    Args:
        vector: A 3D vector in FLU coordinates.

    Returns:
        The corresponding vector in FRD coordinates.
    """
    if vector.shape[-1] != 3:
        raise ValueError("Input vector must be a 3D vector.")

    # FLU to FRD conversion
    return np.stack([vector[..., 0], -vector[..., 1], -vector[..., 2]], axis=-1)


def quat_aero_isaac(quaternion: NDArray) -> NDArray:
    """Convert a quaternion between conventions.

    Convert a quaternion between aerospace (FRD body to NED world) and Isaac (FLU
    body to ENU world) conventions.

    The two conventions are 180 degrees apart, so two applications of this
    function will yield the original result and this function appropriately
    interconverts between both conventions.

    Args:
        quaternion: A quaternion, represented as [x, y, z, w].

    Returns:
        The corresponding quaternion in the other convention, represented
        as [x, y, z, w].
    """
    if quaternion.shape[-1] != 4:
        raise ValueError("Input quaternion must have 4 components (x, y, z, w).")

    x = quaternion[..., 0]
    y = quaternion[..., 1]
    z = quaternion[..., 2]
    w = quaternion[..., 3]

    # ENU to NED conversion for quaternions
    SQRT_1_2 = 0.70710678118654757

    return np.stack(
        [
            -(SQRT_1_2 * (x + y)),
            -(SQRT_1_2 * (x - y)),
            (SQRT_1_2 * (z - w)),
            -(SQRT_1_2 * (z + w)),
        ],
        axis=-1,
    )
