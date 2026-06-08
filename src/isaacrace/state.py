"""State: rigid-body state of the drone.

This is structurally equivalent to Pegasus's `pegasus/simulator/logic/state.py` in terms
of field naming and semantics. However, we switch to a dataclass and we treat it as a
pure data container.
"""

import dataclasses

import numpy as np


@dataclasses.dataclass
class State:
    """Stores the state of a given vehicle.

    Note:
        - position - A numpy array with the [x,y,z] of the vehicle expressed in the
          inertial frame according to an ENU convention.
        - orientation - A numpy array with the quaternion [qx, qy, qz, qw] that encodes
          the attitude of the vehicle's FLU body frame, relative to an ENU inertial
          frame, expressed in the ENU inertial frame.
        - linear_velocity - A numpy array with [vx,vy,vz] that defines the velocity of
          the vehicle expressed in the inertial frame according to an ENU convention.
        - linear_body_velocity - A numpy array with [u,v,w] that defines the velocity of
          the vehicle expressed in the FLU body frame.
        - angular_velocity - A numpy array with [p,q,r] with the angular velocity of the
          vehicle's FLU body frame, relative to an ENU inertial frame, expressed in the
          FLU body frame.
        - linear acceleration - An array with [x_ddot, y_ddot, z_ddot] with the
          acceleration of the vehicle expressed in the inertial frame according to an
          ENU convention.
    """

    # The position [x,y,z] of the vehicle's body frame relative to the inertial frame,
    # expressed in the inertial frame
    position: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))

    # The attitude (orientation) of the vehicle's body frame relative to the inertial
    # frame of reference, expressed in the inertial frame. This quaternion should follow
    # the convention [qx, qy, qz, qw], such that "no rotation" equates to the
    # quaternion=[0, 0, 0, 1]
    attitude: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0])
    )

    # The linear velocity [x_dot, y_dot, z_dot] of the vehicle's body frame expressed in
    # the inertial frame of reference
    linear_velocity: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))

    # The angular velocity [wx, wy, wz] of the vehicle's body frame relative to the
    # inertial frame, expressed in the body frame
    angular_velocity: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(3)
    )
