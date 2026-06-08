"""Faithful port of the optimal_quad_control_RL (OQCRL) 5-inch-quad dynamics.

Faithful port of the OQCRL 5-inch-quad dynamics + motor model, as pure NumPy — so
the Isaac Sim env can reproduce the OQCRL drone the trained policy was trained on.
(The racecourse and the gate-relative observation live in course.py; frame
conversions in conversions.py.).

OQCRL conventions: world = NED (z down), body = FRD (x fwd, y right, z down),
euler = aerospace ZYX (phi,theta,psi). Isaac/Pegasus: world = ENU, body = FLU;
env.py handles the ENU↔NED / FLU↔FRD swaps.

The Isaac env keeps PhysX as the integrator: each control step it (1) advances the
motor RPM state with OQCRL's first-order lag, (2) computes OQCRL's body-frame
thrust/drag force and body angular acceleration (below), (3) applies
force = m·accel to the PhysX body (which adds gravity) and integrates the body
rates kinematically. So m cancels and the realized accelerations match OQCRL.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


class QuadParams(NamedTuple):
    """Quadrotor parameters governing the motor-effectiveness and drag model.

    While optimal_quad_control_rl uses a 23-vector of parameters, we wrap it in a
    NamedTuple for clarity so we don't need to maintain a separate list of names. The
    NamedTuple converts to a np.array trivially.
    """

    k_x: float
    k_y: float
    k_w: float
    k_p1: float
    k_p2: float
    k_p3: float
    k_p4: float
    k_q1: float
    k_q2: float
    k_q3: float
    k_q4: float
    k_r1: float
    k_r2: float
    k_r3: float
    k_r4: float
    k_r5: float
    k_r6: float
    k_r7: float
    k_r8: float
    tau: float
    k: float
    w_min: float
    w_max: float

    def scaled(self, factor: ArrayLike) -> "QuadParams":
        """Element-wise scale (domain randomization / tests) -> a new QuadParams."""
        return QuadParams._make(np.asarray(self) * np.asarray(factor))

    def randomized(self, rng: np.random.Generator, pct: float) -> "QuadParams":
        """Per-field multiplicative domain randomization.

        Each parameter is scaled by U(1-pct, 1+pct); ``k`` is then clamped to <=1 (it
        is a [0,1] fraction in the motor model).

        Args:
            rng: A random number generator to use for sampling the uniform noise.
            pct: The percentage for uniform multiplicative noise, e.g. 0.1 for ±10

        Returns:
            A new QuadParams instance with randomized parameters.
        """
        p = self.scaled(rng.uniform(1.0 - pct, 1.0 + pct, len(self)))
        return p._replace(k=min(float(p.k), 1.0))


# 5-inch nominal values from optimal_quad_control_rl/randomization.py (params_5inch).
PARAMS_5INCH = QuadParams(
    k_w=2.49e-06,
    k_x=4.85e-05,
    k_y=7.28e-05,
    k_p1=6.55e-05,
    k_p2=6.61e-05,
    k_p3=6.36e-05,
    k_p4=6.67e-05,
    k_q1=5.28e-05,
    k_q2=5.86e-05,
    k_q3=5.05e-05,
    k_q4=5.89e-05,
    k_r1=1.07e-02,
    k_r2=1.07e-02,
    k_r3=1.07e-02,
    k_r4=1.07e-02,
    k_r5=1.97e-03,
    k_r6=1.97e-03,
    k_r7=1.97e-03,
    k_r8=1.97e-03,
    w_min=238.49,
    w_max=3295.50,
    k=0.95,
    tau=0.04,
)
G = 9.81
W_MIN_N, W_MAX_N = 0.0, 3000.0  # normalization range for the motor-state -> rad/s map
DT = 0.01  # 100 Hz, OQCRL integration step


def _euler_kinematic_rates(angles: NDArray, body_rates: NDArray) -> NDArray:
    """Strapdown navigation equations for ZYX Euler angles and body rates."""
    phi, theta, _ = angles
    cph, sph = np.cos(phi), np.sin(phi)
    cth, tth = np.cos(theta), np.tan(theta)
    p, q, r = body_rates
    return np.array([
        p + q * sph * tth + r * cph * tth,
        q * cph - r * sph,
        q * sph / cth + r * cph / cth,
    ])


def unproject_motor(w: NDArray) -> NDArray:
    """Unprojects motor state `w` in [-1,1] to actual rad/s in [w_min_n, w_max_n]."""
    return (w + 1.0) / 2.0 * (W_MAX_N - W_MIN_N) + W_MIN_N


def motor_derivative(w: NDArray, u: NDArray, p: QuadParams):
    """Time-derivative of the *normalized* motor state (first-order lag)."""
    p = QuadParams._make(p)
    W = unproject_motor(w)

    # Unproject command u in [-1,1] to U in [0,1] (throttle fraction)
    U = (np.asarray(u) + 1.0) / 2.0
    # Compute steady-state target rad/s from command u
    Wc = (p.w_max - p.w_min) * np.sqrt(p.k * U**2 + (1.0 - p.k) * U) + p.w_min
    # Apply first-order lag
    d_W = (Wc - W) / p.tau
    return d_W / (W_MAX_N - W_MIN_N) * 2.0, d_W


def body_force_torque_accel(w: NDArray, u: NDArray, vb, p):
    """OQCRL thrust+drag force (FRD body, as accel) and body angular acceleration.

    Returns (force_frd[3], torque_frd[3]) as accelerations (m/s^2, rad/s^2). vb is
    the FRD body-frame velocity. The Isaac env multiplies these by mass / inertia
    for PhysX.
    """
    p = QuadParams._make(p)
    W = unproject_motor(w)
    _, d_W = motor_derivative(w, u, p)
    W_sum = W.sum()
    Wsq = W**2

    force = np.array([
        -p.k_x * vb[0] * W_sum,
        -p.k_y * vb[1] * W_sum,
        -p.k_w * Wsq.sum(),
    ])
    torque = np.array([
        -p.k_p1 * Wsq[0] - p.k_p2 * Wsq[1] + p.k_p3 * Wsq[2] + p.k_p4 * Wsq[3],
        -p.k_q1 * Wsq[0] + p.k_q2 * Wsq[1] - p.k_q3 * Wsq[2] + p.k_q4 * Wsq[3],
        (
            -p.k_r1 * W[0]
            + p.k_r2 * W[1]
            + p.k_r3 * W[2]
            - p.k_r4 * W[3]
            - p.k_r5 * d_W[0]
            + p.k_r6 * d_W[1]
            + p.k_r7 * d_W[2]
            - p.k_r8 * d_W[3]
        ),
    ])
    return force, torque


def f_func(state, u, p):
    """Full OQCRL state derivative (16-dim, NED-FRD), for validation.

    For validation against the golden dump.
    state = [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r, w1,w2,w3,w4]
    (NED world / normalized motors).
    """
    vel = state[3:6]
    angles = state[6:9]
    body_rates = state[9:12]
    w = state[12:16]
    rot = Rotation.from_euler("xyz", angles)
    vb = rot.inv().apply(vel)
    force, torque = body_force_torque_accel(w, u, vb, p)
    d_w, _ = motor_derivative(w, u, p)
    out = np.empty(16)
    out[0:3] = vel
    out[3:6] = rot.apply(force)
    out[5] += G
    out[6:9] = _euler_kinematic_rates(angles, body_rates)
    out[9:12] = torque
    out[12:16] = d_w
    return out
