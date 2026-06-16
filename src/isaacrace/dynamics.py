"""Pure numpy port of optimal_quad_control_rl's quadrotor dynamics.

The quadrotor dynamics is parameterized by 23 parameters governing the
motor-effectiveness and drag model. The drag model influences translational dynamics by
introducing a velocity-dependent rotor drag on the body frame x/y axes. The
motor-effectiveness model entirely replaces the classic rotational dynamics equations,
which depend on a inertia matrix, with a more general model that directly represents the
drone's motors' abilities to generate angular accelerations around each axis. These
parameters are learned.

The original implementation uses sympy to generate the dynamics code. We re-implement it
by hand here in numpy and ensure compliance with the original in tests/test_dynamics.py.
The numpy implementation is directly compatible with batch inputs.
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
    """Strapdown ZYX Euler-rate equations. Batch-aware: (...,3) -> (...,3)."""
    phi, theta = angles[..., 0], angles[..., 1]
    cph, sph = np.cos(phi), np.sin(phi)
    cth, tth = np.cos(theta), np.tan(theta)
    p, q, r = body_rates[..., 0], body_rates[..., 1], body_rates[..., 2]
    return np.stack(
        [
            p + q * sph * tth + r * cph * tth,
            q * cph - r * sph,
            (q * sph + r * cph) / cth,
        ],
        axis=-1,
    )


def unproject_motor(w: NDArray) -> NDArray:
    """Unprojects motor state `w` in [-1,1] to actual rad/s in [w_min_n, w_max_n]."""
    return (w + 1.0) / 2.0 * (W_MAX_N - W_MIN_N) + W_MIN_N


def motor_derivative(
    w: NDArray, u: NDArray, p: QuadParams | NDArray
) -> tuple[NDArray, NDArray]:
    """Time-derivative of the *normalized* motor state (first-order lag). Batch-aware.

    Args:
        w: (n_batch x 4) motor state in [-1,1]
        u: (n_batch x 4) motor command in [-1,1]
        p: Parameters of the quadrotor model as QuadParams or a (23,) array

    Returns:
        A tuple of (d_w, d_W), where d_w is the time-derivative of the normalized motor
        state and d_W is the time-derivative of the unnormalized motor speed in rad/s.
        Both are (n_batch x 4) arrays.
    """
    p = QuadParams._make(p)
    k = np.asarray(p.k)[..., None]
    w_min, w_max = np.asarray(p.w_min)[..., None], np.asarray(p.w_max)[..., None]
    W = unproject_motor(w)
    U = (np.asarray(u) + 1.0) / 2.0  # command [-1,1] -> throttle fraction [0,1]
    Wc = (w_max - w_min) * np.sqrt(
        k * U**2 + (1.0 - k) * U
    ) + w_min  # steady-state rad/s
    d_W = (Wc - W) / np.asarray(p.tau)[..., None]  # first-order lag
    return d_W / (W_MAX_N - W_MIN_N) * 2.0, d_W


def body_force_torque_accel(
    w: NDArray, u: NDArray, vb: NDArray, p: QuadParams | NDArray
) -> tuple[NDArray, NDArray]:
    """Computes linear accelerations due to thrust and drag and angular acceleration.

    Returns (force_frd[3], torque_frd[3]) as accelerations (m/s^2, rad/s^2). vb is
    the FRD body-frame velocity. The Isaac env multiplies the linear part by mass
    for PhysX; the angular part is integrated kinematically (no inertia involved).

    Args:
        w: (n_batch x 4) The motor state in [-1,1]
        u: (n_batch x 4) The motor command in [-1,1]
        vb: (n_batch x 3) The body-frame velocity in m/s, used to compute drag
        p: Parameters of the quadrotor model as QuadParams or a (23,) array

    Returns:
        A tuple of (force, torque), where force is the linear acceleration in FRD due to
        thrust and drag, and torque is the angular acceleration in FRD due to the motor
        thrusts and gyroscopic drag, both as (n_batch x 3) arrays.
    """
    p = QuadParams._make(p)
    vb = np.asarray(vb)
    W = unproject_motor(w)
    _, d_W = motor_derivative(w, u, p)
    W_sum = W.sum(axis=-1)
    sq = W**2

    force = np.stack(
        [
            -p.k_x * vb[..., 0] * W_sum,
            -p.k_y * vb[..., 1] * W_sum,
            -p.k_w * sq.sum(axis=-1),
        ],
        axis=-1,
    )
    torque = np.stack(
        [
            -p.k_p1 * sq[..., 0]
            - p.k_p2 * sq[..., 1]
            + p.k_p3 * sq[..., 2]
            + p.k_p4 * sq[..., 3],
            -p.k_q1 * sq[..., 0]
            + p.k_q2 * sq[..., 1]
            - p.k_q3 * sq[..., 2]
            + p.k_q4 * sq[..., 3],
            -p.k_r1 * W[..., 0]
            + p.k_r2 * W[..., 1]
            + p.k_r3 * W[..., 2]
            - p.k_r4 * W[..., 3]
            - p.k_r5 * d_W[..., 0]
            + p.k_r6 * d_W[..., 1]
            + p.k_r7 * d_W[..., 2]
            - p.k_r8 * d_W[..., 3],
        ],
        axis=-1,
    )
    return force, torque


def rotor_thrusts(w: NDArray, p: QuadParams | NDArray) -> NDArray:
    """Per-rotor thrust as an acceleration (m/s^2): ``k_w * W_i^2``. Batch-aware.

    ``body_force_torque_accel`` implicitly handles per-rotor thrust contributions by
    ``-p.k_w * sq.sum(axis=-1)`` which means motor thrust is the square of the motor
    speed times ``k_w``, and summing them all gets the collective thrust. Without the
    sum, the per-rotor contributions is an array made up of ``p.k_w * W_i^2``.

    Args:
        w: (..., 4) motor state in [-1, 1].
        p: QuadParams or a (23,) parameter array.

    Returns:
        (..., 4) per-rotor thrust acceleration; ``sum(-1)`` equals the model's
        collective thrust accel magnitude. Multiply by mass for force in Newtons.
    """
    p = QuadParams._make(p)
    W = unproject_motor(np.asarray(w))
    return np.asarray(p.k_w) * W**2


def model_derivatives(state: NDArray, u: NDArray, p: QuadParams) -> NDArray:
    """16-state system dynamics according to `optimal_quad_control_RL`.

    Args:
        state: The (n_batch x 16) model state consisting of
          [pos,vel,euler angles,body rates,motor_states]
        u: (n_batch x 4) The motor command in [-1,1]
        p: Parameters of the quadrotor model

    Returns:
        The (n_batch x 16) time-derivative of the state, in the same format as the input
        state.
    """
    state = np.asarray(state, dtype=float)
    vel, angles = state[..., 3:6], state[..., 6:9]
    rot = Rotation.from_euler("xyz", angles)  # body(FRD)->world(NED); single or batch
    vb = rot.inv().apply(vel)
    force, torque = body_force_torque_accel(state[..., 12:16], u, vb, p)
    d_w, _ = motor_derivative(state[..., 12:16], u, p)
    out = np.empty_like(state)
    out[..., 0:3] = vel
    out[..., 3:6] = rot.apply(force)
    out[..., 5] += G
    out[..., 6:9] = _euler_kinematic_rates(angles, state[..., 9:12])
    out[..., 9:12] = torque
    out[..., 12:16] = d_w
    return out
