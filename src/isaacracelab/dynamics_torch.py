"""Torch port of the OQCRL motor + force/torque model (actuation-relevant subset).

GPU twin of isaacrace.dynamics for the Isaac Lab classical plant: the first-order motor
lag, per-rotor thrust, body drag, and the FRD force/torque accelerations. The full
``model_derivatives`` integrator is NOT ported — in Isaac Lab PhysX integrates the
rigid-body motion; we only feed it forces/torques. Parity-tested in test_parity.py.

Parameters ``p`` are a (..., 23) tensor in isaacrace.dynamics.QuadParams field order.
"""

import torch

# QuadParams field order (MUST match isaacrace.dynamics.QuadParams).
KX, KY, KW = 0, 1, 2
KP1, KP2, KP3, KP4 = 3, 4, 5, 6
KQ1, KQ2, KQ3, KQ4 = 7, 8, 9, 10
KR1, KR2, KR3, KR4, KR5, KR6, KR7, KR8 = 11, 12, 13, 14, 15, 16, 17, 18
TAU, K, WMIN, WMAX = 19, 20, 21, 22

G = 9.81
W_MIN_N, W_MAX_N = 0.0, 3000.0  # motor-state -> rad/s normalization range
DT = 0.01  # 100 Hz OQCRL step


def unproject_motor(w: torch.Tensor) -> torch.Tensor:
    """Normalized motor state w in [-1, 1] -> rad/s in [W_MIN_N, W_MAX_N]."""
    return (w + 1.0) / 2.0 * (W_MAX_N - W_MIN_N) + W_MIN_N


def motor_derivative(
    w: torch.Tensor, u: torch.Tensor, p: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order motor lag; returns (d_w_norm, d_W_rps), each (..., 4)."""
    k = p[..., K].unsqueeze(-1)
    w_min, w_max = p[..., WMIN].unsqueeze(-1), p[..., WMAX].unsqueeze(-1)
    tau = p[..., TAU].unsqueeze(-1)
    W = unproject_motor(w)
    U = (u + 1.0) / 2.0  # command [-1,1] -> throttle [0,1]
    Wc = (w_max - w_min) * torch.sqrt(k * U**2 + (1.0 - k) * U) + w_min
    d_W = (Wc - W) / tau
    return d_W / (W_MAX_N - W_MIN_N) * 2.0, d_W


def rotor_thrusts(w: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Per-rotor thrust acceleration ``k_w * W_i^2`` (..., 4); sum(-1) = collective."""
    W = unproject_motor(w)
    return p[..., KW].unsqueeze(-1) * W**2


def body_force_torque_accel(
    w: torch.Tensor, u: torch.Tensor, vb: torch.Tensor, p: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """FRD (force, torque) accel (m/s^2, rad/s^2). vb: FRD body velocity (..., 3)."""
    W = unproject_motor(w)
    _, d_W = motor_derivative(w, u, p)
    W_sum = W.sum(dim=-1)
    sq = W**2

    force = torch.stack(
        [
            -p[..., KX] * vb[..., 0] * W_sum,
            -p[..., KY] * vb[..., 1] * W_sum,
            -p[..., KW] * sq.sum(dim=-1),
        ],
        dim=-1,
    )
    torque_z = (
        -p[..., KR1] * W[..., 0]
        + p[..., KR2] * W[..., 1]
        + p[..., KR3] * W[..., 2]
        - p[..., KR4] * W[..., 3]
        - p[..., KR5] * d_W[..., 0]
        + p[..., KR6] * d_W[..., 1]
        + p[..., KR7] * d_W[..., 2]
        - p[..., KR8] * d_W[..., 3]
    )
    torque = torch.stack(
        [
            -p[..., KP1] * sq[..., 0]
            - p[..., KP2] * sq[..., 1]
            + p[..., KP3] * sq[..., 2]
            + p[..., KP4] * sq[..., 3],
            -p[..., KQ1] * sq[..., 0]
            + p[..., KQ2] * sq[..., 1]
            - p[..., KQ3] * sq[..., 2]
            + p[..., KQ4] * sq[..., 3],
            torque_z,
        ],
        dim=-1,
    )
    return force, torque
