"""The gate racecourse, built from a (hydra) RaceTrackConfig.

This is a configuration driven representation of the racecourse geometry ported from
optimal_quad_control_rl's validated course.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from isaacrace.config import RaceTrackConfig
from isaacrace.conversions import vec_enu_ned


def _angle_rotate_point(
    angle: NDArray, point: NDArray, inverse: bool = False
) -> NDArray:
    """Rotates a 2D point by the given angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    if inverse:
        s = -s
    return np.stack(
        [
            c * point[..., 0] - s * point[..., 1],
            s * point[..., 0] + c * point[..., 1],
        ],
        axis=-1,
    )


def _wrap_to_pi(angles):
    """Wraps angles (in radians) to the range [-pi, pi]."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


class RaceCourse:
    """A gate racecourse built from a RaceTrackConfig."""

    def __init__(self, cfg: RaceTrackConfig):
        """Initializes the racecourse geometry from the config."""
        self.gate_pos = np.asarray(cfg.gate_pos, dtype=np.float32)  # NED [N,3]
        scale = np.pi / 2.0 if cfg.gate_yaw_unit == "multiples_pi_2" else 1.0
        self.gate_yaw = (
            np.asarray(cfg.gate_yaw, dtype=np.float32) * scale
        )  # NED radians [N]
        self.start_pos = np.asarray(cfg.start_pos, dtype=np.float32)  # NED [3]
        self.gate_size = float(cfg.gate_size)
        self.num_gates = int(self.gate_pos.shape[0])
        self._gate_pos_rel, self._gate_yaw_rel = self._relatives()  # NED, for build_obs

        # ENU (env-native): gate centres, "through" normals, headings, start.
        self.gate_pos_enu = vec_enu_ned(self.gate_pos).astype(np.float32)
        self.gate_normal_enu = np.stack([
            np.array([np.sin(y), np.cos(y), 0.0]) for y in self.gate_yaw
        ]).astype(np.float32)
        self.gate_yaw_enu = (-self.gate_yaw).astype(np.float32)
        self.start_pos_enu = vec_enu_ned(self.start_pos).astype(np.float32)

    def _relatives(self):
        """Per-gate position/heading relative to the previous gate (its frame)."""
        n = self.num_gates
        gpr = np.zeros((n, 3), dtype=np.float32)
        gyr = np.zeros(n, dtype=np.float32)
        for i in range(n):
            gpr[i] = self.gate_pos[i] - self.gate_pos[i - 1]
            yaw_prev = self.gate_yaw[i - 1]
            rot = np.array([
                [np.cos(yaw_prev), np.sin(yaw_prev)],
                [-np.sin(yaw_prev), np.cos(yaw_prev)],
            ])
            gpr[i, 0:2] = rot @ gpr[i, 0:2]
            yr = (self.gate_yaw[i] - yaw_prev) % (2 * np.pi)
            gyr[i] = (
                yr - 2 * np.pi
                if yr > np.pi
                else (yr + 2 * np.pi if yr < -np.pi else yr)
            )
        return gpr, gyr

    def build_obs(
        self,
        world_state: ArrayLike,
        target_gate: ArrayLike,
        gates_ahead: int = 1,
    ) -> NDArray:
        """Builds the 20-dim gate-relative observations.

        This method outputs the RL observation, therefore it uses NED coordinates and
        aerospace Euler angles.

        Args:
            world_state: The (n_batch x 16) world state in aerospace (FRD/NED)
              convention, consisting of [pos,vel,euler angles,body rates,motor_states]
            target_gate: Up to n_batch indices of the next gate to pass (0-based).
            gates_ahead: The number of future gates to include in the obs (default 1).

        Returns:
            A (n_batch x 20) obs consisting of [
              pos w.r.t. gate in body frame,
              vel w.r.t. gate in body frame,
              attitude w.r.t. gate (Euler angle difference),
              body_rates (direct observation),
              motor_states (direct observation),
              relative pos and yaw of the next gates (in the current gate's frame)
            ]
        """
        # Ensure inputs are 2D (batch_size, features) or 1D arrays for indices
        s = np.atleast_2d(world_state)
        batch_dims = s.shape[0]

        # Ensure target_gate is an array matching the batch dimension
        target_gate = np.atleast_1d(target_gate).astype(np.int_)

        wrapped_target_gate = target_gate % self.num_gates
        gp = self.gate_pos[wrapped_target_gate]  # Shape: (batch_dims, 3)
        gy = self.gate_yaw[wrapped_target_gate]  # Shape: (batch_dims,)

        obs = np.zeros((batch_dims, 16 + 4 * gates_ahead), dtype=np.float32)

        pos_diff = s[..., 0:2] - gp[..., 0:2]
        obs[..., 0:2] = _angle_rotate_point(gy, pos_diff, inverse=True)  # Relative X/Y
        obs[..., 2] = s[..., 2] - gp[..., 2]  # Relative Z
        obs[..., 3:5] = _angle_rotate_point(gy, s[..., 3:5], inverse=True)  # Velocity
        obs[..., 5] = s[..., 5]  # Velocity Z
        obs[..., 6:8] = s[..., 6:8]  # Drone roll/pitch
        obs[..., 8] = _wrap_to_pi(s[..., 8] - gy)  # yaw relative to the gate heading

        obs[..., 9:12] = s[..., 9:12]  # Body rates
        obs[..., 12:16] = s[..., 12:16]  # Motor states

        lookahead_offsets = np.arange(1, gates_ahead + 1)
        # Generate a grid of indices for each gate index and look-ahead offset using
        # broadcasting addition (batch_dims x gates_ahead)
        idx = (target_gate[:, None] + lookahead_offsets[None]) % self.num_gates

        future_pos = self._gate_pos_rel[idx]  # batch_dims x gates_ahead x 3
        future_yaw = self._gate_yaw_rel[idx, None]  # batch_dims x gates_ahead x 1

        # batch_dims x gates_ahead x 4
        gate_coords = np.concatenate((future_pos, future_yaw), axis=-1)

        obs[..., 16:] = gate_coords.reshape(batch_dims, -1)
        return obs if batch_dims > 1 else obs[0]  # Return 1D if input was 1D

    def gate_passed(
        self,
        pos_old: ArrayLike,
        pos_new: ArrayLike,
        target_gate: ArrayLike,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Tests gate-plane crossing and in window.

        This method only need to interact with the drone state in isaacsim to determine
        if a gate was passed, therefore it uses ENU coordinates.

        Args:
            pos_old: The (n_batch x 3) previous position of the drone in ENU coordinates
            pos_new: The (n_batch x 3) position of the drone in ENU coordinates
            target_gate: Up to `n_batch` indices of the gate to test against (0-based).

        Returns:
            A tuple representing (passed, passed but outside window)
        """
        pos_old = np.asarray(pos_old)
        pos_new = np.asarray(pos_new)
        target_gate = np.asarray(target_gate, dtype=np.int_)

        # Lookup gate properties
        wrapped_target_gate = target_gate % self.num_gates
        gp = self.gate_pos_enu[wrapped_target_gate]
        nrm = self.gate_normal_enu[wrapped_target_gate]

        # Project vectors onto normal
        proj_old = np.sum((pos_old[..., :2] - gp[..., :2]) * nrm[..., :2], axis=-1)
        proj_new = np.sum((pos_new[..., :2] - gp[..., :2]) * nrm[..., :2], axis=-1)

        crossed = (proj_old < 0) & (proj_new > 0)
        in_window = np.all(np.abs(pos_new - gp) < self.gate_size / 2, axis=-1)

        passed = crossed & in_window
        collided = crossed & ~in_window

        # Extract scalar bools if the input was a single environment/scalar
        if target_gate.ndim == 0:
            return passed.item(), collided.item()

        return passed, collided
