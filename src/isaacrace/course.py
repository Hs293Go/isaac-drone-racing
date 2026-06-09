"""The gate racecourse, built from a (hydra) RaceTrackConfig.

This is a configuration driven representation of the racecourse geometry ported from
optimal_quad_control_rl's validated course. It also manages "racing rules" like reward,
termination, and spawn state generation.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from isaacrace.config import RaceTrackConfig, RandomSpawnConfig
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
    """A gate racecourse built from a RaceTrackConfig.

    Also the single source of the race "rules" (reward, termination, spawn) shared by
    both the Isaac RaceEnv and the vectorized BatchedRaceEnv, so the two stay in lock
    step. The rule methods are batch-aware (scalar or leading (N,) like build_obs).
    """

    # Reward / termination constants (one source for both envs).
    RATE_PENALTY = 0.001  # reward -= this * |body rates|
    CRASH_REWARD = -10.0  # reward on crash (ground / out-of-bounds / collision)

    def __init__(
        self, cfg: RaceTrackConfig, spawn_cfg: RandomSpawnConfig | None = None
    ):
        """Initializes the racecourse geometry and random spawn configuration.

        Args:
            cfg: The RaceTrackConfig defining the gate positions, headings, and sizes.
            spawn_cfg: Optional RandomSpawnConfig defining how to randomize the drone's
              initial pose. If None, a default configuration is used. Only used if
              `sample_spawn` is called with `randomize=True`.
        """
        self.spawn_cfg = spawn_cfg if spawn_cfg is not None else RandomSpawnConfig()
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
        self._bound_xy = float(cfg.bound_xy)
        self._bound_z = float(cfg.bound_z)

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

        This method only needs to interact with the drone state in isaacsim to determine
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

    # ------------------------------------------------- race rules (batch-aware)
    def progress(
        self, pos_old: ArrayLike, pos_new: ArrayLike, target_gate: ArrayLike
    ) -> NDArray:
        """Reward for closing distance to the current gate.

        This method only needs to interact with the drone state in isaacsim to compute
        the progress reward, therefore it uses ENU coordinates.

        Args:
            pos_old: The (n_batch x 3) previous position of the drone in ENU coordinates
            pos_new: The (n_batch x 3) position of the drone in ENU coordinates
            target_gate: Up to `n_batch` indices of the gate to test against (0-based).

        Returns:
            A scalar or (n_batch,) reward representing the change in distance to the
            target gate (positive if the drone got closer, negative if it got farther).
        """
        gp = self.gate_pos_enu[np.asarray(target_gate) % self.num_gates]
        return np.linalg.norm(pos_old - gp, axis=-1) - np.linalg.norm(
            pos_new - gp, axis=-1
        )

    def out_of_bounds(self, pos_enu: ArrayLike) -> NDArray[np.bool_]:
        """True where the drone hit the ground or left the arena (ENU).

        This method only needs to interact with the drone state in isaacsim to determine
        if the drone is out of bounds, therefore it uses ENU coordinates.

        Args:
            pos_enu: The (n_batch x 3) position of the drone in ENU coordinates.

        Returns:
            A bool or (n_batch,) array where True indicates the drone is out of bounds
            (below ground or outside the horizontal/vertical bounds).
        """
        pos_enu = np.asarray(pos_enu)
        ground = pos_enu[..., 2] < 0.0
        oob = (np.abs(pos_enu[..., 0:2]) > self._bound_xy).any(axis=-1) | (
            pos_enu[..., 2] > self._bound_z
        )
        return ground | oob

    def advance_gate(self, target_gate: ArrayLike, passed: ArrayLike) -> NDArray:
        """Advance the target gate index where ``passed`` (wraps).

        Args:
            target_gate: Up to `n_batch` indices of the current target gate (0-based).
            passed: A bool or (n_batch,) array where True indicates the gate was passed.

        Returns:
            An array of the same shape as `target_gate` with the updated target gate
            indices (incremented by 1 where `passed` is True, with wrap-around to 0
            after the last gate).
        """
        return np.where(
            passed, (np.asarray(target_gate) + 1) % self.num_gates, target_gate
        )

    def sample_spawn(
        self, rng: np.random.Generator, n: int | None = None, randomize: bool = True
    ) -> tuple[int | NDArray, NDArray]:
        """Sample spawn state(s) in front of a target gate.

        This is the source of the reset distribution; RaceEnv converts the NED state to
        an ENU-FLU PhysX pose, BatchedRaceEnv uses it directly.

        Args:
            rng: The random number generator to use for sampling.
            n: The number of spawn states to sample. If None, a single state is
              returned.
            randomize: Whether to randomize the spawn state. If true, we draw a jittered
              pose from ``self.spawn_cfg`` near a random gate, otherwise we use a
              deterministic demo start (gate 0, level, zero vel/rates/motor). Defaults
              to True so the implicit behavior is randomized spawns, not a silently
              frozen start.

        Returns:
            A tuple of (gate_index, state) where gate_index is the index (either an
            integer or an array) of the gate to spawn in front of, and state is the
            corresponding (16,) or (n, 16) array of NED pose/vel/rates/motor.
        """
        scalar = n is None
        m = 1 if scalar else n
        state = np.zeros((m, 16))
        if randomize:
            sc = self.spawn_cfg
            g = rng.integers(0, self.num_gates, m)
            pos_enu = self.gate_pos_enu[g].astype(float) - sc.dist_back * (
                self.gate_normal_enu[g].astype(float)
            )
            state[:, 0:3] = vec_enu_ned(pos_enu)
            state[:, 3:6] = vec_enu_ned(
                rng.uniform(-sc.vel_bounds, sc.vel_bounds, (m, 3))
            )
            state[:, 6] = rng.uniform(-sc.tilt_bounds, sc.tilt_bounds, m)
            state[:, 7] = rng.uniform(-sc.tilt_bounds, sc.tilt_bounds, m)
            state[:, 8] = rng.uniform(-np.pi, np.pi, m)
            state[:, 9:12] = rng.uniform(-sc.rate_bounds, sc.rate_bounds, (m, 3))
            state[:, 12:16] = rng.uniform(-1.0, 1.0, (m, 4))
        else:
            g = np.zeros(m, dtype=int)
            # level, facing gate 0 (NED euler = 0); vel/rates/motor stay zero
            state[:, 0:3] = vec_enu_ned(self.start_pos_enu.astype(float))
        return (int(g[0]), state[0]) if scalar else (g, state)
