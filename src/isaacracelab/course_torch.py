"""Torch port of isaacrace.course.RaceCourse rules (build_obs / gate / reward / spawn).

GPU twin for the Isaac Lab spike. Geometry (gate positions, normals, per-gate relatives)
is built once by the NumPy ``RaceCourse`` and copied to device tensors — so the layout
is parity-guaranteed and only the per-step math is re-implemented in torch. The rule
methods are batch-aware (leading dim = num_envs). Parity-tested in test_parity.py.
"""

import torch

from isaacrace.config import RaceTrackConfig, RandomSpawnConfig
from isaacrace.conversions import (
    vec_enu_ned,
)
from isaacrace.course import RaceCourse


def _angle_rotate_point(
    angle: torch.Tensor, point: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Rotate a 2D point by ``angle`` (rad). point: (..., 2)."""
    c, s = torch.cos(angle), torch.sin(angle)
    if inverse:
        s = -s
    return torch.stack(
        [
            c * point[..., 0] - s * point[..., 1],
            s * point[..., 0] + c * point[..., 1],
        ],
        dim=-1,
    )


def _wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles (rad) to [-pi, pi]."""
    return (a + torch.pi) % (2 * torch.pi) - torch.pi


class RaceCourseTorch:
    """Device-resident race rules mirroring isaacrace.course.RaceCourse."""

    RATE_PENALTY = RaceCourse.RATE_PENALTY
    CRASH_REWARD = RaceCourse.CRASH_REWARD

    def __init__(
        self,
        cfg: RaceTrackConfig,
        device: torch.device | str = "cpu",
        spawn_cfg: RandomSpawnConfig | None = None,
    ):
        """Build geometry via the NumPy course, then mirror it onto ``device``."""
        base = RaceCourse(cfg, spawn_cfg)
        self.device = torch.device(device)
        self.num_gates = base.num_gates
        self.gate_size = base.gate_size
        self.spawn_cfg = base.spawn_cfg
        self._bound_xy = base._bound_xy
        self._bound_z = base._bound_z

        def t(a):
            return torch.as_tensor(a, dtype=torch.float32, device=self.device)

        self.gate_pos = t(base.gate_pos)  # NED [N,3]
        self.gate_yaw = t(base.gate_yaw)  # NED [N]
        self.gate_pos_enu = t(base.gate_pos_enu)  # [N,3]
        self.gate_normal_enu = t(base.gate_normal_enu)  # [N,3]
        self.start_pos_enu = t(base.start_pos_enu)  # [3]
        self._gate_pos_rel = t(base._gate_pos_rel)  # [N,3]
        self._gate_yaw_rel = t(base._gate_yaw_rel)  # [N]

    def build_obs(
        self, world_state: torch.Tensor, target_gate: torch.Tensor, gates_ahead: int = 1
    ) -> torch.Tensor:
        """(N,16) NED-FRD state + (N,) target gate -> (N, 16+4*gates_ahead) obs."""
        s = world_state
        n = s.shape[0]
        tg = target_gate.long()
        wrapped = tg % self.num_gates
        gp = self.gate_pos[wrapped]
        gy = self.gate_yaw[wrapped]

        obs = torch.zeros((n, 16 + 4 * gates_ahead), dtype=s.dtype, device=s.device)
        obs[:, 0:2] = _angle_rotate_point(gy, s[:, 0:2] - gp[:, 0:2], inverse=True)
        obs[:, 2] = s[:, 2] - gp[:, 2]
        obs[:, 3:5] = _angle_rotate_point(gy, s[:, 3:5], inverse=True)
        obs[:, 5] = s[:, 5]
        obs[:, 6:8] = s[:, 6:8]
        obs[:, 8] = _wrap_to_pi(s[:, 8] - gy)
        obs[:, 9:12] = s[:, 9:12]
        obs[:, 12:16] = s[:, 12:16]

        offsets = torch.arange(1, gates_ahead + 1, device=s.device)
        idx = (tg[:, None] + offsets[None]) % self.num_gates
        future_pos = self._gate_pos_rel[idx]  # (N, gates_ahead, 3)
        future_yaw = self._gate_yaw_rel[idx].unsqueeze(-1)  # (N, gates_ahead, 1)
        obs[:, 16:] = torch.cat([future_pos, future_yaw], dim=-1).reshape(n, -1)
        return obs

    def gate_passed(
        self, pos_old: torch.Tensor, pos_new: torch.Tensor, target_gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ENU positions -> (passed, collided) bool tensors (N,)."""
        wrapped = target_gate.long() % self.num_gates
        gp = self.gate_pos_enu[wrapped]
        nrm = self.gate_normal_enu[wrapped]
        proj_old = ((pos_old[..., :2] - gp[..., :2]) * nrm[..., :2]).sum(dim=-1)
        proj_new = ((pos_new[..., :2] - gp[..., :2]) * nrm[..., :2]).sum(dim=-1)
        crossed = (proj_old < 0) & (proj_new > 0)
        in_window = (torch.abs(pos_new - gp) < self.gate_size / 2).all(dim=-1)
        return crossed & in_window, crossed & ~in_window

    def progress(
        self, pos_old: torch.Tensor, pos_new: torch.Tensor, target_gate: torch.Tensor
    ) -> torch.Tensor:
        """Closing-distance reward to the target gate (ENU). (N,)."""
        gp = self.gate_pos_enu[target_gate.long() % self.num_gates]
        return torch.linalg.norm(pos_old - gp, dim=-1) - torch.linalg.norm(
            pos_new - gp, dim=-1
        )

    def out_of_bounds(self, pos_enu: torch.Tensor) -> torch.Tensor:
        """True where below ground or outside the arena (ENU). (N,)."""
        ground = pos_enu[..., 2] < 0.0
        oob = (torch.abs(pos_enu[..., 0:2]) > self._bound_xy).any(dim=-1) | (
            pos_enu[..., 2] > self._bound_z
        )
        return ground | oob

    def advance_gate(
        self, target_gate: torch.Tensor, passed: torch.Tensor
    ) -> torch.Tensor:
        """Increment (wrap) the target gate where ``passed``."""
        return torch.where(passed, (target_gate + 1) % self.num_gates, target_gate)

    def sample_spawn(
        self, n: int, generator: torch.Generator | None = None, randomize: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (gate_idx, state) NED on device; mirrors the NumPy sampler."""
        dev = self.device
        state = torch.zeros((n, 16), device=dev)

        def u(*shape):  # uniform [-1, 1)
            return torch.rand(shape, device=dev, generator=generator) * 2.0 - 1.0

        if randomize:
            sc = self.spawn_cfg
            g = torch.randint(0, self.num_gates, (n,), device=dev, generator=generator)
            pos_enu = self.gate_pos_enu[g] - sc.dist_back * self.gate_normal_enu[g]
            state[:, 0:3] = vec_enu_ned(pos_enu)
            state[:, 3:6] = vec_enu_ned(u(n, 3) * sc.vel_bounds)
            state[:, 6] = u(n) * sc.tilt_bounds
            state[:, 7] = u(n) * sc.tilt_bounds
            state[:, 8] = u(n) * torch.pi
            state[:, 9:12] = u(n, 3) * sc.rate_bounds
            state[:, 12:16] = u(n, 4)
        else:
            g = torch.zeros(n, dtype=torch.long, device=dev)
            state[:, 0:3] = vec_enu_ned(self.start_pos_enu).expand(n, 3)
        return g, state
