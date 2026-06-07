"""RaceCourse — the gate racecourse, built from a (hydra) RaceTrackConfig.

Owns the gate geometry in both frames so the env stays config-driven (swap
conf/track/*.yaml to race a different course): the policy's NED frame for the
gate-relative observation (build_obs), and Isaac's ENU frame for spawning +
gate-pass tests. The math mirrors the validated OQCRL course exactly, so the
trained racer flies unchanged on the default figure-8.
"""

import numpy as np
from numpy.typing import NDArray

from isaacrace.config import RaceTrackConfig
from isaacrace.conversions import vec_enu_ned


class RaceCourse:
    """A gate racecourse built from a RaceTrackConfig, in both NED and ENU frames."""

    def __init__(self, cfg: RaceTrackConfig):
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

    def build_obs(self, world_state: NDArray, target_gate: int, gates_ahead: int = 1):
        """OQCRL 20-dim gate-relative obs from a NED-FRD world state + target gate."""
        s = world_state
        n = self.num_gates
        gp = self.gate_pos[target_gate % n]
        gy = self.gate_yaw[target_gate % n]
        rot = np.array([[np.cos(gy), np.sin(gy)], [-np.sin(gy), np.cos(gy)]])
        obs = np.zeros(16 + 4 * gates_ahead, dtype=np.float32)
        obs[0:2] = (s[0:2] - gp[0:2]) @ rot.T
        obs[2] = s[2] - gp[2]
        obs[3:5] = s[3:5] @ rot.T
        obs[5] = s[5]
        obs[6:8] = s[6:8]
        yaw = (s[8] - gy) % (2 * np.pi)
        obs[8] = (
            yaw - 2 * np.pi
            if yaw > np.pi
            else (yaw + 2 * np.pi if yaw < -np.pi else yaw)
        )
        obs[9:12] = s[9:12]
        obs[12:16] = s[12:16]
        idx = (target_gate + np.arange(gates_ahead) + 1) % n
        gate_coords = np.hstack((
            self._gate_pos_rel[idx],
            self._gate_yaw_rel[idx, None],
        ))
        obs[16 : 16 + gates_ahead * 4] = gate_coords.ravel()

        return obs

    def gate_passed(self, pos_old, pos_new, target_gate):
        """ENU gate-plane crossing + window test. Returns (passed, collided)."""
        n = self.num_gates
        gp = self.gate_pos_enu[target_gate % n]
        nrm = self.gate_normal_enu[target_gate % n]
        proj_old = (pos_old[0] - gp[0]) * nrm[0] + (pos_old[1] - gp[1]) * nrm[1]
        proj_new = (pos_new[0] - gp[0]) * nrm[0] + (pos_new[1] - gp[1]) * nrm[1]
        crossed = (proj_old < 0) and (proj_new > 0)
        in_window = np.all(np.abs(pos_new - gp) < self.gate_size / 2)
        return (crossed and in_window), (crossed and not in_window)
