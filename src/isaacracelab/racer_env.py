"""DirectRLEnv: classical-plant drone racing on Isaac Lab (GPU-parallel).

STATUS: validated. Trains on GPU (2048 envs, ~257k steps/s) to stable flight — mean
episode length 11 -> 1155/1200 steps over 200 PPO iterations, reward -0.25 -> +18. The
torch math it builds on (dynamics_torch/course_torch + the shared backend-agnostic
isaacrace.conversions) is verified vs NumPy in test_parity.py; the GPU classical plant +
frame conventions (body-local wrench, root_lin_vel_b FLU) are validated by that learning
curve.

Classical plant, pre-summed to one body wrench (equivalent to applying 4 forces at the
arms because r x F is frame-linear): with the racer a single rigid body, we compute the
net FLU force + torque ourselves and hand PhysX one wrench — PhysX still integrates the
rotation through the real inertia tensor + gyroscopic coupling.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import numpy as np
import torch

from isaacrace import conversions as ct, dynamics as dyn
from isaacrace.config import RaceTrackConfig, RandomSpawnConfig
from isaacracelab import course_torch, dynamics_torch as dt

RACER_USD = str(Path(__file__).resolve().parents[2] / "src/isaacrace/assets/racer.usd")

# --- classical-plant rotor geometry (FLU), mirror of isaacrace.quadrotor ---
_ARM_X, _ARM_Y = 0.1008, 0.0766  # m, forward / lateral motor offsets (SourceOne CAD)
_ROTOR_RX = (-_ARM_X, +_ARM_X, -_ARM_X, +_ARM_X)  # rotor x (FLU), betaflight layout
_ROTOR_RY = (-_ARM_Y, -_ARM_Y, +_ARM_Y, +_ARM_Y)  # rotor y (FLU)
_YAW_SIGN_FRD = (-1.0, +1.0, +1.0, -1.0)  # per-rotor yaw-reaction sign (FRD-z)
_KAPPA = 0.0157  # cq/ct reaction-to-thrust ratio
_I_ROTOR = 6.19e-6  # rotor+prop polar inertia (kg*m^2), the dW yaw-reaction term

# Measured perception-net error (experimental/perception_train.py): gate Δposition MAE
# ~0.1 m, gate-yaw MAE ~0.15 rad. Used as the obs-noise std so a controller trained here
# tolerates the modular perception net's estimate at composition time.
_PERCEP_POS_STD = 0.10  # m, gate-relative position (build_obs[0:3])
_PERCEP_YAW_STD = 0.15  # rad, gate-relative yaw (build_obs[8])


def _default_track() -> RaceTrackConfig:
    """Placeholder square track so RacerEnvCfg() builds; train.py loads figure8."""
    return RaceTrackConfig(
        gate_pos=[[3, 0, -1.5], [3, 3, -1.5], [0, 3, -1.5], [0, 0, -1.5]],
        gate_yaw=[0, 1, 2, 3],
        start_pos=[0, 0, -1.5],
    )


@configclass
class RacerEnvCfg(DirectRLEnvCfg):
    """Config for the classical-plant racer DirectRLEnv."""

    decimation = 1  # one control decision per physics step => 100 Hz, matches OQCRL
    episode_length_s = 12.0  # 1200 steps * 0.01 s
    action_space = 4  # normalized motor commands [-1, 1]
    observation_space = 20  # course.build_obs
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=dyn.DT, render_interval=1)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=12.0, replicate_physics=True
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=RACER_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.5), joint_pos={}, joint_vel={}
        ),  # racer is a 0-DOF articulation (props are fixed joints) -> no joint init
        actuators={},  # props are nulled fixed-joint links; no actuated DOFs
    )

    # custom (not part of DirectRLEnvCfg)
    track: RaceTrackConfig = _default_track()
    spawn: RandomSpawnConfig = RandomSpawnConfig()
    randomize_reset: bool = True
    rate_penalty: float = course_torch.RaceCourseTorch.RATE_PENALTY
    gate_bonus: float = 0.0  # +reward per gate passed (racing incentive; 0 = off)
    # Observation-noise domain randomization: scale on the measured perception error,
    # added to the gate-relative pose obs so a controller trained here tolerates the
    # perception net's estimate. 0 = clean; 1 = measured magnitude. Weight-only.
    obs_noise: float = 0.0
    # Optional FPV camera (default off; needs AppLauncher --enable_cameras). When set,
    # a TiledCamera mounts on /body and renders each step -> self._camera.data.output.
    tiled_camera: TiledCameraCfg | None = None


class RacerEnv(DirectRLEnv):
    """GPU-parallel classical-plant drone racing (OQCRL course rules)."""

    cfg: RacerEnvCfg

    def __init__(self, cfg: RacerEnvCfg, render_mode: str | None = None, **kw):
        """Build the env, the device-resident course, and classical-plant buffers."""
        super().__init__(cfg, render_mode, **kw)
        dev = self.device
        self._course = course_torch.RaceCourseTorch(
            cfg.track, device=dev, spawn_cfg=cfg.spawn
        )
        self._params = torch.as_tensor(
            np.asarray(dyn.PARAMS_5INCH), dtype=torch.float32, device=dev
        )
        self._rx = torch.tensor(_ROTOR_RX, device=dev)
        self._ry = torch.tensor(_ROTOR_RY, device=dev)
        self._yaw_sign = torch.tensor(_YAW_SIGN_FRD, device=dev)

        n = self.num_envs
        self._motor_w = torch.zeros((n, 4), device=dev)
        self._actions = torch.zeros((n, 4), device=dev)
        self._target_gate = torch.zeros(n, dtype=torch.long, device=dev)
        self._pos_old = torch.zeros((n, 3), device=dev)
        self._crashed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._gates_passed = torch.zeros(n, dtype=torch.long, device=dev)  # per-episode

        self._body_id = self._robot.find_bodies("body")[0]  # /body link (confirmed)
        # body mass (racer.usd bakes ~0.472 kg into /body; rotors nulled).
        masses = self._robot.root_physx_view.get_masses().to(dev)
        self._mass = masses[:, self._body_id[0]]  # (n,)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        # cosmetic ground: termination is z<0 (out_of_bounds), so the drone never
        # rests on it before the bound fires. VERIFY collider handling.
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        if self.cfg.tiled_camera is not None:  # FPV camera, cloned per env
            self._camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["camera"] = self._camera
        self._spawn_extra_prims()  # subclass hook (e.g. visible gates) before clone
        self.scene.clone_environments(copy_from_source=False)
        sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.85, 0.92)).func(
            "/World/Light", sim_utils.DomeLightCfg(intensity=1500.0)
        )

    def _spawn_extra_prims(self):
        """Hook: add per-env prims (e.g. visible gates) under env_0 before cloning."""

    # ------------------------------------------------------------- step pipeline
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clamp(-1.0, 1.0)
        # pre-step ENU position (env-local) for the progress reward
        self._pos_old = self._robot.data.root_pos_w - self.scene.env_origins

    def _apply_action(self):
        """Classical plant: motor lag -> per-rotor thrust/yaw/drag -> body wrench."""
        u = self._actions
        d_w, _ = dt.motor_derivative(self._motor_w, u, self._params)
        self._motor_w = torch.clamp(self._motor_w + self.cfg.sim.dt * d_w, -1.0, 1.0)

        thrust = self._mass.unsqueeze(-1) * dt.rotor_thrusts(
            self._motor_w, self._params
        )
        # body drag (FRD x,y) from the effectiveness model, mapped FRD->FLU.
        # (root_lin_vel_b is the FLU body velocity — validated by the flight curve.)
        vb_frd = ct.vec_flu_frd(self._robot.data.root_lin_vel_b)
        accel_frd, _ = dt.body_force_torque_accel(
            self._motor_w, u, vb_frd, self._params
        )
        # yaw reaction: steady prop drag (kappa*thrust) + rotor dW (k_r5-8 analog)
        _, d_w_rps2 = dt.motor_derivative(self._motor_w, u, self._params)
        yaw_frd = (self._yaw_sign * (_KAPPA * thrust + _I_ROTOR * d_w_rps2)).sum(dim=-1)

        # net FLU wrench: thrust along +z at the arms => collective force + r x F torque
        force = torch.stack(
            [
                self._mass * accel_frd[..., 0],
                -self._mass * accel_frd[..., 1],
                thrust.sum(dim=-1),
            ],
            dim=-1,
        )
        torque = torch.stack(
            [
                (self._ry * thrust).sum(dim=-1),  # sum_i r_y,i * T_i
                -(self._rx * thrust).sum(dim=-1),  # -sum_i r_x,i * T_i
                -yaw_frd,  # FRD-z -> FLU-z
            ],
            dim=-1,
        )

        # set_external_force_and_torque applies in the body-local frame (validated by
        # the flight curve); 2.3 also exposes a wrench composer if you prefer it.
        self._robot.set_external_force_and_torque(
            force.unsqueeze(1), torque.unsqueeze(1), body_ids=self._body_id
        )

    def _state_obs(self) -> torch.Tensor:
        """The privileged 20-dim gate-relative state obs (what the racer consumes).

        Reused by the vision track as supervised labels (image -> this), and as input
        to the state control policy in the modular perception->control composition.
        """
        d = self._robot.data
        pos_enu = d.root_pos_w - self.scene.env_origins
        q_xyzw = ct.quat_wxyz_to_xyzw(d.root_quat_w)
        euler_ned = ct.quat_xyzw_to_euler_xyz(ct.quat_aero_isaac(q_xyzw))
        ned = torch.empty((self.num_envs, 16), device=self.device)
        ned[:, 0:3] = ct.vec_enu_ned(pos_enu)
        ned[:, 3:6] = ct.vec_enu_ned(d.root_lin_vel_w)
        ned[:, 6:9] = euler_ned
        ned[:, 9:12] = ct.vec_flu_frd(d.root_ang_vel_b)
        ned[:, 12:16] = self._motor_w
        return self._course.build_obs(ned, self._target_gate)

    def _get_observations(self) -> dict:
        """Policy obs = gate-relative state, optionally + perception-magnitude noise.

        Domain randomization on the gate-pose estimate the perception net supplies, so
        the controller learns to tolerate it; _state_obs stays clean for labels and the
        oracle. cfg.obs_noise=0 reproduces the clean env exactly.
        """
        obs = self._state_obs()
        s = self.cfg.obs_noise
        if s > 0.0:
            obs[:, 0:3] += s * _PERCEP_POS_STD * torch.randn_like(obs[:, 0:3])
            obs[:, 8] += s * _PERCEP_YAW_STD * torch.randn_like(obs[:, 8])
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pos_new = self._robot.data.root_pos_w - self.scene.env_origins
        tg = self._target_gate
        rew = self._course.progress(self._pos_old, pos_new, tg)
        rew -= self.cfg.rate_penalty * torch.linalg.norm(
            self._robot.data.root_ang_vel_b, dim=-1
        )
        passed, collided = self._course.gate_passed(self._pos_old, pos_new, tg)
        self._gates_passed += passed.long()
        rew += self.cfg.gate_bonus * passed.float()  # racing incentive
        self._target_gate = self._course.advance_gate(tg, passed)
        self._crashed = self._course.out_of_bounds(pos_new) | collided
        return torch.where(
            self._crashed, torch.full_like(rew, self._course.CRASH_REWARD), rew
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # snapshot per-episode gate count into info BEFORE the auto-reset zeros it
        self.extras["gates_passed"] = self._gates_passed.clone()
        return self._crashed, truncated

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        g, ned = self._course.sample_spawn(n, randomize=self.cfg.randomize_reset)
        self._target_gate[env_ids] = g
        self._motor_w[env_ids] = ned[:, 12:16]
        self._gates_passed[env_ids] = 0

        pos_enu = ct.vec_enu_ned(ned[:, 0:3]) + self.scene.env_origins[env_ids]
        q_isaac = ct.quat_aero_isaac(ct.euler_xyz_to_quat_xyzw(ned[:, 6:9]))  # FLU->ENU
        lin_enu = ct.vec_enu_ned(ned[:, 3:6])
        ang_enu = ct.quat_rotate_xyzw(q_isaac, ct.vec_flu_frd(ned[:, 9:12]))
        quat_wxyz = q_isaac[:, [3, 0, 1, 2]]
        self._robot.write_root_pose_to_sim(
            torch.cat([pos_enu, quat_wxyz], dim=-1), env_ids
        )
        self._robot.write_root_velocity_to_sim(
            torch.cat([lin_enu, ang_enu], dim=-1), env_ids
        )
