"""RaceEnv — a Gymnasium env: a Quadrotor racing a figure-8 gate course in Isaac Sim.

The env is a thin gym wrapper around the drone: it owns the World, the scene
(lighting/ground/gates), the RaceCourse, and the gym loop. The drone, represented by
RacingDrone (see quadrotor.py), owns the physics state and dynamics. The env drives the
drone explicitly each step (apply_disc_control -> world.step -> update_state) instead of
via World physics callbacks, because the RL action must be injected per step.

Since this env interacts with Isaac, it uses an ENU-native convention for gates, reward,
reset, and bounds. NED appears only at the observation boundary (_ned_obs_state) and in
converting the shared NED spawn sampler's output in reset() (the trained policies are
NED).

A SimulationApp must already be running before this module's isaacsim imports
execute (see demo/train).
"""

from typing import Any, ClassVar

import gymnasium as gym
from gymnasium import spaces
from isaacsim.core.api.world import World
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
import numpy as np
from numpy.typing import ArrayLike, NDArray
import omni.timeline
from pxr import Gf, UsdGeom, UsdLux
from scipy.spatial.transform import Rotation

from isaacrace import perception
from isaacrace.config import DynamicsMode, FpvConfig
from isaacrace.conversions import quat_aero_isaac, vec_enu_ned, vec_flu_frd
from isaacrace.course import RaceCourse
import isaacrace.dynamics as dyn
from isaacrace.quadrotor import RacingDrone


class RaceEnv(gym.Env):
    """Environment for a single drone racing in Isaac Sim.

    This environment lets the RacingDrone object manage its state vector internally. Its
    primary public method `step` takes a (4,) action of normalized motor commands and
    returns a (20,) observation of the drone's state in addition to a scalar reward,
    done/truncated flags, and an info dict.
    """

    metadata: ClassVar[dict] = {"render_modes": ["human"]}

    def __init__(
        self,
        course: RaceCourse,
        headless: bool = True,
        randomize_reset: bool = True,
        seed: int = 0,
        max_steps: int = 1200,
        fpv: FpvConfig | None = None,
        randomize_params: bool = False,
        param_dr_pct: float = 0.1,
        perception_weight: float = 0.0,
        backpedal_weight: float = 0.0,
        dynamics_mode: DynamicsMode = "kinematic",
    ):
        """Initialize the racing environment.

        This spawns the scene (lighting, ground, gates) and the drone, and initializes
        the physics world. The drone is reset to the start pose.

        Args:
            course: the RaceCourse defining the gate layout and obs construction.
            headless: if True, disable rendering (for faster training).
            randomize_reset: if True, randomize the drone's start pose each episode.
            seed: random seed for reset randomization (obs and params).
            max_steps: max steps per episode (for truncation).
            fpv: if not None, config for the drone's FPV camera.
            randomize_params: if True, apply domain randomization to the drone's model
                parameters each episode (in reset); see param_dr_pct.
            param_dr_pct: if randomize_params, the ±% range to sample each parameter
                around its nominal value (e.g. 0.1 = ±10%).
            perception_weight: weight on the FOV-gated gate-visibility shaping reward
                (matches BatchedRaceEnv); 0 = the bare progress reward.
            backpedal_weight: weight on the tail-first penalty (reward nose-first);
                0 suppresses it.
            dynamics_mode: Isaac plant model, either:
                - "kinematic": overrides rotational dynamics to be based on TU Delft's
                  motor effectiveness/INDI model
                - "classical": per-rotor forces through a real inertia tensor;
                see RacingDrone
        """
        super().__init__()
        self.course = course
        self.randomize_reset = randomize_reset
        self.randomize_params = randomize_params
        self.param_dr_pct = param_dr_pct
        self.perception_weight = perception_weight
        self.backpedal_weight = backpedal_weight
        self._fpv = fpv if fpv is not None else FpvConfig()
        self.rng = np.random.default_rng(seed)
        self.p = dyn.PARAMS_5INCH
        self.dt = dyn.DT
        self.max_steps = max_steps

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(20,), dtype=np.float32
        )

        self.target_gate = 0
        self.step_count = 0
        self._obs = np.zeros(20, dtype=np.float32)
        self._render = not headless

        self.world = World(
            physics_dt=self.dt, rendering_dt=self.dt, stage_units_in_meters=1.0
        )
        self._stage = self.world.stage

        # Scene + drone are all built BEFORE world.reset(): the
        # gate/ground/rotor/camera prim edits are stage-structure changes that
        # would invalidate the physics tensor view if done afterwards.
        self._spawn_lighting()
        self._spawn_ground()
        self._spawn_gates()
        self.quad = RacingDrone(
            self.world,
            self.course.start_pos_enu,
            self.p,
            fpv=fpv,
            dynamics_mode=dynamics_mode,
        )
        self.world.reset()
        # PhysX tensor-view reads/writes (RigidPrim) need the timeline playing; start
        # it once here so the env is self-contained (demo/train just construct it and
        # use the gym API — no external play()).
        omni.timeline.get_timeline_interface().play()

    # ------------------------------------------------------------------ scene
    def _spawn_lighting(self):
        # DomeLight = soft ambient fill from all directions (so nothing is pure
        # black); a tilted DistantLight = directional key that gives the
        # gates/drone visible 3D shading. Kept moderate and the dome faintly tinted
        # so the scene reads with depth instead of washing out to flat white.
        dome = UsdLux.DomeLight.Define(self._stage, "/World/DomeLight")
        dome.CreateIntensityAttr(1000.0)
        dome.CreateColorAttr(Gf.Vec3f(0.80, 0.85, 0.92))  # faint cool "sky"
        key = UsdLux.DistantLight.Define(self._stage, "/World/KeyLight")
        key.CreateIntensityAttr(3000.0)
        UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(
            Gf.Vec3f(-55.0, 0.0, 35.0)
        )

    def _spawn_ground(self):
        # Pegasus/Isaac default ground plane — the blue grid that curls up at the
        # edges. Fetched from the Isaac asset server on first use (then cached
        # locally). We disable its collider so the drone still falls through z=0
        # and the z<0 ground-termination fires as before; the ground is cosmetic.
        # (Done before world.reset() — deactivating a prim is a structure change
        # that would otherwise invalidate the physics tensor view.)
        self.world.scene.add_default_ground_plane()
        for path in (
            "/World/defaultGroundPlane/GroundPlane/CollisionPlane",
            "/World/defaultGroundPlane/SphereLight",
        ):
            prim = get_prim_at_path(path)
            if prim and prim.IsValid():
                prim.SetActive(False)

    def _spawn_gates(self):
        h = self.course.gate_size / 2.0
        th = 0.06  # bar thickness
        span = (
            self.course.gate_size
        )  # bars span the full window so corners meet at (±h, ±h)
        for i in range(self.course.num_gates):
            pos = self.course.gate_pos_enu[i]
            path = f"/World/Gate_{i}"
            define_prim(path, "Xform")
            x = UsdGeom.Xformable(get_prim_at_path(path))
            x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
            x.AddRotateZOp().Set(float(np.degrees(self.course.gate_yaw_enu[i])))
            # gate 0 = green (first target / course start); the rest racing-orange
            color = Gf.Vec3f(0.15, 0.8, 0.2) if i == 0 else Gf.Vec3f(0.95, 0.45, 0.08)
            # square frame in the local X-Z plane; the drone flies through along
            # local y (the gate normal, since AddRotateZOp(GATE_YAW_ENU) maps local
            # y -> GATE_NORMAL_ENU). top/bottom span X (thin in y,z); left/right
            # span Z (thin in x,y) -> a flat, closed square window.
            for name, off, half in [
                ("t", (0, 0, h), (span, th, th)),
                ("b", (0, 0, -h), (span, th, th)),
                ("l", (-h, 0, 0), (th, th, span)),
                ("r", (h, 0, 0), (th, th, span)),
            ]:
                c = UsdGeom.Cube.Define(self._stage, f"{path}/{name}")
                c.CreateSizeAttr(1.0)
                c.CreateDisplayColorAttr([color])
                cx = UsdGeom.Xformable(c)
                cx.AddTranslateOp().Set(Gf.Vec3d(*off))
                cx.AddScaleOp().Set(Gf.Vec3d(*half))

    # ----------------------------------------------- observation (NED boundary)
    @property
    def _fpv_cam_path(self):
        """USD path of the FPV camera prim (or None); demo.py docks a viewport."""
        return self.quad.camera.path if self.quad.camera else None

    @property
    def _world_state(self):
        """ENU-FLU 16-vec view of the drone state (pos, vel, euler, rates, rpms).

        For demo logging.
        """
        s = self.quad.state
        out = np.empty(16)
        out[0:3] = s.position
        out[3:6] = s.linear_velocity
        out[6:9] = Rotation.from_quat(s.attitude).as_euler("xyz")
        out[9:12] = s.angular_velocity
        out[12:16] = self.quad.motor_w
        return out

    def _ned_obs_state(self) -> NDArray:
        """Assemble the NED-FRD 16-vector build_obs expects.

        From the quad's State + rotor speeds. This honors the NED policy convention to
        output a RL-compatible observation.
        """
        s = self.quad.state
        out = np.empty(16)
        out[0:3] = vec_enu_ned(s.position)
        out[3:6] = vec_enu_ned(s.linear_velocity)
        out[6:9] = Rotation.from_quat(quat_aero_isaac(s.attitude)).as_euler("xyz")
        out[9:12] = vec_flu_frd(s.angular_velocity)
        out[12:16] = self.quad.motor_w
        return out

    def _update_obs(self):
        self._obs = self.course.build_obs(self._ned_obs_state(), self.target_gate)

    # ------------------------------------------------------------------ gym API
    def step(
        self, action: ArrayLike
    ) -> tuple[NDArray, np.float32, bool, bool, dict[str, Any]]:
        """Advance one control step.

        Args:
            action: (4,) array-like of normalized motor commands in [-1, 1].

        Returns:
            obs: (20,) observation vector of the drone's state relative to the target
              gate (see course.build_obs for details).
            reward: scalar reward (positive for progress toward the target gate, plus
              optional shaping terms; negative for crashing).
            terminated: True if the episode ended due to drone crash or out-of-bounds.
            truncated: True if the episode ended due to max_steps.
            info: dict with extra info (e.g. whether the gate was passed, collision,
              visibility reward, etc) for logging/debugging but not needed by the
              policy.
        """
        actions = np.asarray(action, dtype=np.float32).reshape(4)
        pos_old = self.quad.state.position.copy()

        self.quad.apply_disc_control(actions, self.dt)
        self.world.step(render=self._render)
        self.step_count += 1
        self.quad.update_state(self.dt)
        ned = self._ned_obs_state()
        self._obs = self.course.build_obs(ned, self.target_gate)
        pos_new = self.quad.state.position

        reward = float(self.course.progress(pos_old, pos_new, self.target_gate))
        reward -= self.course.RATE_PENALTY * float(
            np.linalg.norm(self.quad.state.angular_velocity)
        )
        # perception visibility (measured for parity/logging; rewarded only if on)
        gate_ned = self.course.gate_pos[self.target_gate % self.course.num_gates]
        cos_a = perception.gate_bearing(ned[0:3], ned[6:9], gate_ned, self._fpv)
        vis = float(perception.visibility_reward(cos_a, self._fpv.fov_deg))
        reward += self.perception_weight * vis  # gate-visibility shaping (0 = off)
        # tail-first penalty, enshrined as part of PA (weight 0 suppresses it)
        reward -= self.backpedal_weight * float(
            perception.backpedal(ned[3:6], ned[6:9])
        )

        passed, collided = self.course.gate_passed(pos_old, pos_new, self.target_gate)
        self.target_gate = int(self.course.advance_gate(self.target_gate, passed))
        crashed = bool(self.course.out_of_bounds(pos_new)) or collided
        if crashed:
            reward = self.course.CRASH_REWARD
        terminated = bool(crashed)
        truncated = bool(self.step_count >= self.max_steps)
        info = {
            "gate_passed": passed,
            "target_gate": int(self.target_gate),
            "collision": crashed,
            "perception_visibility": vis,
        }
        return self._obs, np.float32(reward), terminated, truncated, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray, dict[str, Any]]:
        """Reset the drone to a (optionally randomized) start.

        Args:
            seed: random seed for reset randomization (obs and params).
            options: gym API reset options (forwarded to super().reset but not used)

        Returns:
            obs: (20,) observation vector of the drone's state relative to the target
              gate (see course.build_obs for details).
            info: dict with extra info for logging/debugging but not needed by the
              policy (empty in this implementation).
        """
        super().reset(seed=seed, options=options)
        self.step_count = 0
        if self.randomize_params:
            # Domain randomization: resample the drone's parameters each episode so the
            # policy learns to fly across the param distribution (obs is unchanged —
            # gate-relative — so the policy is robust but not param-conditioned).
            self.quad.params = np.asarray(
                self.p.randomized(self.rng, self.param_dr_pct)
            )
        # Shared spawn sampler is NED-native; convert the NED-FRD 16-vec to the
        # ENU-FLU PhysX pose.
        g, ned = self.course.sample_spawn(self.rng, randomize=self.randomize_reset)
        self.target_gate = g
        q_isaac = quat_aero_isaac(Rotation.from_euler("xyz", ned[6:9]).as_quat())
        R = Rotation.from_quat(q_isaac)
        self.quad.set_pose(vec_enu_ned(ned[0:3]), q_isaac)
        self.quad.set_linear_velocity(vec_enu_ned(ned[3:6]))
        self.quad.set_angular_velocity(R.apply(vec_flu_frd(ned[9:12])))
        self.quad.motor_w = ned[12:16].copy()
        self.quad._spin_rotors()
        self.world.step(render=self._render)
        self.quad.update_state(self.dt)
        self._update_obs()
        return self._obs, {}

    def close(self):
        """Release env resources (no-op; the app owns sim teardown)."""
