"""RaceEnv — a Gymnasium env: a Quadrotor racing a figure-8 gate course.

The env is a thin gym wrapper around the drone: it owns the World, the scene
(lighting/ground/gates), the RaceCourse, and the gym loop + reward/termination —
and delegates everything drone-shaped to a `RacingDrone` (which owns the drone
prim, a first-class FPV `camera`, the `State`, and the OQCRL faithful physics; see
quadrotor.py). The env drives the drone explicitly each step (apply_disc_control ->
world.step -> update_state) instead of via World physics callbacks, because the RL
action must be injected per step.

Frames: the env is ENU-native (Isaac's world) — gates, reward, reset, and bounds
are all ENU/FLU. The lone NED is inside the observation (the trained policies are
forever-NED): _ned_obs_state reads the quad's State through its NED-FRD accessors
and feeds build_obs, and nowhere else.

A SimulationApp must already be running before this module's isaacsim imports
execute (see demo/train).
"""

from typing import ClassVar

import gymnasium as gym
from gymnasium import spaces
from isaacsim.core.api.world import World
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
import numpy as np
from numpy.typing import NDArray
import omni.timeline
from pxr import Gf, UsdGeom, UsdLux
from scipy.spatial.transform import Rotation

from isaacrace.conversions import quat_aero_isaac, vec_enu_ned, vec_flu_frd
from isaacrace.course import RaceCourse
import isaacrace.dynamics as dyn
from isaacrace.quadrotor import RacingDrone


class RaceEnv(gym.Env):
    """obs: 20-dim gate-relative state (pos/vel/att/rates/rpms + next gate).

    action: 4 motor cmds in [-1,1].
    """

    metadata: ClassVar[dict] = {"render_modes": ["human"]}

    def __init__(
        self,
        course: RaceCourse,
        headless=True,
        randomize_reset=True,
        seed=0,
        max_steps=1200,
        fpv=None,
        randomize_params=False,
        param_dr_pct=0.1,
    ):
        super().__init__()
        self.course = course
        self.randomize_reset = randomize_reset
        self.randomize_params = randomize_params
        self.param_dr_pct = param_dr_pct
        self.rng = np.random.default_rng(seed)
        self.p = dyn.params_vec()
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
        self.quad = RacingDrone(self.world, self.course.start_pos_enu, self.p, fpv=fpv)
        self.world.reset()
        # dynamic_control reads/writes need the timeline playing; start it once
        # here so the env is self-contained (demo/train just construct it and use
        # the gym API — no external play()).
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

        From the quad's State + rotor speeds. This is the single place the
        forever-NED policy convention is honored; byte-identical to the old env, so
        the vendored racer flies unchanged.
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
    def step(self, action):
        """Advance one control step; return the gym 5-tuple."""
        actions = np.asarray(action, dtype=np.float32).reshape(4)
        pos_old = self.quad.state.position.copy()

        self.quad.apply_disc_control(actions, self.dt)
        self.world.step(render=self._render)
        self.step_count += 1
        self.quad.update_state(self.dt)
        self._update_obs()
        pos_new = self.quad.state.position

        gp = self.course.gate_pos_enu[self.target_gate % self.course.num_gates]
        reward = float(np.linalg.norm(pos_old - gp) - np.linalg.norm(pos_new - gp))
        reward -= 0.001 * float(np.linalg.norm(self.quad.state.angular_velocity))

        passed, collided = self.course.gate_passed(pos_old, pos_new, self.target_gate)
        if passed:
            self.target_gate = (self.target_gate + 1) % self.course.num_gates
        ground = bool(pos_new[2] < 0.0)  # ENU z up: below the ground plane
        oob = bool(np.any(np.abs(pos_new[0:2]) > 5) or pos_new[2] > 7)
        if ground or oob or collided:
            reward = -10.0
        terminated = bool(ground or oob or collided)
        truncated = bool(self.step_count >= self.max_steps)
        info = {
            "gate_passed": passed,
            "target_gate": int(self.target_gate),
            "collision": bool(collided or ground or oob),
        }
        return self._obs, np.float32(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the drone to a (optionally randomized) start; return (obs, info)."""
        super().reset(seed=seed)
        self.step_count = 0
        if self.randomize_params:
            # Domain randomization: resample the drone's OQCRL params each episode so
            # the policy learns to fly across the param distribution (obs is unchanged
            # — gate-relative — so the policy is robust but not param-conditioned).
            self.quad.params = np.asarray(
                self.p.randomized(self.rng, self.param_dr_pct)
            )
        if self.randomize_reset:
            g = int(self.rng.integers(0, self.course.num_gates))
            self.target_gate = g
            # 1 m before the gate, along -through-direction (ENU)
            pos = self.course.gate_pos_enu[g].astype(
                float
            ) - self.course.gate_normal_enu[g].astype(float)
            vel = self.rng.uniform(-0.5, 0.5, 3)
            R = Rotation.from_euler(
                "xyz",
                [
                    self.rng.uniform(-np.pi / 9, np.pi / 9),
                    self.rng.uniform(-np.pi / 9, np.pi / 9),
                    self.rng.uniform(-np.pi, np.pi),
                ],
            )  # ENU-FLU attitude: small tilt, random heading
            rates_flu = self.rng.uniform(-0.1, 0.1, 3)
            motor_w = self.rng.uniform(-1.0, 1.0, 4)
        else:
            self.target_gate = 0
            pos = self.course.start_pos_enu.astype(float)
            vel, rates_flu = np.zeros(3, dtype=float), np.zeros(3)
            motor_w = np.zeros(4)
            R = Rotation.from_euler(
                "z", np.pi / 2
            )  # level, facing gate 0 (== old NED euler=0 start)

        self.quad.set_pose(pos, R.as_quat())
        self.quad.set_linear_velocity(vel)
        self.quad.set_angular_velocity(R.apply(np.asarray(rates_flu, float)))
        self.quad.motor_w = motor_w
        self.quad._spin_rotors()
        self.world.step(render=self._render)
        self.quad.update_state(self.dt)
        self._update_obs()
        return self._obs, {}

    def close(self):
        """Release env resources (no-op; the app owns sim teardown)."""
