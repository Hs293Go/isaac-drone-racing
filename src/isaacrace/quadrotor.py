"""RacingDrone — the drone vehicle.

Our racing drone is heavily inspired from Pegasus Simulator's vehicle abstraction and
follows its examples in managing the USD assets, and implementing physical actuation via
dynamic_control and state updates. However, we implement a concrete racing drone instead
of a hierarchy of vehicles since this codebase is focused on racing. We also discard
support for multi-vehicle registration.
"""

from pathlib import Path
from typing import get_args

import carb
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.world import World
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
import numpy as np
from numpy.typing import ArrayLike, NDArray
from omni.isaac.dynamic_control import _dynamic_control  # noqa: PLC2701
from pxr import UsdGeom
from scipy.spatial.transform import Rotation

from isaacrace.camera import FpvCamera, FpvConfig
from isaacrace.config import DynamicsMode
from isaacrace.conversions import vec_flu_frd
import isaacrace.dynamics as dyn
from isaacrace.state import State

RACER_USD = str(Path(__file__).parent / "assets" / "racer.usd")

# --- classical-plant 5" airframe (used only by dynamics_mode="classical") ---
# Our own 5" racer. Refer to github.com/Hs293Go/source_one_SolidWorks.git for the CAD
# and derivation of these numbers.
_ARM_X = 0.1008  # m, forward motor offset (extracted)
_ARM_Y = 0.0766  # m, lateral motor offset (extracted)
# Betaflight-layout rotor positions in the FLU body frame (x fwd, y left, z up),
# matching the modeling heritage of optimal_quad_control_rl.
_ROTOR_POS_FLU = np.array([
    [-_ARM_X, -_ARM_Y, 0.0],  # rotor 0: back-right
    [+_ARM_X, -_ARM_Y, 0.0],  # rotor 1: front-right
    [-_ARM_X, +_ARM_Y, 0.0],  # rotor 2: back-left
    [+_ARM_X, +_ARM_Y, 0.0],  # rotor 3: front-left
])
# Per-rotor yaw-reaction sign in FRD-z from the betaflight spin convention; converted to
# FLU at apply time.
_YAW_SIGN_FRD = np.array([-1.0, +1.0, +1.0, -1.0])
_KAPPA = 0.0157  # cq/ct: generic 5" reaction-to-thrust ratio (prop unmeasured)
# Rotor+prop polar inertia about the spin axis (kg*m^2): the rotor angular-momentum
# reaction torque a pure rigid-body plant omits. The value is identified k_r5 (1.97e-3)
# x CAD body yaw inertia Jz (3.14e-3)
_I_ROTOR = 6.19e-6


def _to_list_float(arr: ArrayLike) -> list[float]:
    """Convert an array-like to a list of floats."""
    return np.asarray(arr, dtype=float).tolist()


def _to_sized_list_float(arr: ArrayLike, size: int) -> list[float]:
    """Convert an array-like to a list of floats of a specific size."""
    arr = np.asarray(arr, dtype=float)
    if arr.size != size:
        raise ValueError(f"Input array must have size {size}, but got {arr.size}.")
    return arr.tolist()


class RacingDrone(Robot):
    """One drone: SourceOne racer + FPV camera + physics, driven step-by-step by env."""

    def __init__(
        self,
        world: World,
        init_pos: ArrayLike,
        params: ArrayLike,
        fpv: FpvConfig | None = None,
        stage_prefix: str = "/World/drone",
        usd_file: str = RACER_USD,
        dynamics_mode: DynamicsMode = "kinematic",
    ):
        """Construct the racing drone.

        Arg:
            world: The Isaac Sim World (already created; reset() is the caller's
              responsibility, AFTER this).
            init_pos: The spawn position of the drone's body in the world frame (ENU);
              the real spawn pose is set each episode in env.reset().
            params: The RL parameter vector (dynamics.params_vec()).
            fpv: Optional FPV camera config; when enabled, a drone.camera is created
              here, before reset().
            stage_prefix: USD path for the drone.
            usd_file: The vendored drone USD (default: our SourceOne racer.usd).
            dynamics_mode: Isaac plant model, either:
              - "kinematic": overrides rotational dynamics to be based on TU Delft's
                motor effectiveness/INDI model
              - "classical": per-rotor forces through a real inertia tensor;
        """
        # Reference the drone USD onto a fresh prim, then wrap it as a Robot
        # (orientation is w-first for NVIDIA's convention; the level spawn here is
        # overwritten by env.reset()).
        define_prim(stage_prefix, "Xform").GetReferences().AddReference(usd_file)
        super().__init__(
            prim_path=stage_prefix,
            name="drone",
            position=_to_list_float(init_pos),
            orientation=[1.0, 0.0, 0.0, 0.0],
        )
        world.scene.add(self)  # ty:ignore[invalid-argument-type]

        self.world = world
        self.stage = world.stage
        self.stage_prefix = stage_prefix
        self.params = np.asarray(params)
        if dynamics_mode not in (valid_dynamics_modes := get_args(DynamicsMode)):
            raise ValueError(
                f"Invalid dynamics_mode: {dynamics_mode} ∉ {valid_dynamics_modes}"
            )
        self.dynamics_mode = dynamics_mode
        self.state = State()
        self.motor_w = np.zeros(4)
        self.mass = None
        self._dc = None
        self._body_handle_cache = None

        # Visual prop-spin ops + the FPV camera prim are stage-structure changes ->
        # add them BEFORE the caller's world.reset(), or the physics tensor view is
        # invalidated.
        self._setup_rotors()
        # First-class FPV camera (None unless enabled).
        self.camera = (
            FpvCamera(self.stage, self.stage_prefix, fpv)
            if (fpv and fpv.enabled)
            else None
        )

    # ----------------------------------------------------- low-level handles
    def get_dc_interface(self):
        """Lazily acquire and cache the dynamic_control interface (mirrors Pegasus)."""
        if self._dc is None:
            self._dc = _dynamic_control.acquire_dynamic_control_interface()
        return self._dc

    def _body(self):
        """Cached rigid-body handle for /body; also caches body mass on first access."""
        if self._body_handle_cache is None:
            self._body_handle_cache = self.get_dc_interface().get_rigid_body(
                self.stage_prefix + "/body"
            )
            self.mass = float(
                self
                .get_dc_interface()
                .get_rigid_body_properties(self._body_handle_cache)
                .mass
            )
        return self._body_handle_cache

    def update_state(self, _dt: float = 0.0):
        """Reads pose and velocities from PhysX into self.state with frame conversions.

        This mirrors Pegasus update_state without using the time step, since we don't
        need a finite-diff acceleration.
        """
        dc = self.get_dc_interface()
        body = self._body()
        pose = dc.get_rigid_body_pose(body)
        lin = np.array(dc.get_rigid_body_linear_velocity(body))  # ENU world
        ang_w = np.array(dc.get_rigid_body_angular_velocity(body))  # ENU world
        R = Rotation.from_quat([pose.r.x, pose.r.y, pose.r.z, pose.r.w])  # FLU->ENU
        self.state.position = np.array(pose.p)
        self.state.attitude = np.array([pose.r.x, pose.r.y, pose.r.z, pose.r.w])
        self.state.linear_velocity = lin
        self.state.angular_velocity = R.inv().apply(ang_w)  # body FLU rates

    def apply_force(
        self, force: ArrayLike, pos: ArrayLike = (0.0, 0.0, 0.0), body_part="/body"
    ):
        """Apply a body-frame force to a rigid body part.

        Args:
            force: A 3D vector representing the force to be applied, expressed in the
              body frame of the drone (FLU convention).
            pos: A 3D vector representing the position of the application point of the
              force relative to the center of mass of the body part, expressed in the
              body frame of the drone (FLU convention).
            body_part: The path to the body part to which the force should be applied,
              relative to the drone's stage prefix.
        """
        force = _to_sized_list_float(force, 3)
        pos = _to_sized_list_float(pos, 3)
        dc = self.get_dc_interface()
        rb = (
            self._body()
            if body_part == "/body"
            else dc.get_rigid_body(self.stage_prefix + body_part)
        )
        dc.apply_body_force(rb, carb.Float3(force), carb.Float3(pos), False)

    def apply_torque(self, torque: ArrayLike, body_part="/body"):
        """Apply a body-frame (FLU) torque to a rigid body part via dynamic_control.

        Args:
            torque: A 3D body-frame (FLU) torque vector (N*m).
            body_part: Path to the body part, relative to the drone's stage prefix.
        """
        torque = _to_sized_list_float(torque, 3)
        dc = self.get_dc_interface()
        rb = (
            self._body()
            if body_part == "/body"
            else dc.get_rigid_body(self.stage_prefix + body_part)
        )
        dc.apply_body_torque(rb, carb.Float3(torque), False)

    def set_pose(self, pos: ArrayLike, quat_xyzw: ArrayLike):
        """Set the body pose of the drone via dynamic_control.

        Args:
            pos: A 3D vector representing the position of the drone's body in the world
              frame (ENU convention).
            quat_xyzw: A 4D vector representing the orientation of the drone as a
              quaternion in the format [qx, qy, qz, qw], where qw is the scalar part,
              representing a body (FLU)-to-world (ENU) rotation.
        """
        self.get_dc_interface().set_rigid_body_pose(
            self._body(),
            _dynamic_control.Transform(
                _to_sized_list_float(pos, 3), _to_sized_list_float(quat_xyzw, 4)
            ),
        )

    def set_linear_velocity(self, velocity: ArrayLike):
        """Set the world-frame (ENU) linear velocity."""
        self.get_dc_interface().set_rigid_body_linear_velocity(
            self._body(), carb.Float3(_to_sized_list_float(velocity, 3))
        )

    def set_angular_velocity(self, velocity: ArrayLike):
        """Set the world-frame (ENU) angular velocity."""
        # Isaac's angular velocity set/get methods are world, not body.
        self.get_dc_interface().set_rigid_body_angular_velocity(
            self._body(), carb.Float3(_to_sized_list_float(velocity, 3))
        )

    # ----------------------------------------------- faithful physics (per RL step)
    #
    def apply_disc_control(self, control_actions: ArrayLike, dt: float):
        """Advance the rotor state and actuate the body for one RL step.

        Dispatches on ``dynamics_mode``: "kinematic" is the faithful TU Delft Model
        splitting PhysX translation motion and kinematic-only rotation; "classical"
        places per-rotor thrust forces at the arms and lets PhysX integrate rotation.
        Either way the rotor command is the RL ``actions`` the env injects.

        Args:
            control_actions: The 4D vector of rotor commands (normalized [-1,1]) from
              the RL policy.
            dt: The time step duration, used to advance the motor state and apply the
              physics forward.
        """
        self._body()  # ensure self.mass is populated
        control_actions = np.asarray(control_actions)
        d_w, _ = dyn.motor_derivative(self.motor_w, control_actions, self.params)
        self.motor_w = np.clip(self.motor_w + dt * d_w, -1.0, 1.0)
        if self.dynamics_mode == "classical":
            self._apply_classical(control_actions)
        else:
            self._apply_kinematic(control_actions, dt)
        self._spin_rotors()

    def _apply_kinematic(self, control_actions: NDArray, dt: float):
        """Faithful TU Delft model: PhysX translation force + kinematic rotation."""
        s_op = self.state
        body2world = Rotation.from_quat(s_op.attitude)

        # Compute FRD-frame body velocity since the model uses it to compute gyroscopic
        # drag as part of the total force/torque
        vb_frd = vec_flu_frd(body2world.inv().apply(s_op.linear_velocity))
        accel_frd, alpha_frd = dyn.body_force_torque_accel(
            self.motor_w, control_actions, vb_frd, self.params
        )

        force_flu = self.mass * vec_flu_frd(accel_frd)
        rates_flu = s_op.angular_velocity + dt * vec_flu_frd(alpha_frd)
        self.apply_force(force_flu)
        self.set_angular_velocity(body2world.apply(rates_flu))

    def _apply_classical(self, control_actions: NDArray):
        """Classical rigid-body plant.

        Applies thrust force at each arm tip, so PhysX forms each rotor's roll/pitch
        torque through its own lever arm calculation. The yaw reaction torque is applied
        as a pure couple.

        While translational thrust is handled as above, translational drag is applied
        separately.
        """
        s_op = self.state
        body2world = Rotation.from_quat(s_op.attitude)

        # Per-rotor thrust (N) along body-up at each arm; PhysX forms r x F -> torque.
        thrust = self.mass * dyn.rotor_thrusts(self.motor_w, self.params)  # (4,)
        for i in range(4):
            self.apply_force([0.0, 0.0, float(thrust[i])], pos=_ROTOR_POS_FLU[i])

        # Yaw reaction couple: steady prop drag (kappa*thrust) plus the rotor
        # angular-momentum reaction I_rotor*dW.
        #
        # Compute motor-wise yaw reactions in the FRD frame, then sum and convert to FLU
        # for application.
        #
        # optimal_quad_control_rl accounts for the angular-momentum reaction in its
        # k_r[5-8] * dW term, which a rigid-body plant omits. This is significant on
        # racers, and is exploited by the policy for fast yaw-authority.
        _, d_w_rps2 = dyn.motor_derivative(self.motor_w, control_actions, self.params)
        yaw_frd = float(np.sum(_YAW_SIGN_FRD * (_KAPPA * thrust + _I_ROTOR * d_w_rps2)))
        self.apply_torque(vec_flu_frd(np.array([0.0, 0.0, yaw_frd])))

        # Velocity-dependent rotor drag (FRD x,y only; z is the collective, above).
        vb_frd = vec_flu_frd(body2world.inv().apply(s_op.linear_velocity))
        accel_frd, _ = dyn.body_force_torque_accel(
            self.motor_w, control_actions, vb_frd, self.params
        )
        drag_frd = np.array([accel_frd[0], accel_frd[1], 0.0])
        self.apply_force(self.mass * vec_flu_frd(drag_frd))

    def start(self):
        """Hook invoked when the simulation starts. No-op."""

    def stop(self):
        """Hook invoked when the simulation stops. No-op."""

    # ----------------------------------------------- propeller spin (visual only)
    def _setup_rotors(self):
        """Add a visual-only spin op (outermost rotateZ) to each propeller mesh.

        Spinning the *mesh* (a non-physics child of the rotor link) leaves the
        props fully decoupled from body dynamics, so flight is identical (driving
        the physics joints instead reacts torque into the body, and PhysX caps
        joint velocity ~100 rad/s — far below the true motor speed).
        """
        self._rotor_angle = np.zeros(4)
        self._rotor_spin = np.array([1.0, 1.0, -1.0, -1.0])  # rotor0,1 CCW; rotor2,3 CW
        self._rotor_ops = []
        for i in range(4):
            prim = get_prim_at_path(f"{self.stage_prefix}/rotor{i}/prop")
            if not (prim and prim.IsValid()):
                self._rotor_ops = []
                return
            x = UsdGeom.Xformable(prim)
            existing = list(x.GetOrderedXformOps())
            op = x.AddRotateZOp(opSuffix="spin")
            x.SetXformOpOrder([
                op,
                *existing,
            ])  # outermost -> spin the positioned prop about rotor Z
            self._rotor_ops.append(op)

    def _spin_rotors(self):
        """Advance each prop's visual angle at its true motor speed.

        W = motor_W(motor_w) (cosmetic). At the real rate this strobes under the
        frame rate — faithful, not a bug.
        """
        if not self._rotor_ops:
            return
        W = dyn.unproject_motor(self.motor_w)
        self._rotor_angle = np.mod(
            self._rotor_angle + self._rotor_spin * W * dyn.DT, 2 * np.pi
        )
        for op, ang in zip(self._rotor_ops, self._rotor_angle, strict=False):
            op.Set(float(np.degrees(ang)))
