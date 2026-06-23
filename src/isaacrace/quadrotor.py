"""RacingDrone — the drone vehicle.

Our racing drone is heavily inspired from Pegasus Simulator's vehicle abstraction and
follows its examples in managing the USD assets. Actuation and state reads go through
the Isaac Sim **PhysX tensor API** (``isaacsim.core.prims.RigidPrim``): Isaac Sim 6.0
removed the ``omni.isaac.dynamic_control`` extension this code used on 5.1, so the body
is now read/written through a tensor view of ``/body``. We implement a concrete racing
drone instead of a hierarchy of vehicles since this codebase is focused on racing. We
also discard support for multi-vehicle registration.
"""

from pathlib import Path
from typing import Any, get_args

from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.world import World
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
import numpy as np
from numpy.typing import ArrayLike, NDArray
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


def _as_batch(vec: ArrayLike, size: int) -> NDArray:
    """Converts a flat array-like to PhysX-batched ``(1, size)`` float array.

    This is used to give our flat vectors a leading axis for PhysX tensor API calls with
    dimension checking: ``reshape(1, size)`` it raises unless ``vec`` holds exactly
    ``size`` elements.
    """
    return np.asarray(vec, dtype=float).reshape(1, size)


def _from_batch(arr: Any, size: int) -> NDArray:
    """Converts a PhysX-batched ``(1, size)`` float array to a flat array.

    Raises if the tensor view holds more than one row, surfacing a mis-scoped view
    (e.g. ``/body`` matching several articulation links) instead of silently using
    row 0. ``arr`` is ``Any`` because the tensor-API getters return Warp
    ``indexedarray`` views, which ``np.asarray`` accepts but which aren't members of
    numpy's ``ArrayLike`` union.
    """
    arr = np.asarray(arr, dtype=float).reshape(-1, size)
    if arr.shape[0] != 1:
        raise ValueError(f"Expected one {size}-element row, but got {arr.shape[0]}.")
    return arr[0]


def _to_sized_list_float(arr: ArrayLike, size: int) -> list[float]:
    """Convert an array-like to a list of floats of a specific size."""
    return _as_batch(arr, size).ravel().tolist()


def _xyzw_to_wxyz(q: ArrayLike) -> np.ndarray:
    """scipy/USD-state [x,y,z,w] -> Isaac tensor-API scalar-first [w,x,y,z]."""
    q = np.asarray(q, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]])


def _wxyz_to_xyzw(q: ArrayLike) -> np.ndarray:
    """Isaac tensor-API scalar-first [w,x,y,z] -> scipy/USD-state [x,y,z,w]."""
    q = np.asarray(q, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]])


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
            position=_to_sized_list_float(init_pos, 3),
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
        self._rigid = None  # PhysX tensor view of /body (lazy, post world.reset)

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
    def _rigid_body(self) -> RigidPrim:
        """Lazily build + initialize the PhysX tensor view of /body; cache body mass.

        Replaces dynamic_control's ``get_rigid_body`` handle. ``RigidPrim`` reads/writes
        the body through PhysX's tensor view, which exists only once the physics
        simulation view is live (after ``world.reset()``). The first caller is
        ``env.reset()`` — well after reset — so ``initialize()`` always finds a live
        view. Mass is read once here, mirroring the old handle's first-access caching.

        Note: ``racer.usd`` is an articulation (``PhysicsArticulationRootAPI``) and
        ``/body`` is its base link. dynamic_control drove that link as a plain rigid
        body; this rigid-body tensor view does the same (NVIDIA's documented
        dynamic_control replacement). If a future on-hardware check shows PhysX 6.0
        won't drive an articulation link through a rigid-body view, switch these
        reads/writes to the inherited ``SingleArticulation`` API — ``self`` is a
        ``Robot``, so ``self.get_world_pose`` / ``set_world_pose`` /
        ``get_linear_velocity`` / ``set_linear_velocity`` / ``get_angular_velocity`` /
        ``set_angular_velocity`` already drive the same base link.
        """
        if self._rigid is None:
            rb = RigidPrim(self.stage_prefix + "/body", name="drone_body_view")
            rb.initialize()  # bind to the live physics sim view (post world.reset)
            self._rigid = rb
            self.mass = _from_batch(rb.get_masses(), 1).item()
        return self._rigid

    def reacquire_physics_view(self):
        """Drop the cached tensor view so the next access rebinds it.

        Creating a replicator render product (the FPV/course capture path) is a
        stage-structure change that can invalidate the PhysX simulation view the
        tensor API reads/writes through. dynamic_control was immune to this; the
        tensor API is not, so the capture path calls this after ``start_capture()``.
        Harmless if the view is still valid — it just rebuilds an equivalent one.
        """
        self._rigid = None

    def update_state(self, _dt: float = 0.0):
        """Reads pose and velocities from PhysX into self.state with frame conversions.

        This mirrors Pegasus update_state without using the time step, since we don't
        need a finite-diff acceleration.
        """
        rb = self._rigid_body()
        pos, quat_wxyz = rb.get_world_poses()  # ENU position, FLU->ENU wxyz quaternion
        pos = _from_batch(pos, 3)
        quat_xyzw = _wxyz_to_xyzw(_from_batch(quat_wxyz, 4))
        lin = _from_batch(rb.get_linear_velocities(), 3)  # ENU world
        ang_w = _from_batch(rb.get_angular_velocities(), 3)  # ENU world
        R = Rotation.from_quat(quat_xyzw)  # FLU->ENU
        self.state.position = pos
        self.state.attitude = quat_xyzw
        self.state.linear_velocity = lin
        self.state.angular_velocity = R.inv().apply(ang_w)  # body FLU rates

    def apply_force(self, force: ArrayLike, pos: ArrayLike = (0.0, 0.0, 0.0)):
        """Apply a body-frame force to the drone body at a body-frame offset.

        Args:
            force: A 3D vector representing the force to be applied, expressed in the
              body frame of the drone (FLU convention).
            pos: A 3D vector representing the position of the application point of the
              force relative to the center of mass of the body, expressed in the body
              frame of the drone (FLU convention).
        """
        # is_global=False -> forces and positions are in the body's local frame, exactly
        # like dynamic_control.apply_body_force(..., is_global=False). PhysX clears
        # applied forces each step, so (as before) this must be re-issued every step.
        self._rigid_body().apply_forces_and_torques_at_pos(
            forces=_as_batch(force, 3), positions=_as_batch(pos, 3), is_global=False
        )

    def apply_torque(self, torque: ArrayLike):
        """Apply a body-frame (FLU) torque to the drone body via the tensor API.

        Args:
            torque: A 3D body-frame (FLU) torque vector (N*m).
        """
        # Torque-only wrench at the COM (is_global=False -> body frame); repeated
        # tensor-API wrench writes accumulate within a step (like dynamic_control), so
        # this adds to the per-rotor apply_force() writes.
        self._rigid_body().apply_forces_and_torques_at_pos(
            torques=_as_batch(torque, 3), is_global=False
        )

    def set_pose(self, pos: ArrayLike, quat_xyzw: ArrayLike):
        """Set the body pose of the drone via the tensor API.

        Args:
            pos: A 3D vector representing the position of the drone's body in the world
              frame (ENU convention).
            quat_xyzw: A 4D vector representing the orientation of the drone as a
              quaternion in the format [qx, qy, qz, qw], where qw is the scalar part,
              representing a body (FLU)-to-world (ENU) rotation.
        """
        # RigidPrim poses are scalar-first (w,x,y,z); our state/scipy quats are xyzw.
        quat_wxyz = _xyzw_to_wxyz(quat_xyzw)
        self._rigid_body().set_world_poses(
            positions=_as_batch(pos, 3), orientations=_as_batch(quat_wxyz, 4)
        )

    def set_linear_velocity(self, velocity: ArrayLike):
        """Set the world-frame (ENU) linear velocity."""
        self._rigid_body().set_linear_velocities(_as_batch(velocity, 3))

    def set_angular_velocity(self, velocity: ArrayLike):
        """Set the world-frame (ENU) angular velocity."""
        # Isaac's angular velocity set/get methods are world, not body.
        self._rigid_body().set_angular_velocities(_as_batch(velocity, 3))

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
        self._rigid_body()  # ensure self.mass is populated
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
