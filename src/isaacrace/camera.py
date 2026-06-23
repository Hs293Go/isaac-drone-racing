"""USD cameras with optional replicator RGB capture (FPV + static course view).

``FpvCamera`` is a Camera prim parented to the drone ``/body`` so it rides the pose;
``CourseCamera`` is a static prim framing the whole track for the "main" recording.
Both can attach a replicator render product for RGB capture via the shared
``_CapturableCamera`` base. Note that capture works reliably in windowed mode only after
the full RTX pipeline is up.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation

from isaacrace.config import FpvConfig

# USD focal-length aperture convention: hFOV = 2*atan(hAperture / (2*focal)).
_HORIZONTAL_APERTURE = 20.955


def _focal_from_fov(fov_deg: float) -> float:
    """USD focal length giving the requested horizontal FOV (degrees)."""
    return _HORIZONTAL_APERTURE / (2.0 * np.tan(np.radians(float(fov_deg)) / 2.0))


def _look_at_quat(
    eye: ArrayLike, look_at: ArrayLike, up: ArrayLike = (0.0, 0.0, 1.0)
) -> Gf.Quatf:
    """Computes the orientation for a USB camera at `eye` to aim at `look_at`.

    USD cameras look down their local -z with +y up; this builds the body-to-world
    rotation whose columns are the camera axes (right, up, -forward) in world ENU.

    Args:
        eye: (3,) position of the camera in world ENU.
        look_at: (3,) point in world ENU for the camera to aim at.
        up: (3,) world ENU up direction (default +z). The orientation vector only
          determines two degrees of freedom, this is the "roll" reference that resolves
          the final one. Defaults to straight up (0, 0, 1)

    Returns:
        A Gf.Quatf representing the camera's orientation in world ENU.
    """
    eye = np.asarray(eye, dtype=float)
    look_at = np.asarray(look_at, dtype=float)
    up = np.asarray(up, dtype=float)
    f = look_at - eye
    f /= np.linalg.norm(f)  # forward
    s = np.cross(f, up)
    s /= np.linalg.norm(s)  # right
    u = np.cross(s, f)  # true up (orthonormal)
    q = Rotation.from_matrix(np.column_stack([s, u, -f])).as_quat()  # xyzw
    return Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))


class _CapturableCamera:
    """A USD camera prim that can attach a replicator render product for RGB capture."""

    path: str  # the camera prim path; set by each subclass's __init__

    def _init_capture(self):
        """Initialize the (lazily-attached) capture annotator to None."""
        self._annot = None

    def start_capture(self, resolution: Sequence[float]):
        """Attach a replicator render product + rgb annotator (lazy Kit import).

        ``omni.replicator.core`` is a Kit extension only importable once SimulationApp
        has booted, so this is called after ``world.reset()``. Creating the render
        product can invalidate the PhysX simulation view that the drone's tensor API
        (``RigidPrim``) reads/writes through — the old dynamic_control path was immune,
        the tensor API is not — so the capture path calls
        ``RacingDrone.reacquire_physics_view()`` afterward to rebind.

        Arg:
            resolution: (width, height) in pixels for the captured frames.
        """
        import omni.replicator.core as rep

        w, h = map(int, resolution)
        rp = rep.create.render_product(self.path, (w, h))
        self._annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annot.attach(rp)  # type: ignore

    def grab(self):
        """Return the latest frame as an (H, W, 3) uint8 RGB array (alpha dropped)."""
        if self._annot is None:
            raise RuntimeError("start_capture() must be called before grab()")
        data = np.asarray(self._annot.get_data())
        return data[..., :3] if data.ndim == 3 and data.shape[-1] == 4 else data


class FpvCamera(_CapturableCamera):
    """A forward, up-tilted FPV camera prim mounted on the drone body."""

    def __init__(self, stage: Usd.Stage, parent_prim_path: str, config: FpvConfig):
        """Create the USD Camera prim at ``<parent>/body/fpv_cam``.

        Must be constructed BEFORE ``world.reset()`` (RacingDrone does this in
        ``__init__``): adding the prim is a stage-structure change that would
        invalidate the physics tensor view if done later.

        Args:
            stage: The USD stage to define the camera on.
            parent_prim_path: The drone prim path (the camera mounts on ``/body``).
            config: A FpvConfig object containing mount/tilt_deg/fov_deg/resolution.
        """
        self._init_capture()
        self.path = parent_prim_path + "/body/fpv_cam"
        cam = UsdGeom.Camera.Define(stage, self.path)
        cam.CreateFocalLengthAttr(float(_focal_from_fov(config.fov_deg)))
        cam.CreateHorizontalApertureAttr(_HORIZONTAL_APERTURE)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 1000.0))
        x = UsdGeom.Xformable(cam.GetPrim())
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, config.mount)))

        # USD camera frame: -z forward, +y up. So -z_cam becomes x_flu, y_cam becomes
        # z_flu, and x_cam follows from the right-hand rule. Finally, tilt up about the
        # camera x-axis.
        rot = Rotation.from_matrix([
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0],
        ]) * Rotation.from_euler("x", float(config.tilt_deg), degrees=True)
        q = rot.as_quat()
        x.AddOrientOp().Set(Gf.Quatf(q[3], *q[0:3]))


class CourseCamera(_CapturableCamera):
    """A static camera framing the whole course, for the main (chase) recording."""

    def __init__(
        self, stage: Usd.Stage, path: str, eye, look_at, fov_deg: float = 60.0
    ):
        """Define a static Camera prim at `eye` aimed at `look_at` (world ENU).

        Args:
            stage: The USD stage to define the camera on.
            path: The camera prim path (e.g. ``/World/course_cam``).
            eye: World-ENU position to place the camera.
            look_at: World-ENU point to aim at (e.g. the course centroid).
            fov_deg: Horizontal field of view in degrees.
        """
        self._init_capture()
        self.path = path
        cam = UsdGeom.Camera.Define(stage, self.path)
        cam.CreateFocalLengthAttr(float(_focal_from_fov(fov_deg)))
        cam.CreateHorizontalApertureAttr(_HORIZONTAL_APERTURE)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 1000.0))
        x = UsdGeom.Xformable(cam.GetPrim())
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, eye)))
        x.AddOrientOp().Set(_look_at_quat(eye, look_at))
