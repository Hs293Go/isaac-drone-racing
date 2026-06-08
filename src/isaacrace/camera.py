"""FpvCamera — a front-mounted FPV camera owned directly by the drone.

The FPV camera is USD Camera prim parented to the drone ``/body`` so it rides the pose,
and exposes that prim's path for a viewport to render. Display-only for now; a
render-product RGB capture can be added later (the on-ramp to vision-based racing).
"""

import numpy as np
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation

from isaacrace.config import FpvConfig

# USD focal-length aperture convention: hFOV = 2*atan(hAperture / (2*focal)).
_HORIZONTAL_APERTURE = 20.955


class FpvCamera:
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
        self.path = parent_prim_path + "/body/fpv_cam"
        cam = UsdGeom.Camera.Define(stage, self.path)

        focal = _HORIZONTAL_APERTURE / (
            2.0 * np.tan(np.radians(float(config.fov_deg)) / 2.0)
        )
        cam.CreateFocalLengthAttr(float(focal))
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
