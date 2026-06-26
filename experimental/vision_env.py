"""Vision-in-the-loop racing: FPV image + proprioception -> motors (classical plant).

Extends RacerEnv with (1) rendered colored gates so the FPV camera sees something to
race toward, and (2) a flat obs = a 64x64 RGB image concatenated with a 13-dim
proprioception vector (body lin vel, body rates, projected gravity, motor states).
Gate-relative features are OMITTED — the policy must read gates from pixels. The plant,
reward, termination, and reset are inherited; only the obs and rendered gates are new.

Needs AppLauncher --enable_cameras. Train with vision_train.py (skrl CNN+MLP PPO).
"""

from __future__ import annotations

from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
import numpy as np
import torch

from isaacrace.course import RaceCourse
from isaacracelab.racer_env import RacerEnv, RacerEnvCfg

CAM_RES = 64
PROPRIO_DIM = 13  # lin_vel_b(3) + ang_vel_b(3) + projected_gravity_b(3) + motor(4)
_IMG_DIM = CAM_RES * CAM_RES * 3


def _fpv_camera_cfg(res: int) -> TiledCameraCfg:
    """Forward-looking FPV TiledCamera mounted on the racer body."""
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/fpv_cam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.12, 0.0, 0.03), rot=(0.5, -0.5, 0.5, -0.5), convention="world"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 40.0),
        ),
        width=res,
        height=res,
    )


@configclass
class VisionRacerEnvCfg(RacerEnvCfg):
    """Vision racing cfg: FPV camera on, flat image+proprio obs, racing reward."""

    observation_space = _IMG_DIM + PROPRIO_DIM  # flat [image | proprio]
    tiled_camera: TiledCameraCfg = _fpv_camera_cfg(CAM_RES)
    gate_bonus: float = 10.0  # vision needs the explicit racing signal


class VisionRacerEnv(RacerEnv):
    """Vision racing: image+proprio observation; gates rendered for the FPV camera."""

    cfg: VisionRacerEnvCfg

    def _spawn_extra_prims(self):
        """Spawn colored gate frames at the course positions (under env_0 -> cloned)."""
        from isaacsim.core.utils.prims import define_prim, get_prim_at_path
        import isaacsim.core.utils.stage as stage_utils
        from pxr import Gf, UsdGeom

        stage = stage_utils.get_current_stage()
        course = RaceCourse(self.cfg.track)
        h = course.gate_size / 2.0
        th = 0.06  # bar thickness
        span = course.gate_size
        for i in range(course.num_gates):
            pos = course.gate_pos_enu[i]
            path = f"/World/envs/env_0/Gate_{i}"
            define_prim(path, "Xform")
            xf = UsdGeom.Xformable(get_prim_at_path(path))
            xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
            xf.AddRotateZOp().Set(float(np.degrees(course.gate_yaw_enu[i])))
            color = Gf.Vec3f(0.15, 0.8, 0.2) if i == 0 else Gf.Vec3f(0.95, 0.45, 0.08)
            for name, off, half in [
                ("t", (0, 0, h), (span, th, th)),
                ("b", (0, 0, -h), (span, th, th)),
                ("l", (-h, 0, 0), (th, th, span)),
                ("r", (h, 0, 0), (th, th, span)),
            ]:
                cube = UsdGeom.Cube.Define(stage, f"{path}/{name}")
                cube.CreateSizeAttr(1.0)
                cube.CreateDisplayColorAttr([color])
                cxf = UsdGeom.Xformable(cube)
                cxf.AddTranslateOp().Set(Gf.Vec3d(*off))
                cxf.AddScaleOp().Set(Gf.Vec3d(*half))

    def _get_observations(self) -> dict:
        d = self._robot.data
        proprio = torch.cat(
            [d.root_lin_vel_b, d.root_ang_vel_b, d.projected_gravity_b, self._motor_w],
            dim=-1,
        )  # (N, 13)
        img = self._camera.data.output["rgb"].to(torch.float32) / 255.0  # (N,H,W,3)
        obs = torch.cat([img.reshape(self.num_envs, -1), proprio], dim=-1)
        return {"policy": obs}
