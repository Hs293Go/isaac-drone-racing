"""Verify gate-corner projection -- the foundation for corner-based yaw perception.

Corner detection beats direct yaw regression only if the corner LABELS are right. This
projects the gate's 4 corners into the FPV image (pose+intrinsics from Isaac) and
overlays them on rendered frames. If the markers land on the gate in the saved PNGs
(/tmp/corners/), the projection is correct and a corner detector can train on it.

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab python -u \
        experimental/corner_verify.py --headless --enable_cameras
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Gate-corner projection check")
parser.add_argument("--control", default="train_out/race_ppo_classical_ft.zip")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

from isaaclab.utils.math import quat_rotate_inverse  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
import torch  # noqa: E402
from vision_env import VisionRacerEnv, VisionRacerEnvCfg  # noqa: E402

from isaacrace.config import RaceTrackConfig  # noqa: E402
from isaacrace.course import RaceCourse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
track = RaceTrackConfig.from_dict(
    OmegaConf.to_container(
        OmegaConf.load(ROOT / "examples/conf/track/figure8.yaml"), resolve=True
    )
)
cfg = VisionRacerEnvCfg()
cfg.track = track
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.randomize_reset = False  # fixed start facing gate 0
env = VisionRacerEnv(cfg)
dev = env.device
control = PPO.load(str(ROOT / args.control), device="cpu")

course = RaceCourse(track)
GATE_POS = torch.as_tensor(course.gate_pos_enu, dtype=torch.float32, device=dev)
GATE_YAW = torch.as_tensor(course.gate_yaw_enu, dtype=torch.float32, device=dev)
H = course.gate_size / 2.0
CORNERS_LOCAL = [(-H, -H), (-H, H), (H, -H), (H, H)]  # (lateral x, vertical z)
outdir = Path("/tmp/corners")  # noqa: S108
outdir.mkdir(exist_ok=True)


def project():
    """Project each env's target-gate corners to FPV pixels. Returns (N,4) u, v, z."""
    cam = env._camera.data
    tg = env._target_gate
    gpos = GATE_POS[tg] + env.scene.env_origins  # course-local -> world
    gyaw = GATE_YAW[tg]
    cphi, sphi, one = torch.cos(gyaw), torch.sin(gyaw), torch.ones_like(gyaw)
    world = torch.stack(
        [
            gpos + torch.stack([cx * cphi, cx * sphi, cz * one], dim=-1)
            for cx, cz in CORNERS_LOCAL
        ],
        dim=1,
    )  # (N,4,3)
    rel = (world - cam.pos_w.unsqueeze(1)).reshape(-1, 3)  # (N*4,3)
    quat = cam.quat_w_ros.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)  # ROS optical
    pcam = quat_rotate_inverse(quat, rel).reshape(-1, 4, 3)  # (N,4,3) camera frame
    uvw = torch.einsum("nij,nkj->nki", cam.intrinsic_matrices, pcam)  # (N,4,3)
    z = uvw[..., 2]  # depth; this rig's optical forward is -z (in front: z<0)
    return uvw[..., 0] / z, uvw[..., 1] / z, z


def overlay(rgb, u, v, z, res):
    """Mark each in-front, in-frame projected corner. rgb: (H,W,3) uint8."""
    img = rgb.copy()
    for k in range(4):
        if abs(z[k].item()) < 1e-4:
            continue
        cu, cv = round(u[k].item()), round(v[k].item())
        if 0 <= cu < res and 0 <= cv < res:
            img[max(cv - 1, 0) : cv + 2, max(cu - 1, 0) : cu + 2] = (255, 0, 255)
    return img


obs = env.reset()[0]["policy"]
res = env.cfg.tiled_camera.width
lines = []
for t in range(46):
    act, _ = control.predict(env._state_obs().cpu().numpy(), deterministic=True)
    obs = env.step(torch.as_tensor(act, device=dev))[0]["policy"]
    if t in {0, 15, 30, 45}:
        u, v, z = project()
        rgb = env._camera.data.output["rgb"][..., :3].to(torch.uint8).cpu().numpy()
        marked = overlay(rgb[0], u[0], v[0], z[0], res)
        Image.fromarray(marked).resize((256, 256), Image.NEAREST).save(
            outdir / f"corners_t{t:02d}.png"
        )
        infront = int((z[0] < 0).sum())  # optical forward is -z
        inframe = int(((u[0] >= 0) & (u[0] < res) & (v[0] >= 0) & (v[0] < res)).sum())
        lines.append(
            f"t={t:2d}  corners in-front={infront}/4  in-frame={inframe}/4  "
            f"u={[round(x, 1) for x in u[0].tolist()]}"
        )

out = "\n".join(lines)
Path("/tmp/corner_verify.txt").write_text(out + "\n", encoding="utf-8")  # noqa: S108
print("\n" + out)
env.close()
sim_app.close()
