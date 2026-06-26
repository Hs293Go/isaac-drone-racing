"""Verify VisionRacerEnv: builds, renders gates, produces image+proprio obs.

The key signal is image std > 0 with real range — if the gates render, the FPV view has
color structure; an all-black image (std ~0) means it sees nothing to race toward.

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab python -u \
        experimental/vision_check.py --num_envs 16 --headless --enable_cameras
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VisionRacerEnv obs check")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

from omegaconf import OmegaConf  # noqa: E402
import torch  # noqa: E402
from vision_env import (  # noqa: E402
    CAM_RES,
    PROPRIO_DIM,
    VisionRacerEnv,
    VisionRacerEnvCfg,
)

from isaacrace.config import RaceTrackConfig  # noqa: E402

IMG_DIM = CAM_RES * CAM_RES * 3
track_yaml = Path(__file__).resolve().parents[1] / "examples/conf/track/figure8.yaml"
track = RaceTrackConfig.from_dict(
    OmegaConf.to_container(OmegaConf.load(track_yaml), resolve=True)
)

cfg = VisionRacerEnvCfg()
cfg.track = track
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
env = VisionRacerEnv(cfg)

obs = env.reset()[0]["policy"]
for _ in range(15):  # let the camera render a few frames
    act = torch.rand((env.num_envs, 4), device=env.device) * 2 - 1
    obs = env.step(act)[0]["policy"]

img = obs[:, :IMG_DIM].reshape(env.num_envs, CAM_RES, CAM_RES, 3)
proprio = obs[:, IMG_DIM:]
lines = [
    f"obs shape   = {tuple(obs.shape)}   (expect (N, {IMG_DIM + PROPRIO_DIM}))",
    (
        f"image       = {tuple(img.shape)}  mean {img.mean():.3f}  std {img.std():.3f}"
        f"  min {img.min():.3f}  max {img.max():.3f}"
    ),
    f"proprio     = {tuple(proprio.shape)}  (expect (N, {PROPRIO_DIM}))",
    f"image std>0 (gates visible)? {'YES' if float(img.std()) > 1e-3 else 'NO'}",
]
out = "\n".join(lines)
Path("/tmp/vision_check.txt").write_text(out + "\n", encoding="utf-8")  # noqa: S108
print("\n" + out)
env.close()
sim_app.close()
