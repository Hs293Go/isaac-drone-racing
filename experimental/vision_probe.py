"""Probe: dump FPV frames + gate geometry to check if gates are recoverable from pixels.

Before committing to distillation, confirm the camera actually sees gates. Saves a few
64x64 FPV frames (upscaled) from a policy rollout + gate distance, to inspect:
gates visible -> perception is fine, survive-not-race is an RL failure; empty view ->
the camera spec / mount is the bottleneck.

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab python -u \
        experimental/vision_probe.py --headless --enable_cameras
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FPV perception probe")
parser.add_argument("--ckpt", default="examples/train_out/race_vision_lrdecay_best")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
import torch  # noqa: E402
from vision_env import VisionRacerEnv, VisionRacerEnvCfg  # noqa: E402

from isaacrace.config import RaceTrackConfig  # noqa: E402

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
cfg.randomize_reset = False  # fixed start facing the course
env = VisionRacerEnv(cfg)
dev = env.device
model = PPO.load(str(ROOT / args.ckpt), device=args.device)
outdir = Path("/tmp/fpv")  # noqa: S108
outdir.mkdir(exist_ok=True)

obs = env.reset()[0]["policy"]
lines = []
for t in range(81):
    act, _ = model.predict(obs.cpu().numpy(), deterministic=True)
    obs, _r, _term, _trunc, _info = env.step(torch.as_tensor(act, device=dev))
    obs = obs["policy"]
    if t in {0, 20, 40, 60, 80}:
        rgb = env._camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(rgb).resize((256, 256), Image.NEAREST).save(
            outdir / f"fpv_t{t:02d}.png"
        )
        pos = (env._robot.data.root_pos_w - env.scene.env_origins)[0]
        tg = int(env._target_gate[0].item())
        dist = float(torch.linalg.norm(env._course.gate_pos_enu[tg] - pos))
        lines.append(f"t={t:2d}  gate{tg}  dist={dist:5.1f}m  img_std={rgb.std():.0f}")

out = "\n".join(lines)
Path("/tmp/vision_probe.txt").write_text(out + "\n", encoding="utf-8")  # noqa: S108
print("\n" + out)
env.close()
sim_app.close()
