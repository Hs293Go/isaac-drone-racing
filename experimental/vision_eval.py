"""Deterministic eval of a vision policy in VisionRacerEnv (fixed ICs, multi-episode).

Reports mean gates, full-lap rate (>=8 gates = one full 8-gate lap), survival, and mean
episode length over a fixed-seed spawn set with deterministic actions — the smoothed,
low-noise metric the brief wants for checkpoint selection and the "stable + racing" bar
(NOT a single training rollout's ep_len, which is the noise we're fighting).

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab python -u \
        experimental/vision_eval.py --ckpt <ckpt> --headless --enable_cameras
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deterministic vision-policy eval")
parser.add_argument("--ckpt", default="examples/train_out/race_vision_trust_best")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stochastic", action="store_true")  # sample vs deterministic mean
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

from omegaconf import OmegaConf  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
import torch  # noqa: E402
from vision_env import VisionRacerEnv, VisionRacerEnvCfg  # noqa: E402

from isaacrace.config import RaceTrackConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
torch.manual_seed(args.seed)  # fix the spawn IC set so evals are comparable

track = RaceTrackConfig.from_dict(
    OmegaConf.to_container(
        OmegaConf.load(ROOT / "examples/conf/track/figure8.yaml"), resolve=True
    )
)
cfg = VisionRacerEnvCfg()
cfg.track = track
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
env = VisionRacerEnv(cfg)
dev = env.device
model = PPO.load(str(ROOT / args.ckpt), device=args.device)
HORIZON = int(env.max_episode_length) + 2

obs = env.reset()[0]["policy"]
n = env.num_envs
recorded = torch.zeros(n, dtype=torch.bool, device=dev)
gates = torch.zeros(n, device=dev)
steps = torch.full((n,), float(HORIZON), device=dev)
survived = torch.zeros(n, dtype=torch.bool, device=dev)
for t in range(HORIZON):
    act, _ = model.predict(obs.cpu().numpy(), deterministic=not args.stochastic)
    obs, _r, term, trunc, info = env.step(torch.as_tensor(act, device=dev))
    obs = obs["policy"]
    first = (term | trunc) & ~recorded
    gates = torch.where(first, info["gates_passed"].float(), gates)
    steps = torch.where(first, torch.full_like(steps, t + 1), steps)
    survived |= first & trunc & ~term
    recorded |= term | trunc
    if bool(recorded.all()):
        break

name = Path(args.ckpt).name
lap = (gates >= 8).float().mean() * 100
msg = (
    f"vision eval [{name}, {n} ICs, seed {args.seed}]: "
    f"gates={gates.mean():.1f}  full_lap_rate={lap:.0f}%  "
    f"survival={survived.float().mean() * 100:.0f}%  ep_len={steps.mean():.0f}"
)
Path("/tmp/vision_eval.txt").write_text(msg + "\n", encoding="utf-8")  # noqa: S108
print(msg)
env.close()
sim_app.close()
