"""Train the vision (image+proprio) racing policy: sb3 PPO + a CNN+MLP extractor.

The extractor splits the flat obs into a 64x64 RGB image (-> a small CNN) and a 13-dim
proprioception vector (-> passthrough), concatenated before the policy/value heads. The
classical plant + racing reward come from VisionRacerEnv. Needs --enable_cameras. sb3 is
used here (familiar CNN-extractor API; rendering caps throughput anyway); skrl is the
GPU-resident option for large runs.

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab python -u \
        experimental/vision_train.py --num_envs 256 --headless --enable_cameras
"""

import argparse
from pathlib import Path
import statistics

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Vision (image+proprio) racing PPO")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--total_steps", type=int, default=2_000_000)
parser.add_argument("--n_steps", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=3)  # fewer: moving CNN overfits
parser.add_argument("--ent_coef", type=float, default=0.0)
parser.add_argument("--lr_init", type=float, default=3e-4)
parser.add_argument("--lr_final", type=float, default=1e-5)  # LR: init -> final
parser.add_argument("--target_kl", type=float, default=0.0)  # 0 = off
parser.add_argument("--run_name", default="race_vision")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from vision_env import (  # noqa: E402
    CAM_RES,
    PROPRIO_DIM,
    VisionRacerEnv,
    VisionRacerEnvCfg,
)

from isaacrace.config import RaceTrackConfig  # noqa: E402

IMG_DIM = CAM_RES * CAM_RES * 3
ROOT = Path(__file__).resolve().parents[1]


class VisionProprioExtractor(BaseFeaturesExtractor):
    """Split flat obs -> CNN(image) + proprio passthrough -> concatenated features."""

    def __init__(self, observation_space, img_features: int = 128):
        """CNN(image) -> img_features, concatenated with the raw proprio vector."""
        super().__init__(observation_space, features_dim=img_features + PROPRIO_DIM)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )  # 64x64 -> 15x15x32 -> 6x6x64 -> 4x4x64 = 1024
        self.head = nn.Sequential(nn.Linear(1024, img_features), nn.ReLU())

    def forward(self, obs):
        """Flat obs -> [CNN(image) | proprio] feature vector."""
        img = obs[:, :IMG_DIM].reshape(-1, CAM_RES, CAM_RES, 3).permute(0, 3, 1, 2)
        return torch.cat([self.head(self.cnn(img)), obs[:, IMG_DIM:]], dim=-1)


class SaveBestEpLen(BaseCallback):
    """Save the policy whenever mean episode length improves (vision PPO oscillates)."""

    def __init__(self, path):
        """Track best mean episode length; save to ``path`` on improvement."""
        super().__init__()
        self.path = path
        self.best = -1.0

    def _on_rollout_end(self):
        ep = self.model.ep_info_buffer
        if ep:
            m = sum(e["l"] for e in ep) / len(ep)
            if m > self.best:
                self.best = m
                self.model.save(self.path)

    def _on_step(self):
        return True


track_yaml = ROOT / "examples/conf/track/figure8.yaml"
track = RaceTrackConfig.from_dict(
    OmegaConf.to_container(OmegaConf.load(track_yaml), resolve=True)
)

cfg = VisionRacerEnvCfg()
cfg.track = track
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
env = Sb3VecEnvWrapper(VisionRacerEnv(cfg))

policy_kwargs = {
    "features_extractor_class": VisionProprioExtractor,
    "net_arch": [128, 128],
}


def _lr_schedule(progress_remaining):
    """Linear LR decay: lr_init at start -> lr_final at end (anneals aggression)."""
    return args.lr_final + (args.lr_init - args.lr_final) * progress_remaining


model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    n_steps=args.n_steps,
    batch_size=args.n_steps * args.num_envs // 4,
    n_epochs=args.n_epochs,
    learning_rate=_lr_schedule,
    gamma=0.999,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=args.ent_coef,
    target_kl=(args.target_kl or None),
    device=args.device,
    verbose=0,
)
base = str(ROOT / "examples/train_out" / args.run_name)
model.set_logger(configure(base + "_log", ["csv"]))
best = SaveBestEpLen(base + "_best")
model.learn(total_timesteps=args.total_steps, progress_bar=False, callback=best)
model.save(base + "_final")

ep = list(model.ep_info_buffer)
if ep:
    r = statistics.mean(e["r"] for e in ep)
    ln = statistics.mean(e["l"] for e in ep)
    msg = (
        f"vision PPO: {args.total_steps} steps, {args.num_envs} envs "
        f"-> ep_rew_mean={r:.1f}  ep_len_mean={ln:.0f}"
    )
else:
    msg = "vision PPO done (no episode info buffered)"
Path("/tmp/vision_train.txt").write_text(msg + "\n", encoding="utf-8")  # noqa: S108
print(msg)
env.close()
sim_app.close()
