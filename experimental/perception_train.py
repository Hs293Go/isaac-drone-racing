"""Modular perception->control: a CNN predicts the gate-relative STATE from pixels.

Decompose pixels->motors. PERCEPTION (image+proprio -> 20-dim gate-relative state) is
supervised regression with free sim labels; CONTROL (state -> action) is the existing
racer, reused unchanged. This trains only perception: roll out a racer on the TRUE state
to fly the course, record (vision obs, true state), and regress state from the image by
MSE. The racer's accuracy is irrelevant (labels = sim truth) — it just drives the data.

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab python -u \
        experimental/perception_train.py --num_envs 128 --headless --enable_cameras
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Supervised perception trainer")
parser.add_argument("--teacher", default="train_out/race_ppo_classical_ft.zip")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--rollout", type=int, default=32)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--batch", type=int, default=512)
parser.add_argument(
    "--supervised",
    action="store_true",
    help="teacher-only data (beta=1, no DAgger anneal)",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402, N812
from vision_env import (  # noqa: E402
    CAM_RES,
    PROPRIO_DIM,
    VisionRacerEnv,
    VisionRacerEnvCfg,
)

from isaacrace.config import RaceTrackConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG_DIM = CAM_RES * CAM_RES * 3


class PerceptionNet(nn.Module):
    """FPV image + proprio -> predicted 20-dim gate-relative state (the racer's obs)."""

    def __init__(self, out_dim: int = 20):
        """CNN over the image, concatenated with proprio, -> out_dim state values."""
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )  # 64x64 -> 1024
        self.head = nn.Sequential(
            nn.Linear(1024 + PROPRIO_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, vis):
        """Flat [image | proprio] obs -> predicted state."""
        img = vis[:, :IMG_DIM].reshape(-1, CAM_RES, CAM_RES, 3).permute(0, 3, 1, 2)
        return self.head(torch.cat([self.cnn(img), vis[:, IMG_DIM:]], dim=-1))


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
teacher = PPO.load(str(ROOT / args.teacher), device="cpu")

pnet = PerceptionNet().to(dev)
opt = torch.optim.Adam(pnet.parameters(), lr=1e-3)
obs = env.reset()[0]["policy"]
log = []
for it in range(args.iters):
    # DAgger: anneal driver teacher->composed so the net sees its OWN distribution;
    # labels are always the true state.
    beta = 1.0 if args.supervised else max(0.0, 1.0 - it / (args.iters * 0.4))
    vis_buf, state_buf = [], []
    with torch.no_grad():
        for _ in range(args.rollout):
            state = env._state_obs()  # (N,20) TRUE gate-relative state (the labels)
            teach_act, _ = teacher.predict(state.cpu().numpy(), deterministic=True)
            comp_act, _ = teacher.predict(pnet(obs).cpu().numpy(), deterministic=True)
            use_teacher = np.random.random((env.num_envs, 1)) < beta
            act = np.where(use_teacher, teach_act, comp_act)
            vis_buf.append(obs)
            state_buf.append(state)
            obs = env.step(torch.as_tensor(act, device=dev))[0]["policy"]
    vis = torch.cat(vis_buf, dim=0)
    states = torch.cat(state_buf, dim=0)
    n = vis.shape[0]
    for _ in range(args.epochs):
        for i in range(0, n, args.batch):
            b = slice(i, i + args.batch)
            loss = F.mse_loss(pnet(vis[b]), states[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        mae = (pnet(vis) - states).abs().mean(dim=0)  # per-component MAE
    gate_pos = float(mae[0:3].mean())  # gate Δposition (m) — the image-critical signal
    yaw = float(mae[8])  # gate-relative yaw (rad)
    log.append(
        f"it {it:3d}  loss={float(loss):7.3f}  gate_pos_MAE={gate_pos:.3f}m  "
        f"yaw_MAE={yaw:.3f}rad"
    )
    print(log[-1], flush=True)

torch.save(pnet.state_dict(), str(ROOT / "examples/train_out/perception_net.pt"))
Path("/tmp/perception_train.txt").write_text("\n".join(log) + "\n", encoding="utf-8")  # noqa: S108
env.close()
sim_app.close()
