"""Train a gate-corner detector: FPV image -> the target gate's 4 corner pixels.

The yaw fix. Direct yaw regression hit 10.6 deg (perception_diag); corners localize more
precisely, and a square's 4 image corners give the pose (yaw) via PnP (next step). This
trains CNN(image) -> 8 corner coords on the projected labels (corner_verify validated)
and reports the corner pixel error -- the crux: can the net find corners at 64x64?

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab python -u \
        experimental/corner_train.py --iters 60 --headless --enable_cameras
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Gate-corner detector trainer")
parser.add_argument("--teacher", default="train_out/race_ppo_classical_ft.zip")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--rollout", type=int, default=32)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--batch", type=int, default=512)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

from isaaclab.utils.math import quat_rotate_inverse  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402, N812
from vision_env import CAM_RES, VisionRacerEnv, VisionRacerEnvCfg  # noqa: E402

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
env = VisionRacerEnv(cfg)
dev = env.device
teacher = PPO.load(str(ROOT / args.teacher), device="cpu")
RES = CAM_RES

course = RaceCourse(track)
GATE_POS = torch.as_tensor(course.gate_pos_enu, dtype=torch.float32, device=dev)
GATE_YAW = torch.as_tensor(course.gate_yaw_enu, dtype=torch.float32, device=dev)
H = course.gate_size / 2.0
CORNERS_LOCAL = [(-H, -H), (-H, H), (H, -H), (H, H)]  # bl, tl, br, tr (lateral, vert)


def project():
    """Target-gate corners -> (N,4) u, v, z pixels (forward is -z; in front z<0)."""
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
    rel = (world - cam.pos_w.unsqueeze(1)).reshape(-1, 3)
    quat = cam.quat_w_ros.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
    pcam = quat_rotate_inverse(quat, rel).reshape(-1, 4, 3)
    uvw = torch.einsum("nij,nkj->nki", cam.intrinsic_matrices, pcam)
    z = uvw[..., 2]
    return uvw[..., 0] / z, uvw[..., 1] / z, z


class CornerNet(nn.Module):
    """FPV image -> 8 normalized corner coords (4 corners x u,v / RES)."""

    def __init__(self):
        """CNN over the 64x64 image -> 8 outputs."""
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
        self.head = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 8))

    def forward(self, img):
        """Img (N,H,W,3) in [0,1] -> (N,8) normalized corners."""
        return self.head(self.cnn(img.permute(0, 3, 1, 2)))


net = CornerNet().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
env.reset()
log = []
for it in range(args.iters):
    img_buf, crn_buf = [], []
    with torch.no_grad():
        for _ in range(args.rollout):
            img = env._camera.data.output["rgb"][..., :3].to(torch.float32) / 255.0
            u, v, z = project()
            # keep frames where the gate is clearly ahead and near/in frame
            valid = (
                (z < -0.2) & (u > -RES) & (u < 2 * RES) & (v > -RES) & (v < 2 * RES)
            ).all(dim=-1)
            crn = torch.stack([u, v], dim=-1).reshape(-1, 8) / RES  # normalized
            img_buf.append(img[valid])
            crn_buf.append(crn[valid])
            act, _ = teacher.predict(env._state_obs().cpu().numpy(), deterministic=True)
            env.step(torch.as_tensor(act, device=dev))
    imgs = torch.cat(img_buf, dim=0)
    crns = torch.cat(crn_buf, dim=0)
    n = imgs.shape[0]
    if n < args.batch:
        continue
    idx = torch.randperm(n, device=dev)
    for _ in range(args.epochs):
        for i in range(0, n, args.batch):
            b = idx[i : i + args.batch]
            loss = F.mse_loss(net(imgs[b]), crns[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        px = (net(imgs) - crns).abs().mean().item() * RES  # mean corner pixel error
    log.append(f"it {it:3d}  n={n:5d}  loss={float(loss):.5f}  corner_px_MAE={px:.2f}")
    print(log[-1], flush=True)

torch.save(net.state_dict(), str(ROOT / "examples/train_out/corner_net.pt"))
Path("/tmp/corner_train.txt").write_text("\n".join(log) + "\n", encoding="utf-8")  # noqa: S108
env.close()
sim_app.close()
