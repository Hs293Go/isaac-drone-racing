"""Eval the composed modular policy: image -> perception -> state -> control -> action.

Load the perception net + the state controller, compose them (predicted state feeds the
controller unchanged), and measure gates / full-lap-rate / survival over fixed-seed ICs.
Compare to the controller's ceiling on the TRUE state (--oracle) and to end-to-end RL.

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab python -u \
        experimental/perception_eval.py --headless --enable_cameras
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Composed perception->control eval")
parser.add_argument("--control", default="train_out/race_ppo_classical_ft.zip")
parser.add_argument("--pnet", default="examples/train_out/perception_net.pt")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--oracle", action="store_true", help="feed TRUE state (ceiling)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

from omegaconf import OmegaConf  # noqa: E402
from perception_net import PerceptionNet  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
import torch  # noqa: E402
from vision_env import VisionRacerEnv, VisionRacerEnvCfg  # noqa: E402

from isaacrace.config import RaceTrackConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
torch.manual_seed(args.seed)
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
control = PPO.load(str(ROOT / args.control), device="cpu")
pnet = PerceptionNet().to(dev)
pnet.load_state_dict(torch.load(str(ROOT / args.pnet), map_location=dev))
pnet.eval()
HORIZON = int(env.max_episode_length) + 2

obs = env.reset()[0]["policy"]
n = env.num_envs
recorded = torch.zeros(n, dtype=torch.bool, device=dev)
gates = torch.zeros(n, device=dev)
steps = torch.full((n,), float(HORIZON), device=dev)
survived = torch.zeros(n, dtype=torch.bool, device=dev)
for t in range(HORIZON):
    with torch.no_grad():
        state = env._state_obs() if args.oracle else pnet(obs)
    act, _ = control.predict(state.cpu().numpy(), deterministic=True)
    obs, _r, term, trunc, info = env.step(torch.as_tensor(act, device=dev))
    obs = obs["policy"]
    first = (term | trunc) & ~recorded
    gates = torch.where(first, info["gates_passed"].float(), gates)
    steps = torch.where(first, torch.full_like(steps, t + 1), steps)
    survived |= first & trunc & ~term
    recorded |= term | trunc
    if bool(recorded.all()):
        break

lap = (gates >= 8).float().mean() * 100
surv = survived.float().mean() * 100
mode = "ORACLE true-state" if args.oracle else "perception->control"
msg = (
    f"[{mode}, {n} ICs]: gates={gates.mean():.1f}  lap_rate={lap:.0f}%  "
    f"survival={surv:.0f}%  ep_len={steps.mean():.0f}"
)
Path("/tmp/perception_eval.txt").write_text(msg + "\n", encoding="utf-8")  # noqa: S108
print("\n" + msg)
env.close()
sim_app.close()
