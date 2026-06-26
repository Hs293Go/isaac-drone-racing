"""Per-group perception ablation: which obs group's error craters the controller?

The composition (net -> controller) crashed (0.1 gates) though the controller tolerates
measured gate-pose NOISE (eval_compare --obs_noise 1.0 -> ~50 gates) -- so the damage is
elsewhere. It feeds the controller TRUE state with one group swapped for the net's REAL
prediction, reporting gates per group to isolate the killer (real errors). Also logs the
net's full per-component error.

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab python -u \
        experimental/perception_diag.py --headless --enable_cameras
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Per-group perception ablation")
parser.add_argument("--control", default="train_out/race_ppo_classical_ft.zip")
parser.add_argument("--pnet", default="examples/train_out/perception_net.pt")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
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

# Finer than build_obs's blocks: separate image-derived dims (pos, vel, yaw, lookahead)
# from the ego/proprio dims (roll-pitch, rates, motor) the net predicts cheaply.
GROUPS = {
    "oracle (all true)": None,
    "gate_pos[0:3]": slice(0, 3),
    "vel[3:6]": slice(3, 6),
    "rollpitch[6:8]": slice(6, 8),
    "yaw[8]": slice(8, 9),
    "rates[9:12]": slice(9, 12),
    "motor[12:16]": slice(12, 16),
    "lookahead[16:20]": slice(16, 20),
    "all pred (composition)": "all",
}
ERR_GROUPS = {
    "gate_pos[0:3]": (0, 3),
    "vel[3:6]": (3, 6),
    "rollpitch[6:8]": (6, 8),
    "yaw[8]": (8, 9),
    "rates[9:12]": (9, 12),
    "motor[12:16]": (12, 16),
    "lookahead[16:20]": (16, 20),
}


def run(group, measure_err=False):
    """One horizon with the controller fed true state + group `group` from the net."""
    torch.manual_seed(args.seed)  # same ICs across configs
    obs = env.reset()[0]["policy"]
    n = env.num_envs
    recorded = torch.zeros(n, dtype=torch.bool, device=dev)
    gates = torch.zeros(n, device=dev)
    steps = torch.full((n,), float(HORIZON), device=dev)
    survived = torch.zeros(n, dtype=torch.bool, device=dev)
    err_sum = torch.zeros(20, device=dev)
    err_cnt = 0
    for t in range(HORIZON):
        with torch.no_grad():
            true = env._state_obs()
            pred = pnet(obs)
        if measure_err:
            err_sum += (pred - true).abs().mean(dim=0)
            err_cnt += 1
        if group is None:
            ctrl_in = true
        elif group == "all":
            ctrl_in = pred
        else:
            ctrl_in = true.clone()
            ctrl_in[:, group] = pred[:, group]
        act, _ = control.predict(ctrl_in.cpu().numpy(), deterministic=True)
        obs, _r, term, trunc, info = env.step(torch.as_tensor(act, device=dev))
        obs = obs["policy"]
        first = (term | trunc) & ~recorded
        gates = torch.where(first, info["gates_passed"].float(), gates)
        steps = torch.where(first, torch.full_like(steps, t + 1), steps)
        survived |= first & trunc & ~term
        recorded |= term | trunc
        if bool(recorded.all()):
            break
    mae = err_sum / max(err_cnt, 1) if measure_err else None
    surv = survived.float().mean().item() * 100
    return gates.mean().item(), surv, steps.mean().item(), mae


rows = [
    f"# per-group ablation ({args.num_envs} ICs): TRUE state, one group from the net",
    f"{'config':26s} {'gates':>7s} {'surv':>6s} {'ep_len':>7s}",
]
mae_all = None
for name, g in GROUPS.items():
    gt, sv, ln, mae = run(g, measure_err=(g is None))  # measure error in oracle rollout
    if mae is not None:
        mae_all = mae
    rows.append(f"{name:26s} {gt:7.1f} {sv:5.0f}% {ln:7.0f}")
    print(rows[-1], flush=True)

rows.append("")
rows.append("# pnet per-component MAE (oracle rollout) -- error each group carries:")
for gname, (a, b) in ERR_GROUPS.items():
    dims = [round(x, 3) for x in mae_all[a:b].tolist()]
    m = mae_all[a:b].mean().item()
    rows.append(f"  {gname:18s} MAE={m:.3f}  per-dim {dims}")
    print(rows[-1], flush=True)

out = "\n".join(rows)
Path("/tmp/perception_diag.txt").write_text(out + "\n", encoding="utf-8")  # noqa: S108
env.close()
sim_app.close()
