"""Smoke-test the Isaac Lab install: boot, make a GPU-parallel stock task, step it.

The pip-only Isaac Lab package ships no runner scripts, so this is our minimal one. It
confirms the pipeline works (Isaac Sim 5.1 + Isaac Lab 2.3) before we write a custom
drone-racing DirectRLEnv.

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab \
        python examples/isaaclab/smoke.py --task Isaac-Quadcopter-Direct-v0 \
        --num_envs 64 --headless

Isaac swallows stdout (carb owns the console), so results go to /tmp/isaaclab_smoke.txt.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab install smoke test")
parser.add_argument("--task", default="Isaac-Quadcopter-Direct-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)  # adds --headless, --device, etc.
args = parser.parse_args()

app_launcher = AppLauncher(args)  # boots Isaac Sim (GPU pipeline by default)
sim_app = app_launcher.app

import gymnasium as gym  # noqa: E402  (heavy imports must follow the app boot)
import isaaclab_tasks  # noqa: E402, F401  (registers the Isaac-* gym ids)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import torch  # noqa: E402

lines = []
try:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    lines += [
        f"task={args.task} num_envs={args.num_envs} device={args.device}",
        f"obs_space={env.observation_space}",
        f"action_space={env.action_space}",
    ]
    obs, _ = env.reset()
    rew = torch.zeros(1)
    for _ in range(args.steps):
        act = torch.from_numpy(env.action_space.sample()).to(args.device)
        obs, rew, term, trunc, info = env.step(act)
    obs_t = obs["policy"] if isinstance(obs, dict) else obs
    lines.append(
        f"after {args.steps} steps: obs={tuple(obs_t.shape)} "
        f"reward_mean={float(rew.mean()):.3f} on {obs_t.device}"
    )
    lines.append("[OK] Isaac Lab GPU-parallel task ran")
    env.close()
except Exception as e:
    import traceback

    lines += ["[FAIL] " + repr(e), traceback.format_exc()]

Path("/tmp/isaaclab_smoke.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")  # noqa: S108
sim_app.close()
