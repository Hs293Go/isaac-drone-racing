"""Baseline-vs-GPU comparison: policies raced through the same GPU classical-plant env.

Each policy is evaluated across N random-spawn envs for one episode horizon; we report
mean gates passed, survival rate (reached the episode end without crashing), and mean
episode length — the robustness metric the single-env study used (20 spawns), at GPU
scale. This answers: does the OQCRL effectiveness-model policy TRANSFER to the classical
PhysX plant, does the single-env-fine-tuned classical policy agree across the two envs
(parity), and how does a policy trained NATIVELY on the GPU classical plant compare.

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab \
        python -u -m isaacracelab.eval_compare --num_envs 256 --headless

--obs_noise injects perception-magnitude noise into the obs at eval, so the policies can
be raced under the noise the modular composition will feed them — the paired test of
whether obs-noise-DR training buys robustness a clean-trained racer lacks.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Baseline-vs-GPU policy comparison")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument(
    "--obs_noise",
    type=float,
    default=0.0,
    help="obs-noise DR scale at eval (1.0 = measured magnitude)",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)
from omegaconf import OmegaConf  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
import torch  # noqa: E402

from isaacrace.config import RaceTrackConfig  # noqa: E402
from isaacracelab.racer_env import RacerEnv, RacerEnvCfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_track():
    """Load the figure-8 track the single-env baselines were trained on."""
    p = ROOT / "examples/conf/track/figure8.yaml"
    d = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
    return RaceTrackConfig.from_dict(d)


# --- env (random spawns = the robustness distribution) ---
cfg = RacerEnvCfg()
cfg.track = load_track()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.randomize_reset = True
cfg.obs_noise = args.obs_noise
env = RacerEnv(cfg)
dev = env.device
HORIZON = int(env.max_episode_length) + 2


def sb3_policy(path):
    """A deterministic sb3 PPO policy as a batched obs->action callable."""
    model = PPO.load(str(ROOT / path), device="cpu")

    def act(obs):
        a, _ = model.predict(obs["policy"].cpu().numpy(), deterministic=True)
        return torch.as_tensor(a, device=dev, dtype=torch.float32)

    return act


def rsl_policy(path):
    """The GPU-native rsl_rl policy (act_inference), loaded via a throwaway runner."""
    agent = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=1,
        experiment_name="eval",
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[128, 128],
            critic_hidden_dims=[128, 128],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.999,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )
    runner = OnPolicyRunner(
        RslRlVecEnvWrapper(env), agent.to_dict(), log_dir=None, device=dev
    )
    runner.load(str(ROOT / path), load_optimizer=False)
    return runner.get_inference_policy(device=dev)


def evaluate(act):
    """Run one episode horizon; return (mean gates, survival rate, mean ep length)."""
    obs = env.reset()[0]  # full obs dict {"policy": (N,20)}; each policy extracts it
    n = env.num_envs
    recorded = torch.zeros(n, dtype=torch.bool, device=dev)
    gates = torch.zeros(n, device=dev)
    steps = torch.full((n,), float(HORIZON), device=dev)
    survived = torch.zeros(n, dtype=torch.bool, device=dev)
    for t in range(HORIZON):
        with torch.no_grad():
            obs, _r, term, trunc, info = env.step(act(obs))
        first = (term | trunc) & ~recorded
        gates = torch.where(first, info["gates_passed"].float(), gates)
        steps = torch.where(first, torch.full_like(steps, t + 1), steps)
        survived |= first & trunc & ~term
        recorded |= term | trunc
        if bool(recorded.all()):
            break
    return gates.mean().item(), survived.float().mean().item(), steps.mean().item()


def random_policy(obs):
    """Uniform random actions in [-1, 1]."""
    return torch.rand((obs["policy"].shape[0], 4), device=dev) * 2 - 1


CKPT_SURVIVOR = "examples/train_out/race_classical/model_199.pt"  # 200it
CKPT_RACER = "examples/train_out/race_classical_race/model_799.pt"  # 800it+bonus

CKPT_NOISE_ROBUST = "examples/train_out/race_noise_robust/model_799.pt"  # 800it+DR

POLICIES = [
    ("random", random_policy),
    ("OQCRL baseline (sb3)", sb3_policy("examples/models/race_ppo_baseline.zip")),
    ("single-env FT (sb3)", sb3_policy("train_out/race_ppo_classical_ft.zip")),
    ("GPU survivor (rsl_rl 200it)", rsl_policy(CKPT_SURVIVOR)),
    ("GPU racer (rsl_rl 800it+bonus)", rsl_policy(CKPT_RACER)),
]
# The obs-noise-DR racer (this experiment) — only once it has been trained.
if (ROOT / CKPT_NOISE_ROBUST).exists():
    POLICIES.append((
        "GPU racer + obs-noise DR (rsl_rl)",
        rsl_policy(CKPT_NOISE_ROBUST),
    ))

rows = [
    (
        f"# GPU classical plant, {args.num_envs} random spawns, horizon {HORIZON}, "
        f"obs_noise={args.obs_noise}"
    ),
    f"{'policy':32s} {'gates':>7s} {'survival':>9s} {'ep_len':>8s}",
]
for name, act in POLICIES:
    g, s, ln = evaluate(act)
    row = f"{name:32s} {g:7.2f} {s * 100:8.1f}% {ln:8.0f}"
    rows.append(row)
    print(row, flush=True)

out = "\n".join(rows)
(Path("/tmp/racer_compare.txt")).write_text(out + "\n", encoding="utf-8")  # noqa: S108
print("\n" + out)
env.close()
sim_app.close()
