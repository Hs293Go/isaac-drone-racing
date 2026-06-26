"""Train the classical-plant racer on Isaac Lab with rsl_rl (scaffold).

Mirrors the standard Isaac Lab rsl_rl runner; PPO hyperparameters track examples/conf
(lr 3e-4, gamma 0.999, gae 0.95, clip 0.2).

Run headless + unbuffered (`python -u`) — Isaac's hard exit discards buffered stdout,
so without `-u` the per-iteration training logs are lost (it looks like a hang):

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab \
        python -u examples/isaaclab/train.py --num_envs 4096 --headless
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the classical-plant racer (rsl_rl)")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=500)
parser.add_argument("--track", default=None, help="track yaml (default: figure8)")
parser.add_argument("--gate_bonus", type=float, default=10.0, help="reward/gate")
parser.add_argument("--rate_penalty", type=float, default=5e-4)
parser.add_argument(
    "--obs_noise",
    type=float,
    default=0.0,
    help="obs-noise DR scale (1.0 = measured perception magnitude)",
)
parser.add_argument("--run_name", default="race_classical")
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
from omegaconf import OmegaConf  # noqa: E402  (heavy imports follow the app boot)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaacrace.config import RaceTrackConfig  # noqa: E402
from isaacracelab.racer_env import RacerEnv, RacerEnvCfg  # noqa: E402


def _load_track(path: str | None) -> RaceTrackConfig:
    """Load the figure-8 track (or a given yaml) into a RaceTrackConfig."""
    default = Path(__file__).resolve().parents[2] / "examples/conf/track/figure8.yaml"
    p = Path(path) if path else default
    d = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
    return RaceTrackConfig.from_dict(d)


# --- env ---
env_cfg = RacerEnvCfg()
env_cfg.track = _load_track(args.track)
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = args.device
env_cfg.gate_bonus = args.gate_bonus
env_cfg.rate_penalty = args.rate_penalty
env_cfg.obs_noise = args.obs_noise
env = RacerEnv(env_cfg)
env = RslRlVecEnvWrapper(env)  # gym (num_envs, ...) -> rsl_rl VecEnv

# --- agent (PPO; fields VERIFY against the installed rsl_rl 2.3) ---
agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24,
    max_iterations=args.max_iterations,
    save_interval=50,
    experiment_name=args.run_name,
    empirical_normalization=False,
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

log_dir = str(
    Path(__file__).resolve().parents[2] / "examples/train_out" / args.run_name
)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=args.device)
runner.learn(
    num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
)

env.close()
sim_app.close()
