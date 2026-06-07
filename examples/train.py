"""Train a fresh PPO policy from scratch on the Isaac Sim (PhysX) race env.

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run python -m isaacrace.train
    ... uv run python -m isaacrace.train \
        session.train.total_timesteps=4000           # smoke test
    ... uv run python -m isaacrace.train \
        session.train.pi=[256,256] session.train.domain_randomization=true

Hyperparameters + course come from conf/ (see conf/session/default.yaml; override
on the CLI as above). Outputs land under ``isaacrace/train_out/``: tensorboard
logs, ``race_ppo_*`` checkpoints, ``race_ppo_final``. Watch
``race/gates_per_1k_steps`` (climbs from ~1 toward ~20+ as the policy learns to
thread the gates), and ``rollout/ep_rew_mean`` (via VecMonitor).

Recipe ported from OQCRL's train.py: separate pi/vf MLPs, ReLU, gamma 0.999,
VecMonitor, tqdm progress. ``session.train.domain_randomization=true`` resamples
the drone's OQCRL params +-``param_dr_pct`` each episode (QuadParams.randomized) so
the policy is robust to drone variation; the observation is unchanged, so it stays
compatible with the packaged model. ``RaceEnv`` is a standard gymnasium env, so
DummyVecEnv wraps it directly and handles reset-on-done.
"""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Entry point: train a PPO policy from scratch on the race env."""
    from isaacrace.config import RaceTrackConfig, SessionConfig

    track_cfg = RaceTrackConfig.from_dict(
        OmegaConf.to_container(cfg.track, resolve=True)
    )
    session = SessionConfig.from_dict(OmegaConf.to_container(cfg.session, resolve=True))
    tcfg = session.train

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": session.headless})

    import carb
    from isaacrace.course import RaceCourse
    from isaacrace.env import RaceEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    import torch

    class GateRateCallback(BaseCallback):
        """Log the running gate-pass rate (gates per 1k env steps)."""

        def __init__(self):
            super().__init__()
            self.gates = 0
            self.steps = 0

        def _on_step(self):
            for info in self.locals.get("infos", []):
                self.gates += int(info.get("gate_passed", False))
                self.steps += 1
            if self.steps:
                self.logger.record(
                    "race/gates_per_1k_steps", 1000.0 * self.gates / self.steps
                )
            return True

    # repo-root train_out/ (this script sits at <root>/examples/, so go up 2 levels)
    out = str(Path(__file__).resolve().parents[1] / "train_out")
    Path(out).mkdir(parents=True, exist_ok=True)
    course = RaceCourse(track_cfg)
    # VecMonitor logs episode reward/length to tensorboard (rollout/ep_rew_mean, ...).
    venv = VecMonitor(
        DummyVecEnv([
            lambda: RaceEnv(
                course,
                headless=session.headless,
                randomize_reset=True,
                seed=session.seed,
                max_steps=session.max_steps,
                randomize_params=tcfg.domain_randomization,
                param_dr_pct=tcfg.param_dr_pct,
            )
        ])
    )

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=tcfg.learning_rate,
        gamma=tcfg.gamma,
        gae_lambda=tcfg.gae_lambda,
        clip_range=tcfg.clip_range,
        ent_coef=tcfg.ent_coef,
        n_steps=tcfg.n_steps,
        batch_size=tcfg.batch_size,
        n_epochs=tcfg.n_epochs,
        policy_kwargs={
            "activation_fn": torch.nn.ReLU,
            "net_arch": {"pi": list(tcfg.pi), "vf": list(tcfg.vf)},
            "log_std_init": tcfg.log_std_init,
        },
        verbose=1,
        device="cpu",
        tensorboard_log=out,
    )
    cbs = [
        GateRateCallback(),
        CheckpointCallback(
            save_freq=tcfg.save_freq, save_path=out, name_prefix="race_ppo"
        ),
    ]
    model.learn(total_timesteps=tcfg.total_timesteps, callback=cbs, progress_bar=True)
    model.save(str(Path(out) / "race_ppo_final"))
    carb.log_warn(
        f"[train] done; saved {out}/race_ppo_final ({tcfg.total_timesteps} steps)"
    )

    app.close()


if __name__ == "__main__":
    main()
