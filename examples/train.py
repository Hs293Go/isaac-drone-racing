"""Train a PPO policy on the race env — batched (fast, default) or Isaac (PhysX).

    ... uv run python examples/train.py                                  # batched, fast
    ... uv run python examples/train.py session.train.total_timesteps=1000000
    ... uv run python examples/train.py session.train.perception_weight=0.03
    ... uv run python examples/train.py session.train.domain_randomization=true
    ... uv run python examples/train.py session.train.backend=isaac      # on PhysX

The default ``batched`` backend is BatchedRaceEnv: ``num_envs`` drones integrated at
once in pure NumPy (``dyn.model_derivatives``). Without SimulationApp introducing
stepping overhead, training completes in minutes. Its dynamics, observation, reward and
perception term match RaceEnv by construction (shared dynamics functions + RaceCourse
rules, same explicit-Euler step), so the policy transfers: validate / visualise it on
Isaac with ``examples/demo.py session.demo.model=train_out/race_ppo_final``. Set
``session.train.backend=isaac`` to train directly on PhysX (slow, on-rig fidelity).

Recipe (OQCRL): separate pi/vf MLPs, ReLU, gamma 0.999, VecMonitor, tqdm progress.
Watch ``race/gates_per_1k_steps``, ``race/perception_visibility`` (perception toggle),
``rollout/ep_rew_mean``. Outputs land in repo-root ``train_out/``.
"""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Entry point: train a PPO policy (batched backend by default, or Isaac)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    import torch

    from isaacrace.config import RaceTrackConfig, SessionConfig
    from isaacrace.course import RaceCourse

    # The policy is a tiny MLP; on CPU, one thread far outruns many (PyTorch's
    # multi-thread overhead dominates the small per-minibatch matmuls). This is the
    # difference between ~800 and ~12k training fps with the batched VecEnv.
    torch.set_num_threads(1)

    track_cfg = RaceTrackConfig.from_dict(
        OmegaConf.to_container(cfg.track, resolve=True)
    )
    session = SessionConfig.from_dict(OmegaConf.to_container(cfg.session, resolve=True))
    tcfg = session.train
    course = RaceCourse(track_cfg)
    fpv = getattr(session, "fpv", None)

    class GateRateCallback(BaseCallback):
        """Log the gate-pass rate (gates/1k steps) and mean perception visibility."""

        def __init__(self):
            super().__init__()
            self.gates = 0
            self.steps = 0

        def _on_step(self):
            vis, m = 0.0, 0
            for info in self.locals.get("infos", []):
                self.gates += int(info.get("gate_passed", False))
                self.steps += 1
                vis += float(info.get("perception_visibility", 0.0))
                m += 1
            if self.steps:
                self.logger.record(
                    "race/gates_per_1k_steps", 1000.0 * self.gates / self.steps
                )
            if m:
                self.logger.record("race/perception_visibility", vis / m)
            return True

    # repo-root train_out/ (this script sits at <root>/examples/, so go up 2 levels)
    out = str(Path(__file__).resolve().parents[1] / "train_out")
    Path(out).mkdir(parents=True, exist_ok=True)

    # --- sim backend ----------------------------------------------------------
    app = None
    if tcfg.backend == "batched":
        from isaacrace.env_batched import BatchedRaceEnv

        venv = VecMonitor(
            BatchedRaceEnv(
                course,
                num_envs=tcfg.num_envs,
                seed=session.seed,
                max_steps=session.max_steps,
                fpv=fpv,
                randomize_params=tcfg.domain_randomization,
                param_dr_pct=tcfg.param_dr_pct,
                perception_weight=tcfg.perception_weight,
            )
        )
        # Per-env rollout length sized so the buffer (num_envs * n_steps) stays a few
        # x10k — few large PPO updates, not many tiny ones (the CPU update dominates
        # once env-stepping is vectorized away).
        n_steps = max(8, 32768 // tcfg.num_envs)
    else:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": session.headless})
        from isaacrace.env import RaceEnv

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
        n_steps = tcfg.n_steps

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=tcfg.learning_rate,
        gamma=tcfg.gamma,
        gae_lambda=tcfg.gae_lambda,
        clip_range=tcfg.clip_range,
        ent_coef=tcfg.ent_coef,
        n_steps=n_steps,
        batch_size=tcfg.batch_size,
        n_epochs=tcfg.n_epochs,
        policy_kwargs={
            "activation_fn": torch.nn.ReLU,
            "net_arch": {"pi": list(tcfg.pi), "vf": list(tcfg.vf)},
            "log_std_init": tcfg.log_std_init,
        },
        verbose=1,
        device="cpu",
        seed=session.seed,  # controls PPO init + sampling -> reproducible seed sweeps
        tensorboard_log=out,
    )
    cbs = [
        GateRateCallback(),
        CheckpointCallback(
            save_freq=max(1, tcfg.save_freq // venv.num_envs),
            save_path=out,
            name_prefix=tcfg.run_name,
        ),
    ]
    model.learn(total_timesteps=tcfg.total_timesteps, callback=cbs, progress_bar=True)
    model.save(str(Path(out) / f"{tcfg.run_name}_final"))
    print(
        f"[train] done ({tcfg.backend}); saved {out}/{tcfg.run_name}_final "
        f"({tcfg.total_timesteps} steps)"
    )

    if app is not None:
        app.close()


if __name__ == "__main__":
    main()
