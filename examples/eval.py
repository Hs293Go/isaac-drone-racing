"""Evaluate a trained policy in the batched env (no Isaac Sim) and print headline stats.

    uv run python examples/eval.py                                    # packaged model
    uv run python examples/eval.py session.demo.model=train_out/race_ppo_final.zip
    uv run python examples/eval.py session.train.num_envs=64          # smaller batch

Rolls ``session.train.num_envs`` drones for ``session.demo.steps`` steps with
deterministic actions and randomized spawns, then reports gates per 1k env-steps (the
same metric train.py logs as ``race/gates_per_1k_steps``), crash rate, mean gate
visibility, and env throughput. For the on-Isaac equivalent, run examples/demo.py with
the same model.
"""

import os
from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Entry point: roll out a policy in BatchedRaceEnv and print summary stats."""
    from stable_baselines3 import PPO
    import torch

    from isaacrace.config import RaceTrackConfig, SessionConfig
    from isaacrace.course import RaceCourse
    from isaacrace.env_batched import BatchedRaceEnv

    torch.set_num_threads(1)  # same rationale as train.py: tiny MLP, one thread wins

    track_cfg = RaceTrackConfig.from_dict(
        OmegaConf.to_container(cfg.track, resolve=True)
    )
    session = SessionConfig.from_dict(OmegaConf.to_container(cfg.session, resolve=True))

    model_path = session.demo.model or str(
        Path(__file__).parent / "models" / "race_ppo_perception.zip"
    )
    model = PPO.load(model_path, device="cpu")
    env = BatchedRaceEnv(
        RaceCourse(track_cfg),
        num_envs=session.train.num_envs,
        seed=session.seed,
        max_steps=session.max_steps,
    )

    obs = env.reset()
    gates = steps = crashes = 0
    vis_sum = 0.0
    t0 = time.perf_counter()
    for _ in range(session.demo.steps):
        action, _ = model.predict(obs, deterministic=True)
        env.step_async(action)
        obs, _rew, dones, infos = env.step_wait()
        for i, info in enumerate(infos):
            gates += info["gate_passed"]
            vis_sum += info["perception_visibility"]
            steps += 1
            if dones[i] and not info.get("TimeLimit.truncated", False):
                crashes += 1
    dt = time.perf_counter() - t0

    # raw env throughput: the batched dynamics step alone, no policy forward pass
    env.reset()
    zero = np.zeros((env.num_envs, 4), dtype=np.float32)
    t1 = time.perf_counter()
    for _ in range(session.demo.steps):
        env.step_async(zero)
        env.step_wait()
    raw_sps = env.num_envs * session.demo.steps / (time.perf_counter() - t1)

    ep_equiv = steps / session.max_steps
    print(f"[eval] model={model_path}")
    print(f"[eval] {env.num_envs} envs x {session.demo.steps} steps in {dt:.1f}s")
    print(f"[eval] RESULT gates_per_1k_steps={1000.0 * gates / steps:.1f}")
    print(f"[eval] crashes_per_episode={crashes / ep_equiv:.3f}")
    print(f"[eval] mean_visibility={vis_sum / steps:.2f}")
    print(f"[eval] env_steps_per_sec={steps / dt:,.0f} (with policy)")
    print(f"[eval] env_steps_per_sec_raw={raw_sps:,.0f} (no policy)")


if __name__ == "__main__":
    main()
