"""BatchedRaceEnv smoke + determinism + auto-reset (NumPy + sb3; no isaacsim)."""

import numpy as np

from isaacrace.config import RaceTrackConfig
from isaacrace.course import RaceCourse
from isaacrace.env_batched import BatchedRaceEnv


def _course():
    return RaceCourse(
        RaceTrackConfig(
            gate_pos=[[1.5, -2.5, -1.5], [2.5, 0.0, -1.5], [1.5, 2.5, -1.5]],
            gate_yaw=[1, 0, -1],
            start_pos=[1.5, -2.5, -1.5],
            gate_yaw_unit="multiples_pi_2",
            gate_size=1.5,
        )
    )


def test_batched_env_rollout():
    env = BatchedRaceEnv(
        _course(), num_envs=8, seed=0, max_steps=50, perception_weight=0.02
    )
    obs = env.reset()
    assert obs.shape == (8, 20)
    assert obs.dtype == np.float32
    for _ in range(60):
        env.step_async(np.zeros((8, 4), dtype=np.float32))
        obs, rew, dones, infos = env.step_wait()
        assert obs.shape == (8, 20)
        assert rew.shape == (8,)
        assert np.all(np.isfinite(rew))
        assert dones.shape == (8,)
        assert dones.dtype == bool
        assert len(infos) == 8
        for info in infos:
            assert 0.0 <= info["perception_visibility"] <= 1.0
    env.close()


def test_batched_env_determinism():
    a = BatchedRaceEnv(_course(), num_envs=4, seed=11)
    b = BatchedRaceEnv(_course(), num_envs=4, seed=11)
    np.testing.assert_array_equal(a.reset(), b.reset())
    act = np.full((4, 4), 0.3, dtype=np.float32)
    a.step_async(act)
    b.step_async(act)
    np.testing.assert_array_equal(a.step_wait()[0], b.step_wait()[0])


def test_batched_env_auto_reset():
    # max_steps=1 -> every env is done after one step; obs is the (reset) next obs and
    # info carries the terminal observation (the sb3 VecEnv auto-reset contract).
    env = BatchedRaceEnv(_course(), num_envs=3, seed=0, max_steps=1)
    env.reset()
    env.step_async(np.zeros((3, 4), dtype=np.float32))
    obs, _rew, dones, infos = env.step_wait()
    assert np.all(dones)
    assert obs.shape == (3, 20)
    for info in infos:
        assert "terminal_observation" in info
