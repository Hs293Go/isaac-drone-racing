"""Batched RaceCourse methods == per-row scalar (pure NumPy; no isaacsim)."""

import numpy as np
import pytest

from isaacrace.config import RaceTrackConfig
from isaacrace.course import RaceCourse


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


def _states(rng, n):
    return np.hstack([
        rng.uniform(-2, 2, (n, 3)),
        rng.uniform(-0.5, 0.5, (n, 3)),
        rng.uniform(-np.pi / 9, np.pi / 9, (n, 2)),
        rng.uniform(-np.pi, np.pi, (n, 1)),
        rng.uniform(-0.1, 0.1, (n, 3)),
        rng.uniform(-1, 1, (n, 4)),
    ]).astype(np.float32)


@pytest.mark.parametrize("seed", range(4))
def test_build_obs_batched_matches_scalar(seed):
    c = _course()
    rng = np.random.default_rng(seed)
    n = 40
    states = _states(rng, n)
    tg = rng.integers(0, c.num_gates, n)
    batched = c.build_obs(states, tg)
    scalar = np.stack([c.build_obs(states[i], int(tg[i])) for i in range(n)])
    np.testing.assert_allclose(batched, scalar, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_gate_passed_batched_matches_scalar(seed):
    c = _course()
    rng = np.random.default_rng(seed)
    n = 200
    gp = c.gate_pos_enu[rng.integers(0, c.num_gates, n)]
    pos_old = gp + rng.uniform(-1.5, 1.5, (n, 3))  # near gates so some cross
    pos_new = gp + rng.uniform(-1.5, 1.5, (n, 3))
    tg = rng.integers(0, c.num_gates, n)
    passed_b, coll_b = c.gate_passed(pos_old, pos_new, tg)
    res = [c.gate_passed(pos_old[i], pos_new[i], int(tg[i])) for i in range(n)]
    np.testing.assert_array_equal(passed_b, np.array([r[0] for r in res]))
    np.testing.assert_array_equal(coll_b, np.array([r[1] for r in res]))
