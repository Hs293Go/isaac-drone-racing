# Examines repeated application of model_derivatives is equivalent to batched
# model_derivatives.

import numpy as np
import pytest

import isaacrace.dynamics as dyn


def _random_states(rng, n):
    return np.hstack([
        rng.uniform(-2.0, 2.0, (n, 3)),
        rng.uniform(-0.5, 0.5, (n, 3)),
        rng.uniform(-np.pi / 9, np.pi / 9, (n, 2)),
        rng.uniform(-np.pi, np.pi, (n, 1)),
        rng.uniform(-0.1, 0.1, (n, 3)),
        rng.uniform(-1.0, 1.0, (n, 4)),
    ])


@pytest.mark.parametrize("seed", range(4))
def test_batched_matches_scalar_shared_params(seed):
    rng = np.random.default_rng(seed)
    n = 50
    states, us = _random_states(rng, n), rng.uniform(-1.0, 1.0, (n, 4))
    batched = dyn.model_derivatives(states, us, dyn.PARAMS_5INCH)
    scalar = np.stack([
        dyn.model_derivatives(states[i], us[i], dyn.PARAMS_5INCH) for i in range(n)
    ])
    np.testing.assert_allclose(batched, scalar, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_batched_matches_scalar_per_env_params(seed):
    """Per-env domain-randomized params: a QuadParams whose fields are (N,) arrays."""
    rng = np.random.default_rng(seed)
    n = 50
    states, us = _random_states(rng, n), rng.uniform(-1.0, 1.0, (n, 4))
    # per-env DR via the production path (clamps k<=1, keeping the motor sqrt valid)
    per_env = [dyn.PARAMS_5INCH.randomized(rng, 0.1) for _ in range(n)]
    params_arr = np.array([np.asarray(pp) for pp in per_env])  # (N, 23)
    batched = dyn.model_derivatives(states, us, dyn.QuadParams._make(params_arr.T))
    scalar = np.stack([
        dyn.model_derivatives(states[i], us[i], per_env[i]) for i in range(n)
    ])
    np.testing.assert_allclose(batched, scalar, rtol=1e-9, atol=1e-12)
