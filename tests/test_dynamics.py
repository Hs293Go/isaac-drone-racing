"""Equivalence: isaacrace.dynamics.f_func vs OQCRL's symbolic (sympy) model.

Ported from optimal_quad_control_RL/tests/test_quad_model_porting.py. OQCRL derives
the 5-inch-quad equations of motion symbolically in sympy and lambdifies them to
numpy; here we rebuild that exact symbolic reference and assert isaacrace's
hand-written NumPy ``f_func`` matches it to floating-point tolerance, over random
states / controls / params.

Pure NumPy: imports only ``isaacrace.dynamics`` (no isaacsim), so it runs under
plain ``pytest`` with no Isaac Sim boot.
"""

import functools

import isaacrace.dynamics as dyn
import numpy as np
import pytest
from sympy import Array, Matrix, cos, lambdify, sin, sqrt, symbols, tan


@functools.lru_cache(maxsize=1)
def _symbolic_f_func():
    """OQCRL's symbolic state derivative, lambdified to numpy (single 16-vec sample).

    A verbatim port of the sympy derivation in OQCRL's test fixture, so it is an
    independent ground truth for isaacrace's f_func. Cached: the lambdify is slow.
    """
    x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, w1, w2, w3, w4 = symbols(
        "x y z v_x v_y v_z phi theta psi p q r w1 w2 w3 w4"
    )
    u1, u2, u3, u4 = symbols("U_1 U_2 U_3 U_4")
    state = (x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, w1, w2, w3, w4)
    control = (u1, u2, u3, u4)
    params = symbols(
        "k_x k_y k_w k_p1 k_p2 k_p3 k_p4 k_q1 k_q2 k_q3 k_q4 "
        "k_r1 k_r2 k_r3 k_r4 k_r5 k_r6 k_r7 k_r8 tau k w_min w_max"
    )
    # fmt: off
    (k_x, k_y, k_w, k_p1, k_p2, k_p3, k_p4, k_q1, k_q2, k_q3, k_q4,
     k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8, tau, k, w_min, w_max) = params
    # fmt: on

    g = 9.81
    Rx = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
    Ry = Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
    Rz = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
    R = Rz * Ry * Rx
    vbx, vby, _ = R.T @ Matrix([vx, vy, vz])

    w_min_n, w_max_n = 0.0, 3000.0
    W = [(wi + 1) / 2 * (w_max_n - w_min_n) + w_min_n for wi in (w1, w2, w3, w4)]
    U = [(ui + 1) / 2 for ui in (u1, u2, u3, u4)]
    Wc = [(w_max - w_min) * sqrt(k * Ui**2 + (1 - k) * Ui) + w_min for Ui in U]
    d_W = [(Wci - Wi) / tau for Wci, Wi in zip(Wc, W, strict=True)]
    d_w = [dWi / (w_max_n - w_min_n) * 2 for dWi in d_W]

    thrust = -k_w * sum(Wi**2 for Wi in W)
    drag_x = -k_x * vbx * sum(W)
    drag_y = -k_y * vby * sum(W)
    mx = -k_p1 * W[0] ** 2 - k_p2 * W[1] ** 2 + k_p3 * W[2] ** 2 + k_p4 * W[3] ** 2
    my = -k_q1 * W[0] ** 2 + k_q2 * W[1] ** 2 - k_q3 * W[2] ** 2 + k_q4 * W[3] ** 2
    mz = (
        -k_r1 * W[0]
        + k_r2 * W[1]
        + k_r3 * W[2]
        - k_r4 * W[3]
        - k_r5 * d_W[0]
        + k_r6 * d_W[1]
        + k_r7 * d_W[2]
        - k_r8 * d_W[3]
    )

    d_v = Matrix([0, 0, g]) + R @ Matrix([drag_x, drag_y, thrust])
    d_phi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
    d_theta = q * cos(phi) - r * sin(phi)
    d_psi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)

    f = [vx, vy, vz, d_v[0], d_v[1], d_v[2], d_phi, d_theta, d_psi, mx, my, mz, *d_w]
    return lambdify((Array(state), Array(control), Array(params)), Array(f), "numpy")


def _random_states(rng, n):
    """Random states in OQCRL's reset_() bounds -> (n, 16)."""
    return np.hstack([
        rng.uniform(-2.0, 2.0, (n, 3)),  # position
        rng.uniform(-0.5, 0.5, (n, 3)),  # velocity
        rng.uniform(-np.pi / 9, np.pi / 9, (n, 2)),  # phi, theta
        rng.uniform(-np.pi, np.pi, (n, 1)),  # psi
        rng.uniform(-0.1, 0.1, (n, 3)),  # body rates
        rng.uniform(-1.0, 1.0, (n, 4)),  # normalized motor states
    ])


@pytest.mark.parametrize("seed", range(6))
def test_f_func_matches_sympy(seed):
    """isaacrace.f_func == OQCRL symbolic f, over random states/controls/params."""
    symbolic = _symbolic_f_func()
    rng = np.random.default_rng(seed)
    states = _random_states(rng, 64)
    controls = rng.uniform(-1.0, 1.0, (64, 4))
    # nominal 5-inch params on even seeds; +-10% randomized on odd seeds (the
    # equivalence must hold for any params, not just the nominal set).
    p = dyn.params_vec()
    if seed % 2:
        p = p.scaled(rng.uniform(0.9, 1.1, len(p)))
    p_arr = np.asarray(p)

    for s, u in zip(states, controls, strict=True):
        expected = np.asarray(symbolic(s, u, p_arr), dtype=float).reshape(16)
        np.testing.assert_allclose(dyn.f_func(s, u, p), expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("seed", range(4))
def test_randomized_params_in_bounds(seed):
    """Domain randomization keeps each param within +-pct and k a valid fraction."""
    rng = np.random.default_rng(seed)
    base = dyn.params_vec()
    pct = 0.2
    p = base.randomized(rng, pct)
    assert isinstance(p, dyn.QuadParams)
    assert 0.0 < p.k <= 1.0
    ratio = np.asarray(p) / np.asarray(base)
    # k may be clamped below 1+pct; every other field stays within [1-pct, 1+pct].
    others = [r for f, r in zip(dyn.PARAM_NAMES, ratio, strict=True) if f != "k"]
    assert np.all(np.asarray(others) >= 1.0 - pct - 1e-9)
    assert np.all(np.asarray(others) <= 1.0 + pct + 1e-9)
