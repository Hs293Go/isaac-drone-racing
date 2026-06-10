"""Perception / flight reward-signal tests (pure NumPy; no isaacsim)."""

import numpy as np
from pytest import approx

from isaacrace import perception
from isaacrace.config import FpvConfig


def test_gate_bearing_on_axis_is_one():
    fpv = FpvConfig()  # fov 90, tilt 25
    axis_frd = perception._optical_axis_frd(fpv.tilt_deg)
    mount = np.asarray(fpv.mount, float)
    mount_frd = np.array([mount[0], -mount[1], -mount[2]])  # FLU -> FRD
    pos, euler = np.zeros(3), np.zeros(3)  # level: body FRD == world NED
    gate = pos + mount_frd + 5.0 * axis_frd  # straight down the optical axis
    cos_a = float(perception.gate_bearing(pos, euler, gate, fpv))
    assert cos_a > 1.0 - 1e-9
    assert perception.visibility_reward(cos_a, fpv.fov_deg) > 1.0 - 1e-9


def test_gate_bearing_behind_is_negative():
    fpv = FpvConfig()
    pos, euler = np.zeros(3), np.zeros(3)
    gate = np.array([-2.0, 0.0, 0.0])  # behind the drone (NED -x)
    cos_a = float(perception.gate_bearing(pos, euler, gate, fpv))
    assert cos_a < 0.0
    assert perception.visibility_reward(cos_a, fpv.fov_deg) == approx(0.0)


def test_visibility_zero_at_fov_edge():
    fpv = FpvConfig()
    cos_half = float(np.cos(np.radians(fpv.fov_deg) / 2.0))
    assert perception.visibility_reward(cos_half, fpv.fov_deg) == approx(
        0.0
    )  # at the edge
    assert perception.visibility_reward(cos_half - 0.1, fpv.fov_deg) == approx(
        0.0
    )  # outside
    assert perception.visibility_reward(1.0, fpv.fov_deg) > 0.0  # centered


def test_backpedal_forward_zero_backward_one():
    e = np.zeros(3)  # level: body nose = NED [1, 0, 0]
    assert float(perception.backpedal([1.0, 0.0, 0.0], e)) == approx(
        0.0
    )  # nose-forward
    assert float(perception.backpedal([-1.0, 0.0, 0.0], e)) == approx(1.0)  # tail-first
    assert float(perception.backpedal([0.0, 1.0, 0.0], e)) == approx(
        0.0
    )  # strafe is free
    # batched: forward / backward / strafe
    vel = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0]])
    np.testing.assert_allclose(
        perception.backpedal(vel, np.zeros((3, 3))), [0.0, 1.0, 0.0]
    )
