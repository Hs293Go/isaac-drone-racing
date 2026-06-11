"""Frame conversions: known answers, involutions, and a scipy ground-truth check."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from isaacrace.conversions import quat_aero_isaac, vec_enu_ned, vec_flu_frd


def test_vec_enu_ned_basis_vectors():
    # East -> ENU x == NED y; North -> ENU y == NED x; Up -> -Down
    np.testing.assert_allclose(vec_enu_ned(np.array([1.0, 0, 0])), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(vec_enu_ned(np.array([0, 1.0, 0])), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(vec_enu_ned(np.array([0, 0, 1.0])), [0.0, 0.0, -1.0])


def test_vec_flu_frd_basis_vectors():
    # Forward unchanged; Left -> -Right; Up -> -Down
    np.testing.assert_allclose(vec_flu_frd(np.array([1.0, 0, 0])), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(vec_flu_frd(np.array([0, 1.0, 0])), [0.0, -1.0, 0.0])
    np.testing.assert_allclose(vec_flu_frd(np.array([0, 0, 1.0])), [0.0, 0.0, -1.0])


@pytest.mark.parametrize("conv", [vec_enu_ned, vec_flu_frd])
def test_vec_conversions_are_involutions(conv):
    # Both maps are their own inverse: applying twice returns the input (batched).
    v = np.random.default_rng(0).standard_normal((32, 3))
    np.testing.assert_allclose(conv(conv(v)), v)


def test_vec_enu_ned_batched():
    vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(vec_enu_ned(vectors), [[2, 1, -3], [5, 4, -6]])


@pytest.mark.parametrize("conv", [vec_enu_ned, vec_flu_frd])
def test_vec_conversions_reject_non_3d(conv):
    with pytest.raises(ValueError, match="3D vector"):
        conv(np.array([1.0, 2.0]))


def test_quat_aero_isaac_involution():
    # Two applications return the original rotation (up to quaternion sign).
    rng = np.random.default_rng(1)
    q = Rotation.random(16, random_state=rng).as_quat()
    q2 = quat_aero_isaac(quat_aero_isaac(q))
    sign = np.sign(np.sum(q * q2, axis=-1, keepdims=True))
    np.testing.assert_allclose(sign * q2, q, atol=1e-12)


def test_quat_aero_isaac_matches_vector_conversions():
    # Ground truth: rotating in one convention then converting the vector must equal
    # converting the quaternion and rotating the converted vector. I.e. for any
    # attitude q (FLU body -> ENU world) and body vector v_flu:
    #   vec_enu_ned(R(q) @ v_flu) == R(quat_aero_isaac(q)) @ vec_flu_frd(v_flu)
    rng = np.random.default_rng(2)
    q_isaac = Rotation.random(32, random_state=rng).as_quat()
    v_flu = rng.standard_normal((32, 3))
    world_enu = Rotation.from_quat(q_isaac).apply(v_flu)
    world_ned = Rotation.from_quat(quat_aero_isaac(q_isaac)).apply(vec_flu_frd(v_flu))
    np.testing.assert_allclose(vec_enu_ned(world_enu), world_ned, atol=1e-12)


def test_quat_aero_isaac_level_yaw_identity():
    # The aero identity attitude (FRD aligned with NED: nose North) corresponds in
    # Isaac convention to a +90 deg yaw about ENU z (nose +y/North). Converting that
    # Isaac quat must give the aero identity (up to sign).
    q_isaac = Rotation.from_euler("z", np.pi / 2).as_quat()
    q_aero = quat_aero_isaac(q_isaac)
    angle = Rotation.from_quat(q_aero).magnitude()
    assert angle == pytest.approx(0.0, abs=1e-12)


def test_quat_aero_isaac_rejects_non_quaternion():
    with pytest.raises(ValueError, match="4 components"):
        quat_aero_isaac(np.array([0.0, 0.0, 1.0]))
