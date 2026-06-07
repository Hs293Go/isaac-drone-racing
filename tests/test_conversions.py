import numpy as np
import pytest

from isaacrace.conversions import quat_aero_isaac, vec_enu_ned, vec_flu_frd

# Assuming your original functions are imported here
# from your_module import vec_enu_ned, vec_flu_frd, quat_aero_isaac


# =====================================================================
# Tests for vec_enu_ned
# =====================================================================


def test_vec_enu_ned_basis_vectors():
    """Test standard basis vector conversions from ENU to NED."""
    # East [1, 0, 0] -> North is index 1, East is index 0, -Down is index 2
    # ENU [1, 0, 0] should map to NED [0, 1, 0] (East)
    assert np.allclose(
        vec_enu_ned(np.array([1.0, 0.0, 0.0])), np.array([0.0, 1.0, 0.0])
    )

    # North ENU [0, 1, 0] -> NED [1, 0, 0] (North)
    assert np.allclose(
        vec_enu_ned(np.array([0.0, 1.0, 0.0])), np.array([1.0, 0.0, 0.0])
    )

    # Up ENU [0, 0, 1] -> NED [0, 0, -1] (Down is opposite of Up)
    assert np.allclose(
        vec_enu_ned(np.array([0.0, 0.0, 1.0])), np.array([0.0, 0.0, -1.0])
    )


def test_vec_enu_ned_batched():
    """Test that vec_enu_ned properly handles batched/multi-dimensional arrays."""
    vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected = np.array([[2.0, 1.0, -3.0], [5.0, 4.0, -6.0]])
    assert np.allclose(vec_enu_ned(vectors), expected)


def test_vec_enu_ned_invalid_shape():
    """Test that vec_enu_ned raises a ValueError for wrong dimensions."""
    with pytest.raises(ValueError, match=r"Input vector must be a 3D vector."):
        vec_enu_ned(np.array([1.0, 2.0]))


# =====================================================================
# Tests for vec_flu_frd
# =====================================================================


def test_vec_flu_frd_basis_vectors():
    """Test standard basis vector conversions from FLU to FRD."""
    # Forward [1, 0, 0] -> FRD [1, 0, 0]
    assert np.allclose(
        vec_flu_frd(np.array([1.0, 0.0, 0.0])), np.array([1.0, 0.0, 0.0])
    )

    # Left [0, 1, 0] -> Right is opposite -> FRD [0, -1, 0]
    assert np.allclose(
        vec_flu_frd(np.array([0.0, 1.0, 0.0])), np.array([0.0, -1.0, 0.0])
    )

    # Up [0, 0, 1] -> Down is opposite -> FRD [0, 0, -1]
    assert np.allclose(
        vec_flu_frd(np.array([0.0, 0.0, 1.0])), np.array([0.0, 0.0, -1.0])
    )


def test_vec_flu_frd_invalid_shape():
    """Test that vec_flu_frd raises ValueError for wrong dimensions."""
    with pytest.raises(ValueError, match=r"Input vector must be a 3D vector."):
        vec_flu_frd(np.array([1.0, 2.0, 3.0, 4.0]))


# =====================================================================
# Tests for quat_aero_isaac
# =====================================================================


def test_quat_aero_isaac_identity():
    """Test conversion of an identity quaternion [0, 0, 0, 1]."""
    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])
    SQRT_1_2 = 0.70710678118654757

    # Expected analytical result based on your equation:
    # x' = -(0.707 * 0) = 0
    # y' = -(0.707 * 0) = 0
    # z' = 0.707 * (0 - 1) = -0.70710678...
    # w' = -0.707 * (0 + 1) = -0.70710678...
    expected = np.array([0.0, 0.0, -SQRT_1_2, -SQRT_1_2])

    assert np.allclose(quat_aero_isaac(identity_quat), expected)


def test_quat_aero_isaac_involution():
    """Test that applying the function twice yields the original result."""
    # Note: Quaternions q and -q represent the exact same rotation.
    # We check if the double conversion results in either q or -q.
    q_original = np.array([0.1, 0.2, 0.3, 0.9])
    # Normalize it to be a valid quaternion
    q_original /= np.linalg.norm(q_original)

    q_twice = quat_aero_isaac(quat_aero_isaac(q_original))

    # Check if they match directly or as an inverse antipodal representation
    assert np.allclose(q_original, q_twice) or np.allclose(q_original, -q_twice)


def test_quat_aero_isaac_invalid_shape():
    """Test that quat_aero_isaac raises ValueError for wrong dimensions."""
    with pytest.raises(ValueError, match=r"Input quaternion must have 4 components"):
        quat_aero_isaac(np.array([0.0, 0.0, 1.0]))
