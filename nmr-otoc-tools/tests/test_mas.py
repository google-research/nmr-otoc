# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the MAS (magic-angle-spinning) effective coupling model."""

import json

import numpy as np
import pytest

from nmr_otoc_tools.mas import (
    MAGIC_ANGLE_RAD,
    MASRecouplingModel,
    effective_coupling_matrix,
    rotation_wigner_d2,
    spatial_tensor_components,
    wigner_d2,
    wigner_small_d2,
)
from nmr_otoc_tools.orientations import rotation_zyz, so3_grid

_SQRT_3_8 = np.sqrt(3.0 / 8.0)


def _random_unit_vectors(num: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(num, 3))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _two_spin_system(unit_vector: np.ndarray, prefactor: float, *, homonuclear: bool) -> tuple:
    """Builds the (spatial, prefactors, same_species) triple for a single pair."""
    spatial = np.zeros((2, 2, 5), dtype=np.complex128)
    components = spatial_tensor_components(unit_vector)
    spatial[0, 1] = components
    spatial[1, 0] = components
    prefactors = np.array([[0.0, prefactor], [prefactor, 0.0]])
    same_species = np.array([[False, homonuclear], [homonuclear, False]])
    return spatial, prefactors, same_species


def test_wigner_small_d2_identity_and_orthogonality() -> None:
    """d^2(0) is the identity and d^2 is orthogonal with d^2(-b) = d^2(b)^T."""
    np.testing.assert_allclose(wigner_small_d2(0.0), np.eye(5), atol=1e-15)
    beta = 0.83
    mat = wigner_small_d2(beta)
    np.testing.assert_allclose(mat @ mat.T, np.eye(5), atol=1e-14)
    np.testing.assert_allclose(wigner_small_d2(-beta), mat.T, atol=1e-14)


def test_wigner_small_d2_symmetry_relation() -> None:
    """The table obeys d^l_{m'm} = (-1)^(m'-m) d^l_{m m'}."""
    mat = wigner_small_d2(0.61)
    for i, mp in enumerate((-2, -1, 0, 1, 2)):
        for j, m in enumerate((-2, -1, 0, 1, 2)):
            assert np.isclose(mat[i, j], ((-1.0) ** (mp - m)) * mat[j, i])


def test_wigner_small_d2_magic_angle_values() -> None:
    """The MAS mechanical scaling factors take their textbook values."""
    mat = wigner_small_d2(MAGIC_ANGLE_RAD)
    zero_column = mat[:, 2]  # d^2_{m,0}
    assert np.isclose(zero_column[2], 0.0, atol=1e-15)  # m = 0 is nulled by the magic angle
    assert np.isclose(zero_column[3], -1.0 / np.sqrt(3.0))  # m = +1
    assert np.isclose(zero_column[1], 1.0 / np.sqrt(3.0))  # m = -1
    assert np.isclose(zero_column[4], 1.0 / np.sqrt(6.0))  # m = +2
    assert np.isclose(zero_column[0], 1.0 / np.sqrt(6.0))  # m = -2


def test_wigner_d2_phase_factors() -> None:
    """D^2 adds the expected exp(-i m' alpha) / exp(-i m gamma) phases."""
    alpha, beta, gamma = 0.4, 1.0, -0.9
    small = wigner_small_d2(beta)
    full = wigner_d2(alpha, beta, gamma)
    for i, mp in enumerate((-2, -1, 0, 1, 2)):
        for j, m in enumerate((-2, -1, 0, 1, 2)):
            expected = np.exp(-1j * mp * alpha) * small[i, j] * np.exp(-1j * m * gamma)
            assert np.isclose(full[i, j], expected)


def test_spatial_tensor_components_reference_values() -> None:
    """Components match the closed forms for the coordinate axes."""
    z_components = spatial_tensor_components(np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(z_components, [0.0, 0.0, 1.0, 0.0, 0.0], atol=1e-15)

    x_components = spatial_tensor_components(np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(
        x_components, [_SQRT_3_8, 0.0, -0.5, 0.0, _SQRT_3_8], atol=1e-15
    )


def test_spatial_tensor_components_norm_invariant() -> None:
    """sum_m |A_2m|^2 = 1 is a rotational invariant of the normalized tensor."""
    components = spatial_tensor_components(_random_unit_vectors(50, seed=1))
    np.testing.assert_allclose(np.sum(np.abs(components) ** 2, axis=-1), 1.0, atol=1e-12)


def test_spatial_tensor_components_conjugation_relation() -> None:
    """A_{2,-m} = (-1)^m conj(A_{2m}), which makes the real-space tensor real."""
    components = spatial_tensor_components(_random_unit_vectors(20, seed=2))
    for m in (1, 2):
        expected = ((-1.0) ** m) * np.conjugate(components[..., m + 2])
        np.testing.assert_allclose(components[..., -m + 2], expected, atol=1e-14)


def test_spatial_tensor_inversion_symmetry() -> None:
    """The rank-2 tensor is even under inversion of the internuclear vector."""
    vectors = _random_unit_vectors(10, seed=7)
    np.testing.assert_allclose(
        spatial_tensor_components(vectors), spatial_tensor_components(-vectors), atol=1e-14
    )


def test_rotation_law_matches_direct_rotation() -> None:
    """Rotating the components with D^2 equals recomputing them from the rotated vector."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        euler = (
            rng.uniform(0, 2 * np.pi),
            np.arccos(rng.uniform(-1, 1)),
            rng.uniform(0, 2 * np.pi),
        )
        rotation = rotation_zyz(*euler)
        vector = _random_unit_vectors(1, seed=int(rng.integers(1 << 30)))[0]
        direct = spatial_tensor_components(rotation @ vector)
        via_wigner = spatial_tensor_components(vector) @ rotation_wigner_d2(euler)
        np.testing.assert_allclose(via_wigner, direct, atol=1e-13)


def test_static_model_reproduces_secular_dipolar_coupling() -> None:
    """With m = 0 only and a zero rotor angle the model is the static secular coupling."""
    rng = np.random.default_rng(6)
    euler = (0.7, 1.3, -0.2)
    rotation = rotation_zyz(*euler)
    vector = rng.normal(size=3)
    vector /= np.linalg.norm(vector)
    prefactor = -3.7

    spatial, prefactors, same_species = _two_spin_system(vector, prefactor, homonuclear=True)
    coupling = effective_coupling_matrix(
        spatial, prefactors, same_species, MASRecouplingModel.static(), euler
    )

    rotor_axis = rotation.T @ np.array([0.0, 0.0, 1.0])
    cos_theta = float(np.dot(rotor_axis, vector))
    expected = prefactor * (1.5 * cos_theta**2 - 0.5)
    assert np.isclose(coupling[0, 1].real, expected)
    assert np.isclose(coupling[0, 1].imag, 0.0, atol=1e-15)


def test_magic_angle_nulls_the_m0_component() -> None:
    """At the magic angle the m = 0 spatial component is averaged away exactly."""
    vector = _random_unit_vectors(1, seed=8)[0]
    spatial, prefactors, same_species = _two_spin_system(vector, 2.0, homonuclear=True)
    model = MASRecouplingModel(eta_dq={0: 1.0}, eta_zz={0: 1.0}, rotor_angle=MAGIC_ANGLE_RAD)
    coupling = effective_coupling_matrix(spatial, prefactors, same_species, model, (0.3, 1.1, 0.5))
    np.testing.assert_allclose(coupling, np.zeros((2, 2)), atol=1e-15)


def test_default_model_is_gamma_encoded() -> None:
    """The default (m = -1) model has a twist-independent magnitude and a linear phase."""
    vector = _random_unit_vectors(1, seed=9)[0]
    spatial, prefactors, same_species = _two_spin_system(vector, 1.5, homonuclear=True)
    model = MASRecouplingModel.single_component()
    n_twist = 8
    grid = so3_grid(6, n_twist)

    values = np.array(
        [
            effective_coupling_matrix(spatial, prefactors, same_species, model, grid.euler_zyz[j])[
                0, 1
            ]
            for j in range(n_twist)
        ]
    )
    np.testing.assert_allclose(np.abs(values), np.abs(values[0]), atol=1e-12)
    phases = np.unwrap(np.angle(values) - np.angle(values[0]))
    np.testing.assert_allclose(phases, 2.0 * np.pi * np.arange(n_twist) / n_twist, atol=1e-10)


def test_default_model_amplitude_is_sin_two_beta_over_two_sqrt_two() -> None:
    """|d| = |b sin(2 beta)| / (2 sqrt(2)), i.e. twice the HORROR/DREAM scaling factor."""
    vector = np.array([0.0, 0.0, 1.0])
    prefactor = 2.0
    spatial, prefactors, same_species = _two_spin_system(vector, prefactor, homonuclear=True)
    model = MASRecouplingModel.single_component()
    for beta in (0.0, 0.3, np.pi / 4, 1.2, np.pi / 2):
        # Rotating the pair (which lies along z) by beta about y puts it at polar angle beta.
        euler = (0.0, beta, 0.0)
        coupling = effective_coupling_matrix(spatial, prefactors, same_species, model, euler)
        expected = abs(prefactor * np.sin(2.0 * beta)) / (2.0 * np.sqrt(2.0))
        assert np.isclose(abs(coupling[0, 1]), expected, atol=1e-12)


def test_heteronuclear_coupling_is_real() -> None:
    """Heteronuclear (Z_p Z_i) entries stay real, as Hermiticity requires."""
    vector = _random_unit_vectors(1, seed=10)[0]
    spatial, prefactors, same_species = _two_spin_system(vector, 1.1, homonuclear=False)
    model = MASRecouplingModel.single_component()
    coupling = effective_coupling_matrix(spatial, prefactors, same_species, model, (0.4, 0.9, 1.7))
    np.testing.assert_allclose(coupling.imag, np.zeros((2, 2)), atol=1e-15)
    assert not np.isclose(coupling[0, 1].real, 0.0)


def test_heteronuclear_coupling_depends_on_rotor_phase() -> None:
    """The Z_p Z_i term is not gamma encoded: its real amplitude varies with the twist."""
    vector = _random_unit_vectors(1, seed=12)[0]
    spatial, prefactors, same_species = _two_spin_system(vector, 1.1, homonuclear=False)
    model = MASRecouplingModel.single_component()
    grid = so3_grid(6, 8)
    values = np.array(
        [
            effective_coupling_matrix(spatial, prefactors, same_species, model, grid.euler_zyz[j])[
                0, 1
            ].real
            for j in range(8)
        ]
    )
    assert values.std() > 1e-3


def test_coupling_matrix_is_symmetric_with_zero_diagonal() -> None:
    """The stored matrix is complex symmetric (not Hermitian) with no self-coupling."""
    vector = _random_unit_vectors(1, seed=13)[0]
    spatial, prefactors, same_species = _two_spin_system(vector, 3.0, homonuclear=True)
    coupling = effective_coupling_matrix(
        spatial, prefactors, same_species, MASRecouplingModel.single_component(), (0.2, 0.8, 1.4)
    )
    np.testing.assert_allclose(coupling, coupling.T, atol=1e-15)
    np.testing.assert_allclose(np.diag(coupling), np.zeros(2), atol=1e-15)


def test_powder_average_second_moment() -> None:
    """The powder average of |d|^2 matches the analytic value b^2 / 15."""
    vector = _random_unit_vectors(1, seed=14)[0]
    prefactor = 2.5
    spatial, prefactors, same_species = _two_spin_system(vector, prefactor, homonuclear=True)
    model = MASRecouplingModel.single_component()
    grid = so3_grid(26, 4)

    second_moment = sum(
        weight
        * abs(effective_coupling_matrix(spatial, prefactors, same_species, model, euler)[0, 1]) ** 2
        for weight, euler in zip(grid.weights, grid.euler_zyz, strict=True)
    )
    assert np.isclose(second_moment, prefactor**2 / 15.0, rtol=1e-10)


def test_powder_average_of_coupling_vanishes() -> None:
    """The powder average of the (phased) coupling itself vanishes."""
    vector = _random_unit_vectors(1, seed=15)[0]
    spatial, prefactors, same_species = _two_spin_system(vector, 2.5, homonuclear=True)
    model = MASRecouplingModel.single_component()
    grid = so3_grid(26, 4)
    mean = sum(
        weight * effective_coupling_matrix(spatial, prefactors, same_species, model, euler)[0, 1]
        for weight, euler in zip(grid.weights, grid.euler_zyz, strict=True)
    )
    assert abs(mean) < 1e-12


def test_single_component_disables_terms() -> None:
    """Order 0 switches the corresponding spin term off."""
    vector = _random_unit_vectors(1, seed=16)[0]
    euler = (0.5, 1.0, 2.0)

    spatial, prefactors, same_species = _two_spin_system(vector, 1.0, homonuclear=True)
    model = MASRecouplingModel.single_component(dq_order=0)
    coupling = effective_coupling_matrix(spatial, prefactors, same_species, model, euler)
    np.testing.assert_allclose(coupling, np.zeros((2, 2)), atol=1e-15)

    spatial, prefactors, same_species = _two_spin_system(vector, 1.0, homonuclear=False)
    model = MASRecouplingModel.single_component(zz_order=0)
    coupling = effective_coupling_matrix(spatial, prefactors, same_species, model, euler)
    np.testing.assert_allclose(coupling, np.zeros((2, 2)), atol=1e-15)


def test_second_order_recoupling_amplitude() -> None:
    """Selecting |m| = 2 gives the sin^2(beta) amplitude with a 2 gamma phase."""
    vector = np.array([0.0, 0.0, 1.0])
    prefactor = 2.0
    spatial, prefactors, same_species = _two_spin_system(vector, prefactor, homonuclear=True)
    model = MASRecouplingModel.single_component(dq_order=2)
    beta = 0.9
    coupling = effective_coupling_matrix(spatial, prefactors, same_species, model, (0.0, beta, 0.0))
    expected = abs(prefactor) * np.sqrt(3.0 / 8.0) * np.sin(beta) ** 2 / np.sqrt(6.0)
    assert np.isclose(abs(coupling[0, 1]), expected, atol=1e-12)

    n_twist = 6
    grid = so3_grid(6, n_twist)
    values = np.array(
        [
            effective_coupling_matrix(spatial, prefactors, same_species, model, grid.euler_zyz[j])[
                0, 1
            ]
            for j in range(n_twist)
        ]
    )
    np.testing.assert_allclose(np.abs(values), np.abs(values[0]), atol=1e-12)
    phases = np.unwrap(np.angle(values) - np.angle(values[0]))
    np.testing.assert_allclose(phases, 2.0 * 2.0 * np.pi * np.arange(n_twist) / n_twist, atol=1e-10)


def test_eta_zz_conjugate_symmetry_is_completed() -> None:
    """Specifying only eta_zz[+m] implies eta_zz[-m] = conj(eta_zz[+m])."""
    vector = _random_unit_vectors(1, seed=17)[0]
    spatial, prefactors, same_species = _two_spin_system(vector, 1.0, homonuclear=False)
    euler = (0.5, 1.0, 2.0)
    implicit = MASRecouplingModel(eta_dq={}, eta_zz={1: 1.0})
    explicit = MASRecouplingModel(eta_dq={}, eta_zz={1: 1.0, -1: 1.0})
    np.testing.assert_allclose(
        effective_coupling_matrix(spatial, prefactors, same_species, implicit, euler),
        effective_coupling_matrix(spatial, prefactors, same_species, explicit, euler),
        atol=1e-15,
    )


def test_eta_zz_rejects_non_hermitian_specification() -> None:
    """A non-conjugate-symmetric eta_zz is rejected rather than silently made real."""
    vector = _random_unit_vectors(1, seed=18)[0]
    spatial, prefactors, same_species = _two_spin_system(vector, 1.0, homonuclear=False)
    model = MASRecouplingModel(eta_dq={}, eta_zz={1: 1.0j, -1: 1.0j})
    with pytest.raises(ValueError, match="must equal conj"):
        effective_coupling_matrix(spatial, prefactors, same_species, model, (0.1, 0.2, 0.3))


def test_model_metadata_round_trip() -> None:
    """The metadata dictionary is JSON friendly and records the model faithfully."""
    model = MASRecouplingModel.single_component(dq_order=2, zz_order=0)
    meta = model.to_metadata()
    assert meta["eta_dq"] == {"-2": [1.0, 0.0]}
    assert meta["eta_zz"] == {}
    assert np.isclose(meta["rotor_angle_deg"], np.degrees(MAGIC_ANGLE_RAD))

    # General (complex) weights remain available through the Python API.
    general = MASRecouplingModel(eta_dq={-2: 0.5 + 0.25j}, eta_zz={})
    assert general.to_metadata()["eta_dq"] == {"-2": [0.5, 0.25]}


@pytest.mark.parametrize(
    "model",
    [
        MASRecouplingModel.single_component(),
        MASRecouplingModel.single_component(dq_order=2, zz_order=0, rotor_angle=np.radians(50.0)),
        MASRecouplingModel.static(0.5),
        MASRecouplingModel(eta_dq={-2: 0.5 + 0.25j}, eta_zz={}),
    ],
)
def test_model_from_metadata_inverts_to_metadata(model: MASRecouplingModel) -> None:
    """from_metadata rebuilds the model from JSON, ignoring the grid entries stored with it."""
    block = {"n_orientations": 48, "grid": "lebedev_x_uniform_twist", **model.to_metadata()}
    restored = MASRecouplingModel.from_metadata(json.loads(json.dumps(block)))
    assert restored.eta_dq == model.eta_dq
    assert restored.eta_zz == model.eta_zz
    assert np.isclose(restored.rotor_angle, model.rotor_angle, rtol=0.0, atol=1e-15)


def test_model_from_metadata_requires_all_entries() -> None:
    """A metadata block without the model entries cannot be turned into a model."""
    with pytest.raises(KeyError, match="eta_dq"):
        MASRecouplingModel.from_metadata({"n_orientations": 48})


def test_single_component_uses_unit_weights() -> None:
    """The CLI model carries no RF scaling; eta is applied at simulation time."""
    model = MASRecouplingModel.single_component(dq_order=1, zz_order=2)
    assert model.eta_dq == {-1: 1.0 + 0.0j}
    assert model.eta_zz == {2: 1.0 + 0.0j}
    with pytest.raises(TypeError):
        MASRecouplingModel.single_component(eta_dq=0.5)  # type: ignore[call-arg]
