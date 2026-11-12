# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Magic-angle-spinning (MAS) effective dipolar couplings.

This module implements the spatial part of the MAS average-Hamiltonian pipeline described in
the "Magic Angle Spinning (MAS)" section of the README:

1. For every spin pair, the five rank-2 spherical tensor components of the (axially symmetric)
   dipolar interaction are computed **once** in the molecular frame, see
   :func:`spatial_tensor_components`.
2. For each powder orientation they are transformed into the rotor frame with a Wigner
   ``D^2`` matrix, :func:`wigner_d2`.
3. The rotor-frame components are contracted with the MAS mechanical scaling
   ``d^2_{m0}(beta_rotor)`` and the RF sequence scaling ``eta_{m,mu}``,
   :class:`MASRecouplingModel`.

Conventions
-----------
* Wigner matrices use the Rose convention,
  ``D^2_{m'm}(alpha, beta, gamma) = exp(-i m' alpha) d^2_{m'm}(beta) exp(-i m gamma)``,
  with all 5-vectors / 5x5 matrices indexed ``m = -2, -1, 0, 1, 2``.
* Spatial components are normalized to ``A^PAS_{20} = 1``, so that a coupling constant
  ``b_ij`` multiplies them directly; for a pair along the frame z-axis ``A_{20} = 1``.
* The **stored** coupling for a pair is

  ``d_ij = b_ij * sum_m eta_m d^2_{m0}(beta_rotor) A^R_{2m}``

  which is *complex* for homonuclear (double-quantum) pairs and *real* for heteronuclear
  (``Z_p Z_i``) pairs.  With ``eta_0 = 1`` and ``beta_rotor = 0`` this reduces exactly to the
  static secular coupling ``b_ij (3 cos^2(theta) - 1) / 2`` used by the ``dipolar`` model, so
  the MAS code path is a strict generalization of the static one.
* The spin-operator convention that goes with a complex ``d_ij`` is

  ``H_hom = sum_{i<j} [ Re(d_ij) (Y_i Y_j - X_i X_j) + Im(d_ij) (X_i Y_j + Y_i X_j) ]``

  i.e. the phase of ``d_ij`` rotates the double-quantum operator about z, exactly as a
  rotor-phase (gamma-encoding) factor should.  Sequence-specific overall factors (e.g. the
  HORROR/DREAM ``kappa = 1 / (4 sqrt(2))``) are deliberately **not** baked in; they are applied
  downstream through ``--eta_hom`` / ``--eta_het``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

MAGIC_ANGLE_RAD: float = float(np.arccos(1.0 / np.sqrt(3.0)))
"""The magic angle, ``arccos(1/sqrt(3)) = 54.7356...`` degrees, in radians."""

M_VALUES: tuple[int, ...] = (-2, -1, 0, 1, 2)
"""Order of the ``m`` index used for all rank-2 vectors and matrices in this module."""

_SQRT_3_8 = np.sqrt(3.0 / 8.0)


def _m_index(m: int) -> int:
    """Returns the array index of a magnetic quantum number ``m`` in ``M_VALUES``.

    Args:
        m: Magnetic quantum number in ``[-2, 2]``.

    Returns:
        The corresponding index into a length-5 array.

    Raises:
        ValueError: If ``m`` is out of range.
    """
    if m not in M_VALUES:
        msg = f"m must be one of {M_VALUES}, got {m}."
        raise ValueError(msg)
    return m + 2


def wigner_small_d2(beta: float) -> NDArray:
    """Returns the real 5x5 Wigner small-d matrix ``d^2_{m'm}(beta)``.

    Rows index ``m'`` and columns index ``m``, both ordered ``-2, -1, 0, 1, 2``.

    Args:
        beta: Rotation angle about the y-axis in radians.

    Returns:
        The 5x5 matrix ``d^2_{m'm}(beta)`` (Rose convention).
    """
    c = np.cos(beta)
    s = np.sin(beta)
    s2 = np.sin(2.0 * beta)
    plus = (1.0 + c) / 2.0
    minus = (1.0 - c) / 2.0

    mat = np.zeros((5, 5))
    # Rows m' = 2, 1, 0 written out explicitly (columns m = -2, -1, 0, 1, 2).
    mat[_m_index(2), :] = [minus**2, -minus * s, _SQRT_3_8 * s**2, -plus * s, plus**2]
    mat[_m_index(1), :] = [
        -minus * s,
        minus * (2.0 * c + 1.0),
        -_SQRT_3_8 * s2,
        plus * (2.0 * c - 1.0),
        plus * s,
    ]
    mat[_m_index(0), :] = [
        _SQRT_3_8 * s**2,
        -_SQRT_3_8 * s2,
        (3.0 * c**2 - 1.0) / 2.0,
        _SQRT_3_8 * s2,
        _SQRT_3_8 * s**2,
    ]
    # Rows m' = -1, -2 follow from d^l_{-m',-m} = (-1)^(m'-m) d^l_{m'm}.
    for mp in (1, 2):
        for m in M_VALUES:
            mat[_m_index(-mp), _m_index(-m)] = ((-1.0) ** (mp - m)) * mat[_m_index(mp), _m_index(m)]
    return mat


def wigner_d2(alpha: float, beta: float, gamma: float) -> NDArray:
    """Returns the complex 5x5 Wigner matrix ``D^2_{m'm}(alpha, beta, gamma)``.

    Args:
        alpha: First Euler angle in radians.
        beta: Second Euler angle in radians.
        gamma: Third Euler angle in radians.

    Returns:
        ``exp(-i m' alpha) d^2_{m'm}(beta) exp(-i m gamma)`` as a 5x5 complex array.
    """
    ms = np.array(M_VALUES, dtype=np.float64)
    phase_row = np.exp(-1j * ms * alpha)
    phase_col = np.exp(-1j * ms * gamma)
    return phase_row[:, np.newaxis] * wigner_small_d2(beta) * phase_col[np.newaxis, :]


def rotation_wigner_d2(euler_zyz: tuple[float, float, float] | NDArray) -> NDArray:
    """Returns the transformation matrix for rank-2 components under a frame rotation.

    If ``Q = Rz(alpha) Ry(beta) Rz(gamma)`` maps molecular-frame vector components to
    rotor-frame components (``u = Q v``), then rank-2 spherical tensor components transform as

    ``A^R_{2m} = sum_{m'} A^M_{2m'} W[m', m]``  with  ``W = D^2(gamma, beta, alpha)``.

    (The Euler angles appear reversed because ``A_{2m}`` is defined here through the *column*
    index of ``D^2``; the relation is verified numerically against direct vector rotation in
    ``tests/test_mas.py``.)

    Args:
        euler_zyz: The ZYZ Euler angles ``(alpha, beta, gamma)`` of the rotation ``Q``.

    Returns:
        The 5x5 complex transformation matrix ``W``.
    """
    alpha, beta, gamma = (float(x) for x in euler_zyz)
    return wigner_d2(gamma, beta, alpha)


def spatial_tensor_components(unit_vectors: NDArray) -> NDArray:
    """Returns the rank-2 spatial tensor components of axially symmetric couplings.

    For a unit vector with polar angles ``(theta, phi)`` the components are
    ``A_{2m} = d^2_{0m}(theta) exp(-i m phi)``, evaluated here in the numerically robust
    Cartesian form using ``zeta = n_x + i n_y`` (no ``arccos``/``arctan2``/division by
    ``sin(theta)``).

    Args:
        unit_vectors: Array of unit vectors with shape ``(..., 3)``.

    Returns:
        Complex array of shape ``(..., 5)`` holding ``A_{2,-2} ... A_{2,2}``.
    """
    n = np.asarray(unit_vectors, dtype=np.float64)
    nz = n[..., 2]
    zeta = n[..., 0] + 1j * n[..., 1]
    zeta_bar = np.conjugate(zeta)

    out = np.empty((*n.shape[:-1], 5), dtype=np.complex128)
    out[..., _m_index(-2)] = _SQRT_3_8 * zeta**2
    out[..., _m_index(-1)] = -_SQRT_3_8 * 2.0 * nz * zeta
    out[..., _m_index(0)] = (3.0 * nz**2 - 1.0) / 2.0
    out[..., _m_index(1)] = _SQRT_3_8 * 2.0 * nz * zeta_bar
    out[..., _m_index(2)] = _SQRT_3_8 * zeta_bar**2
    return out


def _eta_vector(eta: Mapping[int, complex], *, conjugate_symmetric: bool) -> NDArray:
    """Expands an ``{m: eta_m}`` mapping into a length-5 complex vector.

    Args:
        eta: Mapping from ``m`` to its (complex) RF scaling factor.  Missing entries are zero.
        conjugate_symmetric: If True, enforce ``eta_{-m} = conj(eta_m)``, which is required for
            Hermitian ``Delta M = 0`` (e.g. ``Z_p Z_i``) spin operators.

    Returns:
        A length-5 complex vector ordered by :data:`M_VALUES`.

    Raises:
        ValueError: If ``conjugate_symmetric`` is requested and the mapping specifies both
            ``m`` and ``-m`` with values that are not complex conjugates.
    """
    vec = np.zeros(5, dtype=np.complex128)
    for m, value in eta.items():
        vec[_m_index(int(m))] = complex(value)
    if conjugate_symmetric:
        for m in (1, 2):
            pos, neg = vec[_m_index(m)], vec[_m_index(-m)]
            if pos != 0.0 and neg != 0.0 and not np.isclose(neg, np.conjugate(pos)):
                msg = (
                    f"eta[{-m}] must equal conj(eta[{m}]) for a Hermitian Delta-M = 0 term, "
                    f"got {neg} and conj({pos}) = {np.conjugate(pos)}."
                )
                raise ValueError(msg)
            if pos != 0.0 and neg == 0.0:
                vec[_m_index(-m)] = np.conjugate(pos)
            elif neg != 0.0 and pos == 0.0:
                vec[_m_index(m)] = np.conjugate(neg)
    return vec


@dataclass(frozen=True)
class MASRecouplingModel:
    """RF + mechanical scaling of the rotor-frame spatial components.

    The effective coupling of a pair is ``d = b * sum_m eta_m d^2_{m0}(rotor_angle) A^R_{2m}``,
    evaluated separately for the homonuclear double-quantum term (``eta_dq``, ``Delta M = +2``,
    complex) and for the heteronuclear ``Z_p Z_i`` term (``eta_zz``, ``Delta M = 0``, real).

    A recoupling sequence works by matching a **single** spatial Fourier component ``m`` to the
    spin operator; keeping several components with equal weights would simply resum them back
    into the static (rotor phase zero) coupling and destroy the gamma encoding.

    The ``eta`` mappings select the recoupled components and their relative weights.  Instances
    generated from the command line use unit weights (:meth:`single_component`): as for the
    static coupling models, the overall RF scaling of a pulse sequence is applied at simulation
    time through ``--eta_hom`` / ``--eta_het``, so the same instances can be reused for
    different sequences.

    Attributes:
        eta_dq: Mapping ``{m: eta}`` for the double-quantum spin component.
        eta_zz: Mapping ``{m: eta}`` for the ``Z_p Z_i`` spin component.  It is completed to be
            conjugate symmetric so that the resulting coupling is real.
        rotor_angle: Angle between the rotor axis and the magnetic field, in radians.
    """

    eta_dq: Mapping[int, complex] = field(default_factory=lambda: {-1: 1.0 + 0.0j})
    eta_zz: Mapping[int, complex] = field(default_factory=lambda: {1: 1.0 + 0.0j})
    rotor_angle: float = MAGIC_ANGLE_RAD

    @classmethod
    def single_component(
        cls,
        dq_order: int = 1,
        zz_order: int = 1,
        rotor_angle: float = MAGIC_ANGLE_RAD,
    ) -> MASRecouplingModel:
        """Builds a model that recouples a single Fourier component per spin operator.

        The selected components get unit weight; the RF scaling of the pulse sequence is not
        part of the model and is applied at simulation time (``--eta_hom`` / ``--eta_het``).

        Args:
            dq_order: ``|m|`` recoupled for the double-quantum term; the component ``m =
                -dq_order`` is selected (its conjugate partner is carried by the Hermitian
                conjugate spin operator).  Use ``0`` to switch the homonuclear term off.
            zz_order: ``|m|`` recoupled for the ``Z_p Z_i`` term; both ``+m`` and ``-m`` enter
                (conjugate symmetric).  Use ``0`` to switch the heteronuclear term off.
            rotor_angle: Rotor angle in radians (defaults to the magic angle).

        Returns:
            The configured :class:`MASRecouplingModel`.
        """
        dq: dict[int, complex] = {} if dq_order == 0 else {-int(dq_order): 1.0 + 0.0j}
        zz: dict[int, complex] = {} if zz_order == 0 else {int(zz_order): 1.0 + 0.0j}
        return cls(eta_dq=dq, eta_zz=zz, rotor_angle=rotor_angle)

    @classmethod
    def static(cls, eta: complex = 1.0) -> MASRecouplingModel:
        """Builds the non-spinning model (``m = 0`` only, rotor angle zero).

        This reproduces the static secular coupling ``b (3 cos^2(theta) - 1) / 2`` exactly and
        exists mainly as a regression reference for the ``dipolar`` coupling mode.

        Args:
            eta: Overall scaling factor.

        Returns:
            The configured :class:`MASRecouplingModel`.
        """
        return cls(eta_dq={0: complex(eta)}, eta_zz={0: complex(eta)}, rotor_angle=0.0)

    def mechanical_scaling(self) -> NDArray:
        """Returns the length-5 vector ``d^2_{m0}(rotor_angle)``."""
        return wigner_small_d2(self.rotor_angle)[:, _m_index(0)].astype(np.complex128)

    def contraction_vectors(self, euler_zyz: tuple[float, float, float] | NDArray) -> NDArray:
        """Returns the molecular-frame contraction vectors for one orientation.

        The returned array ``u`` has shape ``(2, 5)``; contracting the molecular-frame
        components with ``u[0]`` gives the double-quantum coupling and with ``u[1]`` the
        ``Z_p Z_i`` coupling (before taking the real part), i.e.

        ``sum_m eta_m d^2_{m0} A^R_{2m} = sum_{m'} A^M_{2m'} u[k, m']``.

        Args:
            euler_zyz: ZYZ Euler angles of the molecular-to-rotor rotation.

        Returns:
            Complex array of shape ``(2, 5)``.
        """
        wig = rotation_wigner_d2(euler_zyz)
        scaling = self.mechanical_scaling()
        dq = _eta_vector(self.eta_dq, conjugate_symmetric=False) * scaling
        zz = _eta_vector(self.eta_zz, conjugate_symmetric=True) * scaling
        return np.stack([wig @ dq, wig @ zz])

    def to_metadata(self) -> dict[str, Any]:
        """Returns a JSON-serializable description of the model."""

        def as_pairs(eta: Mapping[int, complex]) -> dict[str, list[float]]:
            return {str(m): [float(np.real(v)), float(np.imag(v))] for m, v in eta.items()}

        return {
            "eta_dq": as_pairs(self.eta_dq),
            "eta_zz": as_pairs(self.eta_zz),
            "rotor_angle_deg": float(np.degrees(self.rotor_angle)),
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> MASRecouplingModel:
        """Rebuilds a model from the output of :meth:`to_metadata`.

        Other entries, such as the grid description that ``nmr-generate-instances`` stores in
        the same ``"mas"`` block of ``metadata.json``, are ignored.

        Args:
            metadata: Mapping with the ``"eta_dq"``, ``"eta_zz"`` and ``"rotor_angle_deg"``
                entries written by :meth:`to_metadata`.

        Returns:
            The equivalent :class:`MASRecouplingModel`.

        Raises:
            KeyError: If a required entry is missing.
        """

        def from_pairs(pairs: Mapping[str, Sequence[float]]) -> dict[int, complex]:
            return {int(m): complex(real, imag) for m, (real, imag) in pairs.items()}

        return cls(
            eta_dq=from_pairs(metadata["eta_dq"]),
            eta_zz=from_pairs(metadata["eta_zz"]),
            rotor_angle=float(np.radians(metadata["rotor_angle_deg"])),
        )


def effective_coupling_matrix(
    spatial: NDArray,
    prefactors: NDArray,
    same_species: NDArray,
    model: MASRecouplingModel,
    euler_zyz: tuple[float, float, float] | NDArray,
) -> NDArray:
    """Computes the complex effective coupling matrix for one powder orientation.

    Args:
        spatial: Molecular-frame rank-2 components of shape ``(N, N, 5)``.
        prefactors: Real coupling constants ``b_ij`` of shape ``(N, N)``.
        same_species: Boolean mask of shape ``(N, N)``; ``True`` marks homonuclear pairs, which
            keep the complex double-quantum phase.  Heteronuclear pairs are projected onto the
            real axis, as required by the Hermitian ``Z_p Z_i`` operator.
        model: The MAS recoupling model supplying ``eta`` and the rotor angle.
        euler_zyz: ZYZ Euler angles of the molecular-to-rotor rotation for this orientation.

    Returns:
        Complex symmetric matrix of shape ``(N, N)`` with a zero diagonal.
    """
    contractions = model.contraction_vectors(euler_zyz)
    dq = prefactors * (spatial @ contractions[0])
    zz = prefactors * (spatial @ contractions[1])
    coupling = np.where(same_species, dq, np.real(zz).astype(np.complex128))
    np.fill_diagonal(coupling, 0.0)
    return coupling
