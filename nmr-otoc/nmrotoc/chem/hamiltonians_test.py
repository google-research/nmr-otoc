# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

import unittest
import numpy as np

from nmrotoc.chem.toluene import make_toluene_molecule
from nmrotoc.chem.hamiltonians import make_coupling_matrices


class TestToluene(unittest.TestCase):

    def test_make_coupling_matrices(self):
        # Re-generating the coupling matrices from position data using old method
        gyro_H = 26.7522187
        gyro_13C = 6.72828
        h = 6.62607015e-34
        vac_permeab_over_4pi = 1e-7
        Hz_to_kHz = 1e-3
        gyro_H_rad_s_T = gyro_H * 1e7
        gyro_13C_rad_s_T = gyro_13C * 1e7
        const_HH_kHz_m = (
            -Hz_to_kHz * vac_permeab_over_4pi * h * gyro_H_rad_s_T**2 / (4 * np.pi**2)
        )
        const_CH_kHz_m = (
            -Hz_to_kHz
            * vac_permeab_over_4pi
            * h
            * gyro_H_rad_s_T
            * gyro_13C_rad_s_T
            / (4 * np.pi**2)
        )
        const_HH_kHz_A = const_HH_kHz_m * 1e30

        gyro_fractions = {"H": 1, "C": const_CH_kHz_m / const_HH_kHz_m}

        def make_circle_sampler(center, radius, v1, v2, offset):
            """Generates a function that samples randomly from a circle of fixed radius around a fixed center"""

            def circle_sampler(theta):
                return center + radius * (
                    np.cos(theta + offset) * v1 + np.sin(theta + offset) * v2
                )

            return circle_sampler

        rng = np.random.RandomState(42)
        cs_methyl_group1 = make_circle_sampler(
            np.array([0, 0, 5.7510]),
            1.0299,
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            0,
        )
        cs_methyl_group2 = make_circle_sampler(
            np.array([0, 0, 5.7510]),
            1.0299,
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            2 * np.pi / 3,
        )
        cs_methyl_group3 = make_circle_sampler(
            np.array([0, 0, 5.7510]),
            1.0299,
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            4 * np.pi / 3,
        )

        pos_list = [
            np.array([0, 0, 1.0866]),
            np.array([0, 0, 0]),
            np.array([0, 2.1486, 1.2417]),
            np.array([0, -2.1486, 1.2417]),
            np.array([0, 2.1290, 3.7051]),
            np.array([0, -2.1290, 3.7051]),
            cs_methyl_group1,
            cs_methyl_group2,
            cs_methyl_group3,
        ]
        static_list = [True] * 6 + [False] * 3
        label_vec = ["C"] + ["H"] * 8
        order_parameters = np.array([-0.1758 + 0.0176, -0.0176, 0.1758])

        hom_old = np.zeros((9, 9))
        het_old = np.zeros((9, 9))

        num_reps_for_average = int(1e4)

        for j1 in range(9):
            for j2 in range(j1 + 1, 9):
                if (static_list[j1] & static_list[j2]) is True:
                    nr = 1
                else:
                    nr = num_reps_for_average
                const = (
                    const_HH_kHz_A
                    * gyro_fractions[label_vec[j1]]
                    * gyro_fractions[label_vec[j2]]
                )
                couplings = []
                for _ in range(nr):
                    theta = rng.uniform(0, 2 * np.pi)
                    pos1 = pos_list[j1] if static_list[j1] else pos_list[j1](theta)
                    pos2 = pos_list[j2] if static_list[j2] else pos_list[j2](theta)
                    r12 = np.linalg.norm(pos1 - pos2)
                    n12 = (pos1 - pos2) / r12
                    couplings.append(
                        const * (r12**-3) * np.dot(n12 * n12, order_parameters)
                    )
                if label_vec[j1] != label_vec[j2]:
                    het_old[j1, j2] = het_old[j2, j1] = np.mean(couplings)
                else:
                    hom_old[j1, j2] = hom_old[j2, j1] = np.mean(couplings)

        # Generating coupling matrices using the new methods
        molecule = make_toluene_molecule(rng)
        labels = ["C0"] + ["H{}".format(j) for j in range(1, 9)]
        hom_new, het_new = make_coupling_matrices(
            molecule, labels, order_parameters=order_parameters
        )
        print(hom_old.round(3))
        print(hom_new.round(3))
        print(het_old.round(3))
        print(het_new.round(3))

        # Comparing old and new coupling matrices
        np.testing.assert_allclose(hom_old, hom_new, atol=5e-2)  # Use a tolerance
        np.testing.assert_allclose(het_old, het_new, atol=5e-2)
