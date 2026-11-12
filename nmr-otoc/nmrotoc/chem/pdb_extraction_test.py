# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

import unittest
from unittest.mock import patch, Mock, MagicMock
import os
import numpy as np
import tempfile

from nmrotoc.chem.pdb.pdb_extraction import (
    get_h_containing_pdb_ids,
    parse_and_extract_atoms,
    process_protein_files,
)
from nmrotoc.chem.pdb.pdb_extraction import (
    get_h_containing_pdb_ids,
    parse_and_extract_atoms,
    process_protein_files,
)


class TestProteinFetcher(unittest.TestCase):

    # --- TEST 1: The API Query ---
    @patch("nmrotoc.chem.pdb_extraction.requests.post")
    def test_get_ids_success(self, mock_post):
        """Test that we correctly extract and deduplicate PDB IDs from the API."""

        # 1. Mock the API response
        mock_response = Mock()
        mock_response.status_code = 200
        # Simulate returning two chains from the same protein (1ABC) and one unique one (2XYZ)
        mock_response.json.return_value = {
            "result_set": [
                {"identifier": "1ABC_1"},
                {"identifier": "1ABC_2"},
                {"identifier": "2XYZ_1"},
            ]
        }
        mock_post.return_value = mock_response

        # 2. Run the function
        ids = get_h_containing_pdb_ids(min_length=50)

        # 3. Verify results
        # Should be list of 2 IDs (duplicates removed)
        self.assertEqual(len(ids), 2)
        self.assertIn("1ABC", ids)
        self.assertIn("2XYZ", ids)

    @patch("nmrotoc.chem.pdb_extraction.requests.post")
    def test_get_ids_no_content(self, mock_post):
        """Test how the function handles a 204 No Content response."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        ids = get_h_containing_pdb_ids()
        self.assertEqual(ids, [])

    # --- TEST 2: The Parser (Logic Core) ---
    def test_parse_and_extract_atoms_logic(self):
        """
        Creates a temporary sample .cif file to verify that
        Carbons and Hydrogens are sorted into the correct arrays.
        """

        # 1. Create a minimal sample CIF content
        # Contains 1 Carbon at (1,1,1) and 1 Hydrogen at (2,2,2)
        cif_content = """
data_1TST
# 
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM   1 C  CA  . ALA A 1 1 ? 1.000 1.000 1.000 1.00 0.00 1 A 1
ATOM   2 H  H   . ALA A 1 1 ? 2.000 2.000 2.000 1.00 0.00 1 A 1
ATOM   3 N  N   . ALA A 1 1 ? 3.000 3.000 3.000 1.00 0.00 1 A 1
"""
        # 2. Write to a temp file
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".cif") as tmp:
            tmp.write(cif_content)
            tmp_path = tmp.name

        try:
            # 3. Run the function
            c_coords, h_coords, success = parse_and_extract_atoms(tmp_path)  # type: ignore

            # 4. Verify Success
            self.assertTrue(success)

            # 5. Verify Carbon (Should be 1 atom at 1,1,1)
            self.assertEqual(len(c_coords), 1)
            np.testing.assert_array_equal(c_coords[0], np.array([1.0, 1.0, 1.0]))

            # 6. Verify Hydrogen (Should be 1 atom at 2,2,2)
            self.assertEqual(len(h_coords), 1)
            np.testing.assert_array_equal(h_coords[0], np.array([2.0, 2.0, 2.0]))

            # Nitrogen should be ignored by our logic

        finally:
            # Cleanup
            os.remove(tmp_path)

    def test_parse_invalid_file(self):
        """Test parsing a non-existent file."""
        c_coords, h_coords, success = parse_and_extract_atoms("non_existent_file.cif")  # type: ignore
        self.assertFalse(success)
        self.assertEqual(len(c_coords), 0)

    # --- TEST 3: The Integration/Workflow ---
    @patch("nmrotoc.chem.pdb_extraction.get_h_containing_pdb_ids")
    @patch("nmrotoc.chem.pdb_extraction.PDBList")
    @patch("nmrotoc.chem.pdb_extraction.parse_and_extract_atoms")
    def test_process_protein_files_flow(self, mock_parse, mock_pdblist, mock_get_ids):
        """
        Test the main orchestrator function.
        We mock everything so we don't actually download or parse anything.
        """

        # 1. Setup Mocks
        mock_get_ids.return_value = ["1TST"]  # API finds 1 ID

        # Mock Downloader to return a fake path
        mock_downloader_instance = mock_pdblist.return_value
        mock_downloader_instance.retrieve_pdb_file.return_value = "placeholder/path/1TST.cif"

        # Mock Parser to return fake data (success=True)
        # Returns: (Carbon Array, Hydrogen Array, Success Bool)
        mock_parse.return_value = (np.array([[1, 1, 1]]), np.array([[2, 2, 2]]), True)

        # 2. Run Function
        # Note: We patch os.path.isfile inside the function context usually,
        # but to keep this simple we assume the script checks existence.
        # For this test, we need to patch os.path.isfile to always return True.
        with patch("os.path.isfile", return_value=True):
            downloaded_files = ["placeholder/path/1TST.cif"]
            result = process_protein_files(downloaded_files)

        # 3. Assertions
        # Did we try to download?
        mock_downloader_instance.retrieve_pdb_file.assert_called()

        # Did we try to parse the file returned by the downloader?
        mock_parse.assert_called_with("placeholder/path/1TST.cif")

        # Is the result dictionary structured correctly?
        self.assertIn("1TST.cif", result)  # The key is usually the filename
        self.assertTrue(len(result["1TST.cif"]["C"]) > 0)


if __name__ == "__main__":
    unittest.main()
