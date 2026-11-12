# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

import requests
import os
import numpy as np
import json
from Bio.PDB import PDBList, MMCIFParser  # type: ignore


def get_h_containing_pdb_ids(min_length=50, fpath=None, filter_NMR=False):
    """
    Queries RCSB PDB for specific polymer entities (chains) that meet criteria.
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query"

    if fpath and os.path.exists(fpath):
        with open(fpath, "r") as infile:
            pdb_ids = json.load(infile)
        print(f"Found file: {fpath} with {len(pdb_ids)} entries.")
        return pdb_ids

    # We search for "polymer_entity" (specific chains) rather than "entry"
    # This allows us to filter by sequence length reliably.
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                # Filter 1: Must be a Protein (Entity Level)
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein",
                    },
                },
                # Filter 2: Sequence Length > X (Entity Level)
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "greater",
                        "value": min_length,
                    },
                },
            ],
        },
        "request_options": {"return_all_hits": True},
        # CRITICAL CHANGE: Return entities (chains), not full entries
        "return_type": "polymer_entity",
    }

    if filter_NMR:
        # Filter 3: Method (Inherited from Entry Level)
        query["query"]["nodes"].append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "exact_match",
                    "value": "SOLID-STATE NMR",
                },
            }
        )

    print("Querying RCSB API...")
    response = requests.post(url, json=query)

    if response.status_code == 200:
        data = response.json()
        raw_ids = [result["identifier"] for result in data.get("result_set", [])]

        # raw_ids will look like ['1A2B_1', '1A2B_2', '3XYZ_1']
        # We need to split at '_' to get the PDB ID, and use set() to remove duplicates
        pdb_ids = list(set([x.split("_")[0] for x in raw_ids]))

        print(f"Found {len(pdb_ids)} unique PDB structures matching criteria.")
        if fpath:
            print(f"Saving to file {fpath}")
            with open(fpath, "w") as outfile:
                json.dump(pdb_ids, outfile)

        return pdb_ids
    elif response.status_code == 204:
        print("Error: API returned 204 (No matches found). Criteria may be too strict.")
        return []
    else:
        print(f"Error querying PDB: Status {response.status_code}")
        print(response.text)
        return []


def parse_and_extract_atoms(file_path):
    """Loads a .cif file, and finds lists of co-ordinates of H and C atoms"""
    parser = MMCIFParser(QUIET=True)
    try:
        structure_id = os.path.basename(file_path).split(".")[0]
        structure = parser.get_structure(structure_id, file_path)
        all_c_coords = []
        all_h_coords = []
        for model in structure:  # type: ignore
            c_coords = []
            h_coords = []
            for atom in model.get_atoms():
                element = atom.element.upper()
                coords = atom.get_coord()
                if element == "C":
                    c_coords.append(coords)
                elif element == "H":
                    h_coords.append(coords)
            all_c_coords.append(np.array(c_coords))
            all_h_coords.append(np.array(h_coords))
        return all_c_coords, all_h_coords
    except Exception as e:
        raise ValueError(f"Failed to parse {file_path}: {e}")


def download_protein_files(
    all_ids, output_dir="pdb_with_hydrogens", sample_size=None, rng=None
):
    """Pulls a set of protein files from PDB and returns filepaths"""

    if sample_size and len(all_ids) > sample_size:
        if rng is None:
            raise ValueError("Need to provide rng when sample_size =/= None.")
        print("Randomly sampling files")
        sample_ids = rng.choice(all_ids, sample_size, replace=False)
    else:
        sample_ids = all_ids

    pdbl = PDBList()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    downloaded_files = []
    for pdb_id in sample_ids:
        path = pdbl.retrieve_pdb_file(pdb_id, pdir=output_dir, file_format="mmCif")
        downloaded_files.append(path)

    return downloaded_files


def process_protein_files(downloaded_files):
    """Wrapper for parse_and_extract_atoms that runs over a list of files and prints output"""

    print("\n--- Parsing Files ---")
    data_registry = {}

    for file_path in downloaded_files:
        if not os.path.isfile(file_path):
            raise ValueError("Cannot find path: " + file_path)
        all_c_pos, all_h_pos = parse_and_extract_atoms(file_path)

        file_name = os.path.basename(file_path)
        print(f"File: {file_name}")
        print(f"  - Carbon Atoms: {len(all_c_pos[0])}")
        print(f"  - Hydrogen Atoms: {len(all_h_pos[0])}")

        if len(all_h_pos[0]) > 0:
            data_registry[file_name] = {"C": all_c_pos, "H": all_h_pos}
        else:
            raise ValueError("  ! No Hydrogens found.")
        print("-" * 30)

    return data_registry
