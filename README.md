<div align="center">

# NMR-OTOC

End-to-end simulation of OTOCs in solid-state NMR spin systems from protein
structures.

[Features](#features) –
[Installation](#installation) –
[Quick start](#quick-start) –
[Documentation](#documentation) –
[Help](#help) –
[Citation](#citation) –
[Contact](#contact)

</div>

NMR-OTOC is a collection of Python libraries and command-line tools (with a
C++-accelerated simulation backend) built for researchers in quantum computing,
quantum simulation, and solid-state [nuclear magnetic resonance][nmr] (NMR)
spectroscopy. It implements an end-to-end pipeline from macromolecular
structures in [Protein Data Bank][pdb] ([PDB/mmCIF][rcsb]) format to effective
[spin Hamiltonians][spin-model] and [out-of-time-ordered correlators][otoc]
(OTOCs)—quantities that, much like a physical interferometer, measure how
quantum information scrambles and spreads across complex many-body correlations
over time.

## Features

*   PDB-to-spin-Hamiltonian pipeline (`nmr-otoc` and `nmr-otoc-tools`): parse
    PDB and mmCIF structures with [Biopython], average coordinates across
    structural conformers, select carbon-13 or fluorine-19 system spin pairs,
    order and truncate surrounding proton spins, and build static dipolar or
    [Magic Angle Spinning][mas] (MAS) recoupling Hamiltonians over Lebedev and
    SO(3) powder-averaging grids.

*   Exact Trotterized OTOC simulation (`nmr-otoc` and `nmr-otoc-tools`):
    construct first- and second-order Trotter circuits—either dense
    all-to-all circuits or 1D linear swap networks—in [Cirq], compute
    infinite-temperature OTOCs via Haar-random state averaging in `qsimcirq`,
    and verify atom mappings and distances (`nmr-map-atoms`,
    `nmr-compute-distances`).

## Installation

### Prerequisites

*   Python 3.12 or later is required

*   [`nmr-otoc/setup.py`](nmr-otoc/setup.py) does not declare runtime
    dependencies in `install_requires`; modules under `nmrotoc/` import
    `numpy`, `scipy`, `cirq`, `qsimcirq`, `openfermion`, `biopython`,
    `matplotlib`, `tqdm`, and `requests` (plus `openmm` and `pdbfixer` in
    [`nmrotoc/chem/clean_pdb.py`](nmr-otoc/nmrotoc/chem/clean_pdb.py)).

### Install from source

1.  Clone the repository and create a Python virtual environment:

    ```bash
    git clone https://github.com/google-research/nmr-otoc.git
    cd nmr-otoc
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  Install the core pipeline packages (`nmrotoc` and `nmr-otoc-tools`):

    ```bash
    pip install openfermion
    pip install -e ./nmr-otoc
    pip install -e ./nmr-otoc-tools
    ```

    If you plan to run the tutorial notebooks or PDB cleaning utilities, also
    install `matplotlib tqdm requests` (and `openmm pdbfixer` for
    `clean_pdb.py`).

3.  Verify an installation by running a package test suite (for example,
    `nmr-otoc-tools`):

    ```bash
    pip install -e "./nmr-otoc-tools[testing]"
    (cd nmr-otoc-tools && pytest)
    ```

## Quick start

Run an immediate 4-spin OTOC simulation using the bundled configuration file in
`nmr-otoc-tools` (no external PDB download required):

```bash
mkdir -p /tmp/otoc-out
nmr-run-from-config \
    --config nmr-otoc-tools/integration_tests/artificial/random/sim_config_N4.json \
    --output_dir /tmp/otoc-out
```

This writes `/tmp/otoc-out/otoc_N4_t80_h50.json` with the computed OTOC values
at each time step.

To run the full pipeline on a protein structure, place a file at
`protein_files/<PDB_ID>/<pdb_id>_out.cif` (lowercase ID, or pass `--filename`),
generate dipolar coupling instances on a 26-point Lebedev grid, and simulate
system sizes from 4 to 8 spins:

```bash
nmr-generate-instances <PDB_ID> -26
nmr-run-otoc-simulation \
    --pdb_id <PDB_ID>_L26 \
    --subfolder d_5 \
    --n_avg 26 \
    --n_min 4 \
    --n_max 8 \
    --max_time 2.0 \
    --n_time 8 \
    --n_trotter 20 \
    --n_haar 20 \
    --circuit_type dense
```

## Documentation

Each subpackage provides its own documentation, examples, or tutorials:

*   Core library (`nmrotoc`) and Jupyter tutorials:
    [`nmr-otoc/README.md`](nmr-otoc/README.md),
    [`trotter_tutorial.ipynb`](nmr-otoc/tutorials/trotter_tutorial.ipynb), and
    [`toluene_data_generation.ipynb`][toluene-nb].

*   Instance generation, Magic Angle Spinning physics, and CLI reference:
    [`nmr-otoc-tools/README.md`](nmr-otoc-tools/README.md).

*   Project governance, support, and BSD 3-Clause Clear license:
    [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SUPPORT.md`](SUPPORT.md), and
    [`LICENSE`](LICENSE).

## Help

If you encounter a bug or have a question about using this software, please
search existing issues or open a new issue on the
[GitHub issue tracker](https://github.com/google-research/nmr-otoc/issues).
Additional guidance is available in [`SUPPORT.md`](SUPPORT.md), or you can
reach the maintainers at `quantum-oss-maintainers@google.com`.

We are committed to providing a friendly, safe, and welcoming environment for
all. Please read and respect our [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

This repository accompanies a forthcoming OTOC-NMR theory publication and
companion datasets to be archived on Zenodo. Citation metadata
(`CITATION.cff` and publication bibliographic details) will be added once the
paper is published. In the meantime, please cite the repository at
<https://github.com/google-research/nmr-otoc>, and if you use the Lebedev
quadrature grids via `nmr-otoc-tools`, please also cite the Lebedev and Laikov
reference listed in [`nmr-otoc-tools/README.md`][tools-citation].

## Contact

For any questions or concerns not addressed here, please email
quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

Copyright 2026 Google LLC.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="./docs/images/quantum-ai-vertical.svg">
  </a>
</div>

[Biopython]: https://biopython.org/
[Cirq]: https://quantumai.google/cirq
[mas]: https://en.wikipedia.org/wiki/Magic_angle_spinning
[nmr]: https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance
[otoc]: https://en.wikipedia.org/wiki/Out-of-time-ordered_correlator
[pdb]: https://en.wikipedia.org/wiki/Protein_Data_Bank
[qsim]: https://quantumai.google/qsim
[rcsb]: https://www.rcsb.org/
[Spinach]: https://spindynamics.org/
[spin-model]: https://en.wikipedia.org/wiki/Spin_model
[toluene-nb]: nmr-otoc/tutorials/toluene_data_generation.ipynb
[tools-citation]: nmr-otoc-tools/README.md#citation
