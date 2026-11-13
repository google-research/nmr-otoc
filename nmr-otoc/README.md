# nmr-otoc

Contains files to assist in the generation of molecular Hamiltonians, the simulation of OTOCs in real NMR systems, and to generate and test different Trotterization/product formula methods.

# Contents

```text
nmr-otoc/
├─ setup.py                   Package installation script
├─ nmrotoc/                   Core Python package
│  ├─ dataclasses.py          Shared dataclasses (see also chem/molecule.py)
│  ├─ chem/                   Chemistry and PDB structure utilities
│  │  ├─ check_quality.py     Checking PDB structures for clashes and missing residues
│  │  ├─ clean_pdb.py         Filtering, conformer averaging, and fixing PDB files
│  │  ├─ hamiltonians.py      Generating spin Hamiltonians from molecular data
│  │  ├─ manipulation.py      Manipulating molecules (e.g., choosing/shifting submolecules)
│  │  ├─ molecule.py          Dataclasses for representing molecules
│  │  ├─ pdb_extraction.py    Querying RCSB PDB and extracting atomic coordinates
│  │  └─ toluene.py           Classes for representing toluene (molecule in experiment)
│  ├─ circuits/               Quantum circuit generation
│  │  ├─ compilation.py       Optimizing circuits for the device
│  │  ├─ gates.py             Quantum gate definitions
│  │  ├─ otocs.py             OTOC circuit generation
│  │  ├─ time_evolution.py    Detailed time evolution (DREAM pulses)
│  │  └─ trotter.py           Representing and generating Trotter circuits
│  ├─ math/                   Fundamental math routines
│  │  ├─ graphs.py            Graph algorithms
│  │  └─ operators.py         Linear algebra and operator generation
│  └─ scripts/                Simulation runner routines
│     └─ otocs.py             Infinite-temperature OTOC simulation runner
└─ tutorials/                 Jupyter notebook tutorials
   └─ trotter_tutorial.ipynb  Walkthrough of Trotterization methods
```

# Recommended tutorials

- tutorials/trotter_tutorial.ipynb  --- explains how Trotter methods here work
- tutorials/toluene_data_generation.ipynb --- shows how to generate data from the molecule class
