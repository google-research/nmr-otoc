# nmr-otoc

Contains files to assist in the generation of molecular Hamiltonians, the simulation of OTOCs in real NMR systems, and to generate and test different Trotterization / product formula methods.

# Contents

├── dataclasses.py        Dataclasses used in the rest of the code (see also molecule)  
├── chem                  Functions pertaining to chemistry (pre-quantum computing)  
│   ├── hamiltonians.py   Generating spin Hamiltonians from molecular data  
│   ├── manipulation.py   Manipulating molecules (eg choosing and shifting submolecules)  
│   ├── molecule.py       Dataclasses for representing molecules  
│   └── toluene.py        Classes for representing toluene (molecule in experiment)  
├── circuits              Functions pertaining to circuit generation  
│   ├── compilation.py    Functions for optimizing circuits for the device  
│   ├── gates.py          Quantum gate definitions  
│   ├── misc.py           Other circuit-related functions (e.g. prep circuits)  
│   ├── noise.py          Functions to generate noisy circuits  
│   ├── otocs.py          Functions for OTOC circuit generation  
│   ├── pauli_echoes.py   Functions to add mid-circuit echoes to a FSim circuit  
│   ├── time_evolution.py Functions for more detailed time evolution (DREAM pulses)  
│   ├── trotter_opt.py    Functions for optimizing Trotter circuits  
│   └── trotter.py        Functions for representing and generating Trotter circuits  
├── math                  Fundamental math routines  
│   ├── graphs.py         Graph algorithms  
│   ├── operators.py      Linear algebra and other operator generation work  
└── scripts               Full scripts for running code  
&nbsp;&nbsp;&nbsp;&nbsp;├── mqc.py            Multiple quantum coherence scripts  
&nbsp;&nbsp;&nbsp;&nbsp;├── otocs.py          OTOC-related scripts (see also run_scripts)  
&nbsp;&nbsp;&nbsp;&nbsp;├── run_scripts.py    General-purpose scripts  

# Recommended tutorials

- tutorials/trotter_tutorial.ipynb  --- explains how Trotter methods here work
- tutorials/toluene_data_generation.ipynb --- shows how to generate data from the molecule class
