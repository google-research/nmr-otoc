# NMR OTOC Tools

This library simulates Out-of-Time-Order Correlators (OTOCs) in solid-state nuclear magnetic resonance (NMR) systems using `cirq` and `qsimcirq`. It focuses on mixed spin systems (e.g., Carbon-13 and Protons) governed by dipolar Hamiltonians.

The spins are partitioned into a **system** (typically 2 Carbon spins) and an **environment** (multiple Proton spins), and the library simulates how information scrambles from the system into the environment.

## Table of Contents
1. [The Simulated Quantity (OTOC)](#the-simulated-quantity-otoc)
2. [Physical System & Hamiltonian](#physical-system--hamiltonian)
   - [Homogeneous Coupling (DQ Term)](#homogeneous-coupling-dq-term)
   - [Heterogeneous Coupling (Secular Term)](#heterogeneous-coupling-secular-term)
3. [Circuit Implementation](#circuit-implementation)
   - [Dense Circuit](#dense-circuit-circuittypedense)
   - [Swap Network Circuit](#swap-network-circuit-circuittypeswap_network)
4. [Library Structure](#library-structure)
5. [Magic Angle Spinning (MAS)](#magic-angle-spinning-mas)
6. [Usage](#usage)
   - [Installation](#installation)
   - [Generating Instances](#generating-instances)
   - [Running Simulations](#running-simulations)
   - [Config-Based Workflow](#config-based-workflow)
   - [Mapping Spins to Atoms and Measuring Distances](#mapping-spins-to-atoms-and-measuring-distances)
7. [Citation](#citation)

---

## The Simulated Quantity (OTOC)

The library computes the **infinite-temperature OTOC** between two operators acting on the system spins:

$$O(t) = \frac{1}{2^N} \text{Tr} \left( X_1(t) X_0 X_1(t) X_0 \right)$$

where:
*   $N$ is the total number of spins (system + environment).
*   $X_0$ is the Pauli-X operator acting on the **measurement spin** (logical index 0, a Carbon spin).
*   $X_1$ is the Pauli-X operator acting on the **butterfly spin** (logical index 1, another Carbon spin).
*   $X_1(t) = e^{iHt} X_1 e^{-iHt}$ is the Heisenberg-evolved butterfly operator at time $t$.
*   $H$ is the total system Hamiltonian (including system-environment couplings).

$O(t)$ is estimated by averaging the circuit's final $\langle Z_0 \rangle$ over `--n_haar` initial states, each with qubit 0 in $|0\rangle$ and the other $N-1$ qubits Haar-random; the noise shrinks as `--n_haar` grows.

---

## Physical System & Hamiltonian

The simulated physical system represents a solid-state NMR setup, typically featuring two species of spins (e.g., $^{13}\text{C}$ as system spins and $^1\text{H}$ as environment spins).

The Hamiltonian is partitioned into homogeneous and heterogeneous parts, scaled by parameters $\eta_{\text{hom}}$ and $\eta_{\text{het}}$:

$$H = \eta_{\text{hom}} H_{\text{hom}} + \eta_{\text{het}} H_{\text{het}}$$

### Homogeneous Coupling (DQ Term)
Homogeneous couplings occur between spins of the same species (Carbon-Carbon and Proton-Proton). In the Double Quantum (DQ) Hamiltonian frame (often used in NMR scrambling experiments like DREAM), this interaction takes the form:

$$H_{\text{hom}} = \sum_{i \lt j \text{ (same species)}} d_{ij} (Y_i Y_j - X_i X_j)$$

where $d_{ij}$ is the dipolar coupling constant between spins $i$ and $j$.

### Heterogeneous Coupling (Secular Term)
Heterogeneous couplings occur between spins of different species (Carbon-Proton). Due to the large difference in Larmor frequencies between different species in a strong magnetic field, this interaction is secularized and takes the form:

$$H_{\text{het}} = \sum_{p \in \text{Carbons}, i \in \text{Protons}} d_{pi} Z_p Z_i$$

where $d_{pi}$ is the heteronuclear dipolar coupling.

---

## Circuit Implementation

The library provides two methods for constructing the Trotterized simulation circuits:

### Dense Circuit (`CircuitType.DENSE`)
Implemented in `make_otoc_circuit2`, this method constructs the circuit by directly applying two-qubit gates ($XX$, $YY$, and $ZZ$) between all interacting pairs of the system–environment partition. It assumes all-to-all connectivity.
*   The $Y_i Y_j - X_i X_j$ terms are Trotterized as $e^{-i d dt YY} e^{i d dt XX}$.
*   The $Z_p Z_i$ terms are Trotterized as $e^{-i d dt ZZ}$.

### Swap Network Circuit (`CircuitType.SWAP_NETWORK`)
Implemented in `make_otoc_circuit` (utilizing the underlying `nmrotoc` library), this method compiles the Trotter steps into a **swap network**. This is designed for architectures with 1D linear nearest-neighbor connectivity. By systematically swapping adjacent qubits, it allows all-to-all interactions to be implemented on a linear chain.

---

## Library Structure

The core logic lives in `nmr_otoc_tools/simulation.py` (circuits and simulators), `mas.py` (MAS couplings), `orientations.py` (powder-averaging grids) and `serialization.py` (config file format). Each `nmr-*` command in [Usage](#usage) is implemented in the module of the same name (e.g. `nmr-map-atoms` in `map_atoms.py`).

---

## Magic Angle Spinning (MAS)

`nmr-generate-instances --coupling mas` replaces the static secular coupling with the
first-order average Hamiltonian of a recoupling sequence under MAS.

### Physics

For every pair the five rank-2 spatial components are computed once in the molecular frame,
$A^M_{2m'} = d^2_{0m'}(\theta)e^{-im'\phi}$ (normalized to $A_{20}^{PAS} = 1$), rotated into
the rotor frame with a Wigner matrix, and contracted with the MAS mechanical scaling and the
RF scaling $\eta_{m\mu}$:

$$A^R_{2m} = \sum_{m'} A^M_{2m'} D^2_{m'm}(\Omega),\qquad
d_{ij} = b_{ij} \sum_m \eta_m\thinspace d^2_{m0}(\beta_{\text{rotor}})\thinspace A^R_{2m}$$

with $b_{ij}$ the usual dipolar constant. Homonuclear (double-quantum) couplings stay
**complex**; heteronuclear ($Z_pZ_i$) couplings are **real** because that spin operator is
Hermitian ($\eta_{-m} = \eta_m^*$ is enforced). The spin-operator convention is

$$H_{\text{hom}} = \sum_{i \lt j}\big[\mathrm{Re}(d_{ij})(Y_iY_j - X_iX_j) + \mathrm{Im}(d_{ij})(X_iY_j + Y_iX_j)\big]$$

so the phase of $d_{ij}$ simply rotates the DQ operator about $z$ — exactly the rotor-phase
(γ-encoding) factor. With $\eta_0 = 1$ and a zero rotor angle the model reduces *exactly* to
the static `dipolar` coupling.

A recoupling sequence matches a **single** spatial Fourier component to the spin operator; the
default is the γ-encoded $m = -1$ component, giving $|d_{ij}| = |b_{ij}\sin 2\beta_{ij}|/(2\sqrt2)$
(twice the HORROR/DREAM $\kappa = 1/{(4\sqrt2)}$) with phase $\gamma_{ij}$. Instances carry unit
weight on the selected component(s); sequence-specific scale factors are applied at simulation
time with `--eta_hom` / `--eta_het`, so the same instances can be reused for different sequences.

### Powder average over SO(3)

Because the rotor phase matters, the powder average runs over $SO(3)$, built as the tensor
product of a Lebedev grid with a uniform twist grid. The twist is taken
about the **rotor axis**, so the Lebedev point *is* the rotor axis in the molecular frame and
the grid degenerates to the plain Lebedev grid in the static limit. Inversion pruning
(`--use-symmetry`) is **not** valid on this grid and is rejected.

With a Lebedev × twist grid, run the simulation with `--n_avg` equal to the number of
orientations: a truncated grid is not a valid quadrature. With random orientations any prefix is
an unbiased estimate.

### Generating MAS Instances

```bash
# 26-point Lebedev x 8 twists = 208 orientations
nmr-generate-instances <PDB_ID> -26 --coupling mas --n-twist 8
```

| Flag | Meaning |
|---|---|
| `--n-twist` | Number of uniform rotor twists (default 8). Check convergence by doubling it. |
| `--rotor-angle-deg` | Rotor axis / B0 angle (default: the magic angle). |
| `--mas-dq-order` | \|m\| recoupled to the homonuclear DQ term (0 disables, default 1). |
| `--mas-zz-order` | \|m\| recoupled to the heteronuclear ZZ term (0 disables, default 1). |

### MAS Outputs

In addition to the [instance files](#generating-instances):

*   `dipolar_matrix{i}.npy` — **complex128** symmetric coupling matrix for orientation `i`.
*   `orientation{i}.npy`, `orientations.npy`, `euler_angles.npy` — the molecular→rotor rotations.
*   `field_direction{i}.npy` — the rotor axis in the reference frame (static-mode analogue).
*   `weights.npy` — $SO(3)$ powder weights, one per `dipolar_matrix{k}.npy`, summing to one:
    $w_i/N_T$ for Lebedev point $i$ and twist $j$ ($k = iN_T + j$, $N_T$ = `--n-twist`), or
    $1/N$ for random orientations.
*   `metadata.json` — gains `coupling_format: "complex"` and a `mas` block recording the grid,
    the recoupled components (unit $\eta$ weights), the rotor angle and the storage convention.

### Simulating MAS Instances

```bash
nmr-run-otoc-simulation --pdb_id <PDB_ID>_L26 --subfolder d_5 --n_avg 208 --n_min 4 --n_max 8 --circuit_type dense [other options]
```

---

## Usage

### Installation

Install the `nmrotoc` dependency from `../nmr-otoc`, then this package, and run the tests:

```bash
pip install -e ../nmr-otoc
pip install -e ".[testing]"
pytest
```

### Generating Instances

To generate Hamiltonian instances from a PDB/CIF structure:

```bash
nmr-generate-instances <PDB_ID> <instance_id> [other options]
```

*   The structure is read from `protein_files/<PDB_ID>/<pdb_id>_out.cif` (lower-case ID); pass
    `--filename` for another file in that folder. All models (conformers) are averaged over.
*   `<instance_id>` selects the powder sampling: `N > 0` draws `N` random field directions
    (random orientations with `--coupling mas`), and `-L` uses the `L`-point Lebedev grid
    (e.g. `-26`; `--use-symmetry` keeps one direction of every inversion-symmetric pair).
*   The two system spins are a random carbon pair about 5 Å apart (fixed seed), or are given
    with `--cidx1` / `--cidx2` as global 0-based atom indices of model 0 or as
    `chain:resname:resnum:atomlabel` specifiers. `--fluorine` uses fluorines instead.
*   `--max-size N` keeps only the first `N` spins of the sorted system.
*   `--coupling` selects the coupling model (`dipolar` or [`mas`](#magic-angle-spinning-mas));
    `--use-unrotated-field` interprets the field directions in the frame of the structure file
    instead of the molecular frame of the system pair.

The instances are written to `data/generated_instances/<PDB_ID>_<instance_id>/<output_name>/`
(`<PDB_ID>_L<L>` for a Lebedev grid), where `<output_name>` is `--output_name`, by default
`d_5` for a random pair or `inds_<cidx1>_<cidx2>` for a given pair. For every field direction
(or orientation) `i`:

*   `dipolar_matrix{i}.npy` — the conformer-averaged coupling matrix.
*   `ordered_indices{i}.npy` — the spin order used by the simulations (system pair first).
*   `original_indices{i}.npy` — only if `--max-size` pruned the system: the kept spins'
    positions in the unpruned system, which `nmr-map-atoms` uses.
*   `field_direction{i}.npy` — the field direction.

plus `weights.npy` (the powder weights, one per instance) and `metadata.json` (pair indices,
distance and options). `metadata.json` is written last; a folder that has one is skipped.

### Running Simulations

To run the OTOC simulation on the generated instances:

```bash
nmr-run-otoc-simulation --pdb_id <PDB_ID>_<instance_id> --subfolder <output_name> --n_avg <num_instances> --n_min <min_spins> --n_max <max_spins> --max_time <max_time> --n_time <time_steps> --n_haar <num_haar_samples> --circuit_type <swap_network|dense>
```

Key arguments:
*   `--pdb_id` / `--subfolder`: The instance folder `data/generated_instances/<pdb_id>/<subfolder>/`.
*   `--n_avg`: Number of instances (field directions or orientations) to simulate.
*   `--n_min` / `--n_max`: Range of system sizes (number of spins) to simulate.
*   `--max_time` / `--n_time`: Maximum simulation time and number of time points (the grid
    `max_time / n_time, 2 max_time / n_time, ..., max_time`).
*   `--n_trotter`: Number of Trotter steps per time interval.
*   `--n_haar`: Number of Haar random states to average over (higher means less noise).
*   `--circuit_type`: `dense` (all-to-all) or `swap_network` (linear connectivity).
*   `--eta_hom` / `--eta_het`: Scale factors of the homogeneous and heterogeneous couplings.
*   `--n_proc`: Number of parallel processes to run.
*   `--n_thread`: Number of threads for `qsim` per process.
*   `--save_circuit`: Also write every OTOC circuit as Cirq JSON.

Results are saved as `otoc_N<n>_t<n_trotter>_h<n_haar>.json` files in the instance folder.

The powder average weights each instance with `weights.npy`:

```python
res = json.loads(Path("otoc_N4_t20_h20.json").read_text())
weights = np.load("weights.npy")[: len(res["otocs"])]
powder_average = np.average(np.array(res["otocs"]), axis=0, weights=weights)
```

### Config-Based Workflow

The simulation can be split into a setup step and runs that only need the config file:

```bash
# Write sim_config_N<n>.json for every n in [n_min, n_max] (same arguments as above).
nmr-setup-simulation --pdb_id <PDB_ID>_<instance_id> --subfolder <output_name> --n_avg 20 --n_min 4 --n_max 6
CONFIG=data/generated_instances/<PDB_ID>_<instance_id>/<output_name>/sim_config_N4.json

nmr-run-from-config --config $CONFIG                   # simulate one config
nmr-run-core-from-config --config $CONFIG --n-core 3   # n_core core spins + each other spin
nmr-circuits-from-config --config $CONFIG              # export the circuits as Cirq JSON
```

The runners write their results to the instance folder stored in the config (relative to the
directory in which `nmr-setup-simulation` was run), or to `--output_dir` if given.

### Mapping Spins to Atoms and Measuring Distances

```bash
# Atoms of every spin of instance 0, in simulation order.
nmr-map-atoms data/generated_instances/<PDB_ID>_<instance_id>/<output_name>/ordered_indices0.npy protein_files/<PDB_ID>/<file> -o atoms.json --verify
# Distance statistics between two atoms over all models.
nmr-compute-distances protein_files/<PDB_ID>/<file> A:ALA:10:CA A:ALA:12:CB
```

`nmr-map-atoms` reads the instance's `metadata.json` and reports the chain, residue, atom name,
serial number and global index of every spin (`--max-size` truncates the list). `--verify`
recomputes the coupling matrix and checks it against `dipolar_matrix{i}.npy`.
`nmr-compute-distances` accepts the same atom indices and specifiers as `--cidx1` / `--cidx2`.

---

## Citation

The Lebedev grids come from PySCF's `pyscf.dft.LebedevGrid` (Apache License 2.0), whose authors
ask users to cite V. I. Lebedev and D. N. Laikov, "A quadrature formula for the sphere of the
131st algebraic order of accuracy", *Doklady Mathematics* **59**, 477–481 (1999).
