# Multicomponent Magnetic Domain Walls in Rhombohedral Graphene

This repository contains the numerical code used to study multicomponent magnetic domain walls in rhombohedral graphene. It includes a microscopic Hartree–Fock calculation and a complementary CP³ continuum calculation.

The repository is being prepared to accompany a Nature Communications manuscript. Production calculations are computationally expensive, and the code should be treated as research software.

## Repository contents

| File | Purpose |
| --- | --- |
| `multilayer_graphene.py` | Constructs the noninteracting multilayer-graphene Hamiltonian from Slonczewski–Weiss–McClure parameters. |
| `Domain_wall_HF.py` | Solves the periodic domain-wall Hartree–Fock problem and evaluates local charge and spin–valley observables. |
| `DW_run.py` | Defines the microscopic parameters and performs a descending continuation scan in `g_perp`. |
| `CP3_domain_wall.ipynb` | Minimizes the CP³ free energy, compares ascending and descending continuation scans, and produces the six-panel continuum figure. |
| `requirements.txt` | Python dependencies. |
| `CITATION.cff` | Software citation metadata. |
| `LICENSE` | MIT License. |

## Installation

Python 3.11 or newer is recommended. Create an isolated environment and install the dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The Hartree–Fock calculation can require substantial memory. Test the code with smaller grids before running the production parameters.

## Microscopic Hartree–Fock calculation

The driver contains the physical parameters, numerical grid, convergence controls, and output settings:

```sh
python DW_run.py
```

The default calculation:

- uses `ne = -0.09e12 cm^-2`;
- uses `soc = 0.1 meV`, while the Hamiltonian coefficient is `soc/2 = 0.05 meV`;
- starts at `g_perp = 1400e-14 meV cm^2`;
- scans downward through ten values ending at `500e-14 meV cm^2`;
- uses the converged occupation density matrix at one coupling to seed the next calculation;
- initializes a domain wall when `DOMAIN_SEED = True`.

Set `DOMAIN_SEED = False` to calculate the matching uniform state. Domain-wall and uniform calculations used in an energy comparison must have identical density, geometry, cutoff, and physical parameters.

### Parallel execution

Independent operations over `k_y` use shared-memory workers. Set the worker count with:

```sh
SLURM_INNER_JOBS=4 python DW_run.py
```

On a cluster, request at least the same number of CPUs per task. Do not set more workers than the allocated CPU cores.

### Output location

Results are written to:

```text
$SCRATCH/domain_wall/
```

If `SCRATCH` is not defined, they are written to a `domain_wall/` directory beside the driver.

For each coupling, the driver saves:

- Pauli observables;
- the local spin–valley order matrix;
- the energy and residual histories;
- the chemical potential;
- eigenvalues;
- compressed eigenvectors in HDF5 format;
- the reported Hartree–Fock energy.

## Hartree–Fock conventions

The retained spin–valley basis is

```text
(K up, K' up, K down, K' down).
```

`Mean_field_State()` returns:

```python
(
    Op_x,
    final_line_energy,
    local_order_matrix,
    E_list,
    error_list,
    mu_new,
    eval_mf,
    evec_mf,
)
```

Important conventions:

- `Op_x` has shape `(4, 4, nx)`.
- `Op_x[0, 0]` is the positive carrier number density in `cm^-2`.
- No electron-charge factor is applied.
- `local_order_matrix` has shape `(nx, 4, 4)`.
- It traces out layer and sublattice and is normalized to unit trace where the local carrier density is nonzero.
- The reported scalar energy includes the bulk contribution.
- The periodic cell contains two equivalent walls. Obtain the excess line energy by subtracting the matching uniform result and applying the wall-count convention used in the manuscript.
- The current Coulomb implementation excludes the `q = 0` Hartree term.
- Check `convergence` and the residual history before treating a result as converged.

## CP³ continuum calculation

Open and run:

```text
CP3_domain_wall.ipynb
```

The notebook uses the four-component basis

```text
(up,K), (up,K'), (down,K), (down,K').
```

It minimizes the normalized CP³ spinor with fixed `(down,K')` and `(up,K)` boundary states.

The Hund-coupling scan uses 11 points:

```python
u_perp_over_lambda = np.linspace(0.0, 3.0, 11)
```

The notebook:

1. performs ascending and descending continuation scans;
2. uses each converged state to seed the next coupling within that scan;
3. compares the two energies at every coupling;
4. retains the lower-energy solution;
5. chooses the symmetry-equivalent orientation with positive dominant `s_x` and `tau_x`;
6. evaluates spin, valley, composite spin–valley, and projector-normalization observables;
7. plots the two-component solution at zero coupling and the four-component solution at `u_perp/lambda_soc = 2.1`;
8. defines the grid-resolved critical coupling as the last sampled point where both wall-center transverse order parameters remain below the numerical threshold.

The plotted transverse quantities are magnitudes, `|s_x|` and `|tau_x|`. Domain-wall widths are obtained by interpolating the two positions where the normalized longitudinal profile reaches `±tanh(1)`.

Because the critical coupling is selected from a discrete scan, its reported value depends on the coupling grid and the numerical threshold. Refine the grid before quoting a higher-precision transition point.

The bidirectional scan is needed to identify the true lowest-energy domain-wall solution. Near the critical coupling, a unidirectional continuation scan can remain on a saddle-point or metastable branch instead of finding the minimum-energy texture.

## Reproducibility notes

Before using these calculations for a figure or numerical comparison:

- record the exact commit or release;
- preserve the random seed and complete input parameters;
- verify convergence against grid size, reciprocal cutoff, system size, mixing, and tolerance;
- for a domain-wall Hartree–Fock run, keep `mix < 0.3` through the first 100 iterations so that the texture develops smoothly; the supplied driver starts with `mix = 0.2`, and the solver does not raise it until after iteration 100;
- monitor the printed spin–valley order parameters together with the density-matrix residual to verify convergence to the intended domain-wall texture rather than relying on the residual alone;
- use both ascending and descending CP³ continuation scans near the critical coupling, since a unidirectional scan can converge to a saddle-point or metastable solution;
- run the matching uniform Hartree–Fock calculation;
- retain the energy and residual histories;
- record the Python and dependency versions;
- do not commit scratch data, credentials, or cluster-specific paths.

## Citation and archival release

When using this code, please cite the accompanying paper:

> Mainak Das, Nemin Wei, and Chunli Huang, “Multicomponent Magnetic Domain Walls in Rhombohedral Graphene,” arXiv:2608.26104 (2026). [https://doi.org/10.48550/arXiv.2608.26104](https://doi.org/10.48550/arXiv.2608.26104)

The paper is available at [arXiv:2608.26104](https://arxiv.org/abs/2608.26104). Software citation metadata are provided in `CITATION.cff`.

For publication, create a tagged GitHub release and archive that exact release in a DOI-minting repository such as Zenodo. Add the resulting software DOI to `CITATION.cff` and the manuscript’s Code Availability statement.

## License

This project is distributed under the MIT License. See `LICENSE`.
