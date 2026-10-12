"""
Run a descending-g_perp Hartree-Fock continuation scan.

The first calculation starts at g_perp = 1400e-14 meV cm^2. Each following
calculation uses the converged occupation density matrix from the preceding
g_perp as its initial density matrix. Set DOMAIN_SEED to False to obtain the
corresponding uniform solutions. Independent ky work is parallelized over
SLURM_INNER_JOBS shared-memory workers.
"""

import os
import time

import h5py
import numpy as np

import Domain_wall_HF as dwHF


output_root = os.environ.get("SCRATCH", os.path.dirname(__file__))
OUTDIR = os.path.join(output_root, "domain_wall")
os.makedirs(OUTDIR, exist_ok=True)

# Momentum-space grid: ng reciprocal harmonics along x and nk_y ky points.
nk_y = 21
ng = 175

# Physical and numerical inputs.
a = 2.46e-8                 # graphene lattice constant in cm
layer_number = 2
ne = -0.09e12              # cm^-2
# Density-dependent periodic supercell length in units of a.
L = 0.8 *200/ np.sqrt(np.pi * abs(ne)) / a #(Periodic length 'L' is set at 80% of 200 inter-particle distance)
U = 25                     # meV
beta = 50                  # meV^-1
soc = 0.1                  # meV
vx, vz, sx, sz = 0, 2, 0, 2
er = 15
ke = 1.44e-4               # Coulomb constant in meV cm
d_gate = 50e-7             # distance from gate to material in cm
max_count = 800
tol = 1e-9
DOMAIN_SEED = True
# To obtain the uniform solution, change DOMAIN_SEED to False.
mix = 0.2 if DOMAIN_SEED else 0.7

# Descending order is required for continuation from the g_perp=1400 solution.
g_perp_values = np.linspace(500, 1400, 10)[::-1] * 1e-14  # meV cm^2

phase = "domain_wall" if DOMAIN_SEED else "uniform"

print("nk_y:", nk_y, flush=True)
print("ng:", ng, flush=True)
print("Number of (G, ky) points:", ng * nk_y, flush=True)
print("L (a):", L, flush=True)
print("Saving to:", OUTDIR, flush=True)
print("Phase initialization:", phase, flush=True)
print(f"soc: {soc}", flush=True)
print("g_perp scan (1e-14 meV cm^2):", g_perp_values / 1e-14, flush=True)

seed_rho = None

for scan_index, g_perp in enumerate(g_perp_values):
    print("--------------------------------------------------", flush=True)
    print(f"Scan {scan_index + 1}/{len(g_perp_values)}", flush=True)
    print(f"g_perp: {g_perp:.6e} meV cm^2", flush=True)
    print(
        "Initial density matrix:",
        "domain/uniform seed" if seed_rho is None else "previous g_perp solution",
        flush=True,
    )

    start = time.time()

    dw = dwHF.Domain_Wall(
        layer_number=layer_number,
        ne=ne,
        U=U,
        beta=beta,
        spin_orbit=soc,
        g_perp=g_perp,
        L=L,
        er=er,
        d_gate=d_gate,
        max_count=max_count,
        tol=tol,
        mix=mix,
        ng=ng,
        nk_y=nk_y,
        ke=ke,
        seed_rho=seed_rho,
        domain_seed=DOMAIN_SEED,
        vx=vx,
        vz=vz,
        sx=sx,
        sz=sz,
    )

    (
        Op_x,
        final_line_energy,
        local_order_matrix,
        E_list,
        error_list,
        mu_new,
        eval_mf,
        evec_mf,
    ) = dw.Mean_field_State()

    # Use the final occupied-state density matrix to initialize the next,
    # lower-g_perp calculation.
    occupations = dw.Fermi_Dirac(eval_mf, mu_new)
    seed_rho = dw.density_matrix(evec_mf, occupations)

    runtime = (time.time() - start) / 60.0
    print(f"Elapsed time: {runtime:.3f} minutes", flush=True)

    tag = (
        f"{phase}_soc{soc:.3f}_gperp{g_perp / 1e-14:.0f}"
        .replace(".", "p")
    )

    np.save(os.path.join(OUTDIR, f"Op_x_{tag}.npy"), Op_x)
    np.save(
        os.path.join(OUTDIR, f"local_order_matrix_{tag}.npy"),
        local_order_matrix,
    )
    np.save(
        os.path.join(OUTDIR, f"line_energy_{tag}.npy"),
        np.array(final_line_energy),
    )
    np.save(os.path.join(OUTDIR, f"E_list_{tag}.npy"), np.asarray(E_list))
    np.save(
        os.path.join(OUTDIR, f"error_list_{tag}.npy"),
        np.asarray(error_list),
    )
    np.save(os.path.join(OUTDIR, f"mu_{tag}.npy"), np.array(mu_new))
    np.save(os.path.join(OUTDIR, f"eigenvalues_{tag}.npy"), eval_mf)

    fname = os.path.join(OUTDIR, f"eigenvectors_{tag}.h5")
    print("Saving eigenvectors to:", fname, flush=True)
    print("evec_mf shape:", evec_mf.shape, flush=True)
    print("evec_mf dtype:", evec_mf.dtype, flush=True)
    print("evec_mf raw size GB:", evec_mf.nbytes / 1024**3, flush=True)

    with h5py.File(fname, "w") as f:
        dset = f.create_dataset(
            "evec_mf",
            shape=evec_mf.shape,
            dtype=np.complex128,
            chunks=(1, evec_mf.shape[1], 32),
            compression="gzip",
            compression_opts=4,
        )
        for ik in range(evec_mf.shape[0]):
            dset[ik] = evec_mf[ik].astype(np.complex128, copy=False)

    print("Finished saving:", tag, flush=True)
