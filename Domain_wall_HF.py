"""
Periodic-x Hartree-Fock solver for rhombohedral graphene.

Place multilayer_graphene.py beside this file. Basis ordering is
(spin, valley, G, layer/sublattice): (K up, K' up, K down, K' down).
Spin and valley observables use Pauli matrices.

Input units inherited from the cluster implementation:
ne: signed carrier number density in cm^-2 (negative for holes);
U, spin_orbit: meV; beta: meV^-1; g_perp: meV cm^2; ke: meV cm;
d_gate: cm; L: length in lattice units a (converted to int).
a = 2.46e-8 cm. The q=0 Hartree term is excluded.

Interface: Domain_Wall(...).Mean_field_State()
returns, in order: observables; averaged energy in meV/nm; the dimensionless
trace-one local spin-valley order matrix of Eq. (4), shape (nx,4,4);
energy history; residual history; chemical potential; eigenvalues; and
eigenvectors.
Observables have shape (4,4,nx): [0,0] is positive
carrier density in cm^-2; other entries are normalized by this density.

The diagonalization and density-matrix construction over ky can run in
parallel on multiple CPU cores through joblib shared-memory workers. Set
the environment variable SLURM_INNER_JOBS to the desired worker count;
the default is one worker. When using Slurm, request at least that many
CPUs per task and avoid assigning more workers than allocated CPU cores.

The scalar energy includes the bulk contribution. For the periodic cell
with two equivalent walls, subtract the matching uniform-run scalar
to obtain excess energy per wall length. Both runs must use identical
geometry, density, cutoff, and other physical parameters.
The scalar averages the last 20 domain-wall or 5 uniform iteration energies;
it need not equal the energy of the final rediagonalized state.

After a run, x_a and x_nm give the sample coordinates.
local_order_matrix has shape (nx,4,4), traces out layer/sublattice,
and is normalized by its local trace. It is dimensionless and trace one
where local carrier density is positive. It need not be idempotent.
local_carrier_density_cm2 stores positive carrier number density separately.
No electron-charge factor is applied.
convergence records stopping reason and whether residual tolerance was met.

Carrier occupations are abs(f-f_ref), with f_ref=1 in the lower half
of sorted bands and 0 above, preserving the original carrier convention.

Scientific kernels, cutoff, FFT convention, mixing, and stopping rules are
retained. Small-grid checks do not establish production convergence.
"""

import numpy as np
from scipy import optimize
from numpy.fft import fftn, ifftn, ifftshift
import multilayer_graphene as mlg
import time
from scipy.linalg import block_diag
import os
import psutil
from joblib import Parallel, delayed
D = np.float64

'''Sigma matrices'''
sigma_0 = np.array([[1, 0], [0, 1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
SU_2 = np.array([sigma_0, sigma_x, sigma_y, sigma_z], dtype=complex)

a = D(2.46e-8)  # cm


class RAMTracker:
    def __init__(self):
        self.max_gb = 0.0

    def update(self):
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        self.max_gb = max(self.max_gb, rss)
        return rss

    def report(self):
        print(f"PEAK RAM used: {self.max_gb:.2f} GB")
def num_workers():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


def fermi_dirac_stable(E, mu, beta):
    # Returns float64 with D = np.float64.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        x = (D(beta) * (E.astype(D, copy=False) - D(mu))).astype(D, copy=False)
        out = np.empty_like(x, dtype=D)
        mask = x > D(0)
        # stable branches
        out[mask] = np.exp(-x[mask]).astype(D, copy=False) / (D(1) + np.exp(-x[mask]).astype(D, copy=False))
        out[~mask] = D(1) / (D(1) + np.exp(x[~mask]).astype(D, copy=False))
        return out
    
def _diag_worker(Hk):
    import numpy as np
    return np.linalg.eigh(Hk)

def _rho_worker(args):
    import numpy as np
    evecs, occ = args
    return (evecs * occ[None, :]) @ np.conjugate(evecs).T
class Domain_Wall(mlg.multilayer):

    def __init__(self, layer_number, ne, U, beta, spin_orbit,g_perp, L, er, d_gate,
                 max_count, tol, mix, ng, nk_y, ke,seed_rho,
                 domain_seed, vx, vz, sx, sz, **kwargs):

        self.layer_number = int(layer_number)
        self.ne = ne * (a**2)
        self.U = U
        self.beta = beta
        self.spin_orbit = spin_orbit
        self.g_perp=g_perp/a**2 #unit of meV
        self.L = int(L)
        self.er = er
        self.d_gate = d_gate / a
        self.max_count = int(max_count)
        self.tol = tol
        self.mix = mix

        self.vx = vx
        self.vz = vz
        self.sx = sx
        self.sz = sz

        self.ng = int(ng)
        self.nk_y = int(nk_y)

        self.G_index = np.arange(-self.ng // 2 + 1,
                                 self.ng // 2 + 1)

        self.ke = ke / a
        self.seed_rho=seed_rho
        self.domain_seed = bool(domain_seed)
        self.L2 = 2 * self.layer_number
        self.N = 4 * self.ng * self.L2

        # ================= OPERATORS =================

        eye_big = np.eye(2 * self.ng * self.L2)

        self.Sx = np.einsum('ab,cd->acbd', sigma_x, eye_big).reshape(self.N, self.N)
        self.Sy = np.einsum('ab,cd->acbd', sigma_y, eye_big).reshape(self.N, self.N)
        self.Sz = np.einsum('ab,cd->acbd', sigma_z, eye_big).reshape(self.N, self.N)

        self.Vx = np.einsum('ab,cd,ef->acebdf',
                            np.eye(2), sigma_x,
                            np.eye(self.ng * self.L2)).reshape(self.N, self.N)

        self.Vy = np.einsum('ab,cd,ef->acebdf',
                            np.eye(2), sigma_y,
                            np.eye(self.ng * self.L2)).reshape(self.N, self.N)

        self.Vz = np.einsum('ab,cd,ef->acebdf',
                            np.eye(2), sigma_z,
                            np.eye(self.ng * self.L2)).reshape(self.N, self.N)

        lx = np.zeros((self.L2, self.L2))
        ly = np.zeros((self.L2, self.L2),dtype=complex)
        lz = np.zeros((self.L2, self.L2))

        lx[0, -1] = lx[-1, 0] = 1
        ly[0, -1] = -1j
        ly[-1, 0] = 1j
        lz[0, 0] = -1
        lz[-1, -1] = 1

        self.Lx = np.einsum('ab,cd->acbd',
                            np.eye(4 * self.ng), lx).reshape(self.N, self.N)
        self.Ly = np.einsum('ab,cd->acbd',
                            np.eye(4 * self.ng), ly).reshape(self.N, self.N)
        self.Lz = np.einsum('ab,cd->acbd',
                            np.eye(4 * self.ng), lz).reshape(self.N, self.N)
        
        
        # ================= RECTANGULAR GRID =================

        self.G0 = 2 * np.pi / self.L

        self.ky_1d = np.linspace(-self.G0*self.ng/2,self.G0*self.ng/2, self.nk_y)

        self.dky = abs(self.ky_1d[1] - self.ky_1d[0])

        self.A = self.L*(2*np.pi)/self.dky

        # Reciprocal-lattice harmonics along the domain-wall direction
        self.qx_flat = self.G_index * self.G0

        self.ky = self.ky_1d

        self.px, self.py = np.meshgrid(self.qx_flat,
                                       self.ky_1d,
                                       indexing='xy')
        assert self.px.shape == (self.nk_y, self.ng)

        self.V_q = self.Coulomb(self.px, self.py)

        self._V_r = fftn(self.V_q, axes=(0,)) / self.A
        G = self.G_index
        Gmin = G[0]
        Gj_idx = (G[None,:] - G[:,None]) - Gmin
        valid = (Gj_idx >= 0) & (Gj_idx < self.ng)
        self._Gj_idx = np.where(valid, Gj_idx, 0)
        self._valid = valid
        self._valid_g_Gi_Gj = [
                            (g, Gi, self._Gj_idx[g, Gi])
                            for g in range(self.ng)
                            for Gi in range(self.ng)
                            if self._valid[g, Gi]
                        ]
        # ---------- Precompute Hartree-Fock operators ----------
        self._hf_d_vals = np.arange(-(self.ng - 1), self.ng)
        '''exclude q=0 Hartree component'''
        self._hf_Vh = np.zeros(self._hf_d_vals.shape)  
        nz = self._hf_d_vals != 0
        self._hf_Vh[nz] = self.Coulomb(self._hf_d_vals[nz] * self.G0, 0.0) / self.A
        '''include q=0 component'''
        # self._hf_Vh = self.Coulomb(self._hf_d_vals * self.G0, 0.0) / self.A

        idx = np.arange(self.ng)
        self._hf_Hartree_offset_index = (
            (idx[:, None] - idx[None, :]) + (self.ng - 1)
        )
        # ---------- Precompute Hund operators ----------

        self._hund_tau_ops = [
            np.kron(sigma_0, sigma_x),
            np.kron(sigma_0, sigma_y),
        ]

        self._hund_d_data = []
        for d in range(-(self.ng - 1), self.ng):
            if d >= 0:
                rows = np.arange(0, self.ng - d)
                cols = rows + d
            else:
                rows = np.arange(-d, self.ng)
                cols = rows + d

            self._hund_d_data.append((d, rows, cols, len(rows)))

        self._hund_O_list = []
        for sigma in range(self.L2):
            for tau_op in self._hund_tau_ops:
                O = np.zeros((4, self.L2, 4, self.L2), dtype=complex)
                O[:, sigma, :, sigma] = tau_op
                self._hund_O_list.append(O)


        # ================= PRINT ONCE =================

        qx_max = self.px.max()
        ky_max = self.py.max()
        area_cm2 = self.A * a**2
        self.rs=1/np.sqrt(np.pi*abs(self.ne))
        print("--------------------------------------------------")
        print(f"qx_max = {qx_max}")
        print(f"ky_max = {ky_max}")
        print(f"area = {area_cm2} cm^2, total charge = {self.ne * self.A:.4f}")
        print(f"total k-points = {self.nk_y}")
        print(f"magnetic field B_x=OFF (Lag. mul)")
        print('average distance between electrons (r_s)=',self.rs,'(a)=',self.rs*a*1e7,'(nm)')
        print("--------------------------------------------------", flush=True)
    def Block_Hamiltonian(self, ky):
        H = [block_diag(*[
                mlg.multilayer(self.layer_number,
                               self.G_index[i] * self.G0,
                               ky,
                               self.U,
                               (-1) ** j).Hamiltonian()
                for i in range(self.ng)
            ]) for j in range(2)]
        out = block_diag(*(H * 2)) # Spin-major order: (K up, K' up, K down, K' down).
        return np.asarray(out)


    def Seed_Hamiltonian(self):
        '''Square wave function'''
        def z_kernel(n):
            return 2 * np.sin(n * np.pi / 2) / (np.pi * n + 1e-15)

        n_GG = (self.G_index[:, None] - self.G_index[None, :])
        z_GG = z_kernel(n_GG)
        '''Double Gausian function'''
        def x_kernel(n):
            sd=self.rs #standard deviation of Gaussian peaks
            x0=self.L/4 #+- location of Gausian peak
            kn=2*np.pi*n/self.L
            cn=2*sd*np.sqrt(2*np.pi)*np.exp(-sd**2*kn**2/2)*np.cos(kn*x0)/(1+np.exp(-2*x0**2/sd**2))/self.L
            # cn=2j*sd*np.sqrt(2*np.pi)*np.exp(-sd**2*kn**2/2)*np.sin(kn*x0)/(1-np.exp(-x0**2/2/sd**2))/self.L
            return cn
        x_GG = x_kernel(n_GG)

        valley_z = np.einsum('sS,vV,gG,lL->svglSVGL',
                             np.eye(2), sigma_z,
                             z_GG,
                             np.eye(self.L2),
                             optimize=True).reshape(self.N, self.N)
        valley_x = np.einsum('sS,vV,gG,lL->svglSVGL',
                        np.eye(2), sigma_x,
                        x_GG,
                        np.eye(self.L2),
                        optimize=True).reshape(self.N, self.N)

        spin_z = np.einsum('sS,vV,gG,lL->svglSVGL',
                           sigma_z, np.eye(2),
                           z_GG,
                           np.eye(self.L2),
                           optimize=True).reshape(self.N, self.N)
        spin_x = np.einsum('sS,vV,gG,lL->svglSVGL',
                        sigma_x, np.eye(2), 
                        x_GG,
                        np.eye(self.L2),
                        optimize=True).reshape(self.N, self.N)
        spin_valley_x = np.einsum('sS,vV,gG,lL->svglSVGL',
                                sigma_x, sigma_x, 
                                x_GG,
                                np.eye(self.L2),
                                optimize=True).reshape(self.N, self.N)
        spin_valley_y = np.einsum('sS,vV,gG,lL->svglSVGL',
                        sigma_y, sigma_y, 
                        x_GG,
                        np.eye(self.L2),
                        optimize=True).reshape(self.N, self.N)
        if self.domain_seed:
            return valley_z + spin_z+spin_valley_x-spin_valley_y+1e-1*(valley_x + spin_x)
        else:
            return (- self.vx * self.Vx - self.vz * self.Vz
                    - self.sx * self.Sx - self.sz * self.Sz)

    def _diag_one_k(self,Hk):
        # Hk: (N,N) complex hermitian
        evals, evecs = np.linalg.eigh(Hk)
        return evals, evecs

    def nonint_H_string(self, which: int):
        if which == 0:
            return np.stack([self.Block_Hamiltonian(ky) for ky in self.ky])
        if which == 1:
            return np.stack([self.Seed_Hamiltonian() for _ in self.ky])
        raise ValueError("which must be 0 (band) or 1 (seed)")


    # ---------- THERMO ----------

    def Fermi_Dirac(self, E, mu):
        return fermi_dirac_stable(E, mu, self.beta)


    def Charge_distribution(self, E, mu):
        nf = fermi_dirac_stable(E, mu, self.beta)
        nf1, nf2 = np.split(nf, 2, axis=-1)
        nf1 = nf1 - 1
        return np.concatenate((nf1, nf2), axis=-1)


    def mu_solver(self, E):

        Ne = self.ne * self.A

        def N_of_mu(mu):
            return float(self.Fermi_Dirac(E, mu).sum())

        target = float(self.N * self.nk_y / 2 + Ne)

        # return optimize.bisect(lambda mu: N_of_mu(mu) - target,
        #                        -500.0, 500.0, xtol=1e-5, maxiter=200)
        return optimize.bisect(lambda mu: N_of_mu(mu) - target,
                                       -5e2, 1e5, xtol=1e-5, maxiter=200)


    # ---------- DENSITY ----------

    def density_matrix(self, Psi, distribution):
        Psi_f = Psi * distribution[:, None, :]
        rho = Psi_f @ np.conjugate(Psi).transpose(0, 2, 1)
        return rho

    def _rho_one_k(self,evecs, occ):
        # evecs: (N,N), occ: (N,)
        # rho = (evecs * occ) @ evecs^\dagger
        Psi_f = evecs * occ[None, :]
        return Psi_f @ np.conjugate(evecs).T
    # ---------- COULOMB ----------

    def Coulomb(self, qx, qy):
        k = np.sqrt(qx**2 + qy**2)
        '''includes q=0'''
        return 2*np.pi*self.ke*np.tanh((k + 1e-15)*self.d_gate)/ (self.er*(k + 1e-15))
        '''excludes q=0'''
        # return 2*np.pi*self.ke*np.tanh(k *self.d_gate)/(self.er*(k + 1e-15))
    ''' ---------- HARTREE–FOCK ----------'''

    def Hartree_Fock(self, rho):

        IJ, N, _ = rho.shape  # IJ = nk_y
        ng = self.ng
        L2 = self.L2

        rho_rs = rho.reshape(IJ, 4, ng, L2, 4, ng, L2)

        # ---------------- Hartree (unchanged) ----------------
        n_GG = np.einsum('kcglcGl->gG', rho_rs, optimize=True)
        diag_sums = np.array([
            np.diagonal(n_GG, offset=-int(d)).sum()
            for d in self._hf_d_vals
        ])

        H_off = self._hf_Vh * diag_sums
        Hartree_GG = H_off[self._hf_Hartree_offset_index]

        H_H_block = np.einsum(
            'ab,cd,ef->acebdf',
            np.eye(4),
            Hartree_GG,
            np.eye(L2),
            optimize=True
        ).reshape(N, N)

    # ---------- Fock ----------

        M = 4 * self.L2

        nk_y = self.nk_y
        ng = self.ng

        # reorder to (ky, Gi, Gj, M, M)
        rho_t = rho_rs.transpose(0, 2, 5, 1, 3, 4, 6)
        rho_t = rho_t.reshape(nk_y, ng, ng, M, M)
        del rho_rs

        # build rho_p_G: (g, ky, Gi, M, M)
        rho_p_G = np.zeros((ng, nk_y, ng, M, M), dtype=rho_t.dtype)

        for g, Gi, Gj in self._valid_g_Gi_Gj:
            rho_p_G[g, :, Gi] = rho_t[:, Gi, Gj]

        # FFT only along ky. The G-axis convolution is done explicitly.
        rho_r_G = fftn(rho_p_G, axes=(1,))
        V_r = self._V_r  # shape (nk_y, ng)

        Fock_r = np.zeros_like(rho_r_G)
        for qG in range(ng):
            Fock_r -= (V_r[None, :, qG, None, None, None]
                       * np.roll(rho_r_G, shift=qG, axis=2))

        Fock_p_G = ifftshift(ifftn(Fock_r, axes=(1,)), axes=(1, 2))
        del rho_r_G, Fock_r

        # unfold back
        H_F = np.zeros((nk_y, 4, ng, L2, 4, ng, L2), dtype=Fock_p_G.dtype)

        for g, Gi, Gj in self._valid_g_Gi_Gj:
            block = Fock_p_G[g, :, Gi]
            block = block.reshape(nk_y, 4, L2, 4, L2)
            H_F[:, :, Gi, :, :, Gj, :] += block
        del Fock_p_G
        H_F = H_F.reshape(IJ, N, N)

        return H_H_block, H_F
    ''' ---------- Intervalley Hund's Coupling ----------'''

    def Hund_DW(self, rho):
        r"""
        Domain-wall intervalley Hund Hartree-Fock self-energy.

        Implements

            Sigma_H =
            + g_perp/(4A) sum_{G_r,sigma,i}
            Tr[tau_i P_sigma T_G_r^\dagger rho]
            tau_i P_sigma T_G_r

            Sigma_F =
            - g_perp/(4A) sum_{G_r,sigma,i}
            tau_i P_sigma T_G_r^\dagger rho tau_i P_sigma T_G_r

        Basis assumed:
            rho shape = (IJ, N, N)
            N = 4 * ng * L2
            internal ordering = (flavor, G, L2)
        """
        IJ, N, _ = rho.shape

        ng = self.ng

        L2 = self.L2

        pref = self.g_perp / (4.0 * self.A)

        rho_tot = np.sum(rho, axis=0)

        rho_rs = rho_tot.reshape(4, ng, L2, 4, ng, L2)

        H_H = np.zeros((4, ng, L2, 4, ng, L2), dtype=complex)

        H_F = np.zeros_like(H_H)

        for sigma in range(L2):

            for tau_op in self._hund_tau_ops:

                for _, rows, cols, _ in self._hund_d_data:

                    # Hartree: Tr[tau_i P_sigma T_d^\dagger rho]

                    rho_pair = rho_rs[:, rows, sigma, :, cols, sigma]

                    # shape: (R, 4, 4), indices R, b, a

                    coeff = np.einsum(
                        "ab,Rba->",tau_op,rho_pair,
                        optimize=True,)

                    H_H[:, rows, sigma, :, cols, sigma] += coeff * tau_op

                    # Fock:

                    # - tau_i rho(G,G+Gr) tau_i

                    block = np.einsum("ac,Rcd,db->ab",
                                      tau_op,rho_pair,
                        tau_op,optimize=True,)

                    H_F[:, rows, sigma, :, cols, sigma] -= block

        H_H = (pref * H_H).reshape(N, N)

        H_F = (pref * H_F).reshape(N, N)

        return H_H, H_F    
        
        # ---------- SCHF ----------

    def reduced_density_matrix(self, rho, x_a, chunk_size=256):
        """Trace orbital indices and Fourier transform to (nx,4,4), in cm^-2.

        rho is (nk_y,N,N) and must include the desired occupation weights.
        Coordinates are in lattice units. This traces both layer and
        sublattice; no further area division should be applied.
        """
        x_a = np.atleast_1d(np.asarray(x_a, dtype=float))
        if x_a.ndim != 1 or chunk_size < 1:
            raise ValueError("x_a must be one-dimensional and chunk_size positive")
        rho = np.asarray(rho)
        if rho.shape != (self.nk_y, self.N, self.N):
            raise ValueError("rho must have shape (nk_y,N,N)")
        orbital_trace = np.einsum(
            "kfglhGl->fghG",
            rho.reshape(self.nk_y, 4, self.ng, self.L2,
                        4, self.ng, self.L2), optimize=True)
        difference = self.G_index[:, None] - self.G_index[None, :]
        result = np.empty((x_a.size, 4, 4), dtype=complex)
        for start in range(0, x_a.size, chunk_size):
            stop = min(start + chunk_size, x_a.size)
            phase = np.exp(1j * self.G0 * difference[:, :, None]
                           * x_a[None, None, start:stop])
            result[start:stop] = np.einsum(
                "fghG,gGx->xfh", orbital_trace, phase, optimize=True)
        return result / (self.A * a**2)

    def Mean_field_State(self):
        if y not in (0, 1):
            raise ValueError("y must be 0 or 1")
        if self.max_count < 1:
            raise ValueError("max_count must be positive")

        ram = RAMTracker()
        n_jobs = int(os.environ.get("SLURM_INNER_JOBS", "1"))
        print(f"Using n_jobs = {n_jobs}", flush=True)

        parallel = Parallel(
            n_jobs=n_jobs,
            require="sharedmem",
        )

        def diagonalize_k_parallel(H_mf):
            out = parallel(
                delayed(_diag_worker)(H_mf[i])
                for i in range(H_mf.shape[0])
            )
            evals_all = np.stack([x[0] for x in out], axis=0)
            evecs_all = np.stack([x[1] for x in out], axis=0)
            return evals_all, evecs_all

        def build_rho_parallel(evecs_all, occ_all):
            rho_list = parallel(
                delayed(_rho_worker)((evecs_all[i], occ_all[i]))
                for i in range(evecs_all.shape[0])
            )
            return np.stack(rho_list, axis=0)

        H0 = self.nonint_H_string(which=0)
        '''Kane-Mele SOC term'''
        H_soc= -(self.spin_orbit/2) * (np.kron(np.eye(4*self.ng*self.layer_number),sigma_z) @ self.Vz @ self.Sz)
        IJ = H0.shape[0]
        assert H0.shape == (self.nk_y, self.N, self.N)
        
        if self.seed_rho is not None:
            rho_old=self.seed_rho
        else:
            H1=H0.copy()
            H1+=self.Seed_Hamiltonian()
            evals, evecs = diagonalize_k_parallel(H1)
            mu_old = self.mu_solver(evals)
            occ_old = self.Fermi_Dirac(evals, mu_old)
            rho_old = build_rho_parallel(evecs, occ_old)
            del evals, evecs
        E_old = np.einsum('iab,iba->', H0, rho_old).real / abs(float(self.A * self.ne))

        err = 3.14
        it = 0
        window = 30
        delta = 1e-4
        Hartree_E_list = []
        E_list=[]
        error_list = []

        vz_list=[]
        sx_dw_list=[]
        x = np.linspace(-int(self.L // 2), int(self.L // 2), 4*int(self.L)+1)

        phase = np.exp(1j * (2 * np.pi / self.L) *
                    (np.outer(self.G_index[:, None] -
                                self.G_index[None, :], x))
                    ).reshape(self.ng, self.ng, x.size)
        
        
        def local_order_from_eigensystem(eval_mf, evec_mf, mu_new, xval):
            phase_x = np.exp(
                1j * (2 * np.pi / self.L)
                * (self.G_index[:, None] - self.G_index[None, :])
                * xval
            )

            rho_hole_x = build_rho_parallel(
                evec_mf,
                abs(self.Charge_distribution(eval_mf, mu_new))).sum(axis=0).reshape(
                2, 2, self.ng, self.L2,
                2, 2, self.ng, self.L2)

            Op = (np.einsum(
                    'svglSVGl,nSs,mVv,gG->nm',
                    rho_hole_x, SU_2, SU_2, phase_x,
                    optimize=True).real / self.A)
            nx=Op[0,0].copy()
            norm = abs(Op[0, 0])
            if norm > 1e-14:
                Op_norm = Op / norm
            else:
                Op_norm = Op
            Op_norm[0,0]=nx
            return Op_norm
        stop_reason = "max_iterations"
        last_residual = None
        attempted_iterations = 0
        t0=time.time()
        while (err > self.tol) and (it < self.max_count):

            attempted_iterations += 1
            ram.update()

            H_H, H_F = self.Hartree_Fock(rho_old)
            H_perp,F_perp=self.Hund_DW(rho_old)
            H_mf = H0 + H_F
            H_mf += H_H+H_soc #Lagrange multiplier
            H_mf +=H_perp+F_perp
            eval_mf, evec_mf = diagonalize_k_parallel(H_mf)
            # np.fill_diagonal(H_H,0)
            del H_mf

            mu_new = self.mu_solver(eval_mf)
            occ_new=self.Fermi_Dirac(eval_mf, mu_new)
            rho_new = build_rho_parallel(evec_mf, occ_new)
            E_H = 0.5 * np.einsum('ab,iba->', H_H, rho_new,
                                optimize=True).real

            Hartree_E_list.append(float(E_H*(self.L/2)/self.A/(a*1e7))) #Line energy in meV/nm

            E_F = 0.5 * np.einsum('iab,iba->', H_F, rho_new,
                                optimize=True).real
            E_perp=0.5 * np.einsum('ab,iba->', H_perp+F_perp, rho_new,
                                optimize=True).real
            E_soc = np.einsum('ab,iba->', H_soc, rho_new,
                                optimize=True).real
            E_new = (np.einsum('iab,iba->', H0, rho_new).real
                        +E_H+ E_F+ E_perp+E_soc)*(self.L/2)/self.A/(a*1e7) #Line energy in meV/nm
            del H_F, H_H, H_perp,F_perp
            vz = np.einsum('kab,ba->',
                                rho_new,self.Vz,
                                optimize=True).real \
                        / abs(float(self.A * self.ne))
            vz_list.append(vz)
            if self.domain_seed:
                if abs(vz) > 1e-10:
                    print('Diverging vz at iteration', it, 'vz=', vz)
                    stop_reason = "valley_imbalance"
                    break

            err_rho = np.linalg.norm(rho_new - rho_old) / (IJ * self.N)
            last_residual = float(err_rho)
            err=err_rho
            E_list.append(E_new)
            error_list.append(float(err_rho))
            rho_old *= 1 - self.mix
            rho_old += self.mix * rho_new
            E_old = E_new
            mu_old = mu_new
            del rho_new    
            it += 1
            if it % 10 == 0:
                Op_dw = local_order_from_eigensystem(
                    eval_mf, evec_mf, mu_new, self.L / 4)

                Op_zero = local_order_from_eigensystem(
                    eval_mf, evec_mf, mu_new, 0.0)

                vx_dw = Op_dw[0, 1]
                sx_dw = Op_dw[1, 0]
                nx_dw=Op_dw[0, 0]/a**2/1e11
                sx_dw_list.append(sx_dw)

                vz_zero = Op_zero[0, 3]
                sz_zero = Op_zero[3, 0]
                nx_zero = Op_zero[0, 0]/a**2/1e11

                print(
                    f"Iter {it}: E = {E_new:.6f} meV/nm, "
                    f"err_rho = {err_rho:.2e}, global vz = {vz:.2e}\n"
                    f"  Hartree energy = {Hartree_E_list[-1]:.6f} meV/nm"
                    f"  Fock energy = {E_F*(self.L/2)/self.A/(a*1e7):.6f} meV/nm\n"
                    f"  At x = +L/4: vx = {vx_dw:.6e}, sx = {sx_dw:.6e}, ne ={nx_dw:.4e}*1e11\n"
                    f"  At x = 0:    vz = {vz_zero:.6e}, sz = {sz_zero:.6e}, ne ={nx_zero:.4e}*1e11\n",
                    flush=True)
                ram.update()
            if it == 1:
                print(f"time for each iteration= {(time.time() - t0)/60:.2f} min", flush=True)
            if it>window:
                self.mix=0.4
                # if abs(sx_dw_list[-1]-sx_dw_list[-2])<1e-4:
                #     break
                recent = np.array(error_list[-window:])
                avg_err = np.mean(recent)
                rel_diff = abs(err_rho - avg_err) / max(abs(avg_err), 1e-13)
                if (rel_diff < delta) and (abs(sx_dw_list[-1]-sx_dw_list[-2])<1e-4):
                    stop_reason = "stagnation"
                    print(f"Stopping at iteration {it}: current error is within {delta*100:.2f}% of window average.")
                    print(f"sx change between last 10 itetations={abs(sx_dw_list[-1]-sx_dw_list[-2])}")
                    break
        converged = (stop_reason != "valley_imbalance"
                     and last_residual is not None
                     and np.isfinite(last_residual)
                     and last_residual <= self.tol)
        if converged:
            stop_reason = "residual_tolerance"
        elif last_residual is not None and not np.isfinite(last_residual):
            stop_reason = "nonfinite_residual"
        self.convergence = {
            "converged": bool(converged),
            "stop_reason": stop_reason,
            "iterations": attempted_iterations,
            "residual": last_residual,
            "tolerance": float(self.tol),
        }
        if not converged:
            print(f"WARNING: residual convergence not established ({stop_reason}).",
                  flush=True)
        # Average the final 20 line-energy values
        if self.domain_seed:
            n_energy_average = 20
        else:
            n_energy_average = 5
        final_energies = np.asarray(
            E_list[-n_energy_average:],
            dtype=float)
        final_line_energy = (np.mean(final_energies) if final_energies.size
                             else float("nan"))
        line_energy_std = (np.std(final_energies) if final_energies.size
                           else float("nan"))
        self.energy_average_count = int(final_energies.size)
        self.energy_std_meV_per_nm = float(line_energy_std)
        print(
            f"Final line energy = {final_line_energy:.6f} meV/nm\n"
            f"Standard deviation = {line_energy_std:.6f} meV/nm\n"
            f"Number of energies averaged = {len(final_energies)}",
            flush=True
        )
        H_H, H_F = self.Hartree_Fock(rho_old)
        H_perp,F_perp=self.Hund_DW(rho_old)
        H_mf = H0 + H_F
        H_mf += H_H+H_soc
        H_mf +=H_perp+F_perp
        eval_mf, evec_mf = diagonalize_k_parallel(H_mf)
        mu_new = self.mu_solver(eval_mf)
        self.x_a = x.copy()
        self.x_nm = x * a * 1e7
        carrier_rho = build_rho_parallel(
            evec_mf, np.abs(self.Charge_distribution(eval_mf, mu_new)))
        rho_hole = build_rho_parallel(
            evec_mf, np.abs(self.Charge_distribution(eval_mf, mu_new))
        ).sum(axis=0).reshape(
            2, 2, self.ng, self.L2, 2, 2, self.ng, self.L2)

        # Fourier transform and trace BOTH layer and sublattice indices.
        # Q_x is density weighted; division by its local trace gives Eq. (4) in maintext.
        Q_x = np.einsum(
            'svglSVGl,gGx->xsvSV', rho_hole, phase, optimize=True
        ).reshape(x.size, 4, 4) / self.A
        n_x = np.trace(Q_x, axis1=1, axis2=2).real
        self.local_order_matrix = Q_x / n_x[:, None, None]
        self.local_carrier_density_cm2 = n_x / a**2

        # Preserve the existing Pauli observables and their array layout.
        Op_x = np.einsum(
            'svglSVGl,nSs,mVv,gGx->nmx',
            rho_hole, SU_2, SU_2, phase, optimize=True
        ).real / self.A
        Op_x /= abs(Op_x[0, 0])
        Op_x[0, 0] = self.local_carrier_density_cm2

        return Op_x, final_line_energy, self.local_order_matrix, E_list, error_list, mu_new, eval_mf, evec_mf
