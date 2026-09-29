"""Continuum SWMC Hamiltonian for rhombohedral multilayer graphene.

Basis: (A1, B1, A2, B2, ..., AN, BN). Energies and U are in meV.
kx and ky are dimensionless wavevectors a*k, with a = 2.46 angstrom.
q = +1 denotes K and q = -1 denotes K'. U is the potential difference
across the stack, distributed linearly before additional on-site shifts.

Parameters, signs, indexing, and the legacy multilayer(...).Hamiltonian()
interface are retained from the supplied research code.
"""

import numpy as np

# SWMC hopping parameters and on-site shifts (meV).
gamma_0 = 3160
gamma_1 = 381
gamma_2 = -15
gamma_3 = -290
gamma_4 = 141
delta = -10.5 / 2
Delta = -2.3 / 2

# Coefficients in meV multiplying the dimensionless momentum a*k.
v = np.sqrt(3) * gamma_0 / 2
v3 = np.sqrt(3) * gamma_3 / 2
v4 = np.sqrt(3) * abs(gamma_4) / 2


class multilayer:
    """Single-valley orbital Hamiltonian for multilayers (N >= 2).

    Spin is not included. Extra keyword arguments are accepted for legacy
    compatibility with the Hartree-Fock solver.
    """

    def __init__(self, layer_number, kx, ky, U, q, **kwargs):
        self.layer_number = layer_number
        self.kx = kx
        self.ky = ky
        self.q = q
        self.U = U

    def P(self, kx, ky, q):
        """Return q*kx + i*ky using stored instance coordinates.

        The arguments are retained for compatibility; as in the original
        code, the stored attributes determine the result.
        """
        return self.q * self.kx + 1j * self.ky

    def Hamiltonian(self):
        """Return the Hermitian 2N-by-2N Hamiltonian in meV."""
        qx, qy = np.indices((2 * self.layer_number, 2 * self.layer_number))
        PI = self.P(self.kx, self.ky, self.q)
        H = np.zeros((2 * self.layer_number, 2 * self.layer_number), dtype=complex)
        i0, j0 = np.where((qy - qx == 1) & (qx % 2 == 0))  # gamma_0
        i1, j1 = np.where((qy - qx == 1) & (qx % 2 == 1))  # gamma_1
        i2, j2 = np.where((qy - qx == 5) & (qx % 2 == 0))  # gamma_2
        i3, j3 = np.where((qy - qx == 3) & (qx % 2 == 0))  # gamma_3
        i4, j4 = np.where(qy - qx == 2)                  # gamma_4

        H[i0, j0] = v * np.conj(PI)
        H[i1, j1] = gamma_1
        H[i2, j2] = gamma_2 / 2
        H[i3, j3] = v3 * PI
        H[i4, j4] = -v4 * np.conj(PI)
        H += np.conj(H.T)

        diagonal = np.linspace(self.U / 2, -self.U / 2, self.layer_number)
        diagonal[0] += Delta
        diagonal[self.layer_number - 1] += Delta
        if self.layer_number > 2:
            diagonal[1] += -Delta
            diagonal[self.layer_number - 2] += -Delta
        diagonal = np.kron(diagonal, np.array([1, 1]))
        # Retain the original gamma_2-pair endpoint on-site shifts.
        diagonal[i2] += delta
        diagonal[j2] += delta
        np.fill_diagonal(H, diagonal)
        return H
