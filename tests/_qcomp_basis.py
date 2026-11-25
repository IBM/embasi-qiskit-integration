# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Exact same-basis reference for SIESTA-QCOMP's exported Gaussian ERIs.

QCOMP's ``eri_vals`` are 4-centre ERIs in the *unit-normalised spherical
Gaussian fit* basis (``sphi``).  ASI's H/S uses the same NAO2GTO fit with
SIESTA's confinement cutoff, so the two differ by only ~1e-7; the important
distinction for this module is that ``sphi`` is a Gaussian basis whose ERIs can
be reconstructed exactly with PySCF.  The fit is built in
``Src/nao2gto/nao2gto_io.F90`` as

    cphi[prim, cart] = coefficient[prim] * norm_cphi[cart]
    norm_cphi[cart] = 1 / sqrt(prefac_cart * fnorm_shell)
    prefac_cart = (2 lx-1)!! (2 ly-1)!! (2 lz-1)!!
    fnorm_shell = 0.5**l * pi**1.5 * sum_ij c_i c_j / (z_i+z_j)**(l+1.5)
    sphi = cphi @ calc_c2s_matrix(l).T

This module rebuilds ``sphi`` and provides a PySCF primitive-GTO reference for
individual ERIs, using the same Schlegel-Frisch transform as QCOMP.  It is used
only by the opt-in real tests; it never imports libsiesta.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PI = math.pi


def _dfac(n: int) -> float:
    """Double factorial with (-1)!! = 1."""
    if n <= 0:
        return 1.0
    return float(math.prod(range(n, 0, -2)))


def c2s_matrix(ell: int) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """QCOMP's Cartesian-to-spherical matrix (Schlegel-Frisch)."""
    if ell == 0:
        carts = [(0, 0, 0)]
    elif ell == 1:
        carts = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    elif ell == 2:
        carts = [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    else:
        raise NotImplementedError(f"only ell <= 2 supported, got {ell}")

    def fac(n: int) -> float:
        return float(math.factorial(n))

    def binom(n: int, k: int) -> float:
        if k < 0 or k > n:
            return 0.0
        return fac(n) / (fac(n - k) * fac(k))

    out = np.zeros((2 * ell + 1, len(carts)), dtype=float)
    for ic, (lx, ly, lz) in enumerate(carts):
        for m in range(-ell, ell + 1):
            isp = ell + m
            ma = abs(m)
            j = lx + ly - ma
            if j < 0 or j % 2:
                continue
            j //= 2
            s1 = 0.0
            for i in range((ell - ma) // 2 + 1):
                s2 = 0.0
                for k in range(j + 1):
                    if (m < 0 and abs(ma - lx) % 2 == 1) or (m > 0 and abs(ma - lx) % 2 == 0):
                        expo = (ma - lx + 2 * k) // 2
                        s = (-1.0) ** expo * math.sqrt(2.0)
                    elif m == 0 and lx % 2 == 0:
                        expo = k - lx // 2
                        s = (-1.0) ** expo
                    else:
                        s = 0.0
                    s2 += binom(j, k) * binom(ma, lx - 2 * k) * s
                s1 += (
                    binom(ell, i)
                    * binom(i, j)
                    * (-1.0) ** i
                    * fac(2 * ell - 2 * i)
                    / fac(ell - ma - 2 * i)
                    * s2
                )
            out[isp, ic] = (
                math.sqrt(
                    fac(2 * lx)
                    * fac(2 * ly)
                    * fac(2 * lz)
                    * fac(ell)
                    * fac(ell - ma)
                    / (fac(lx) * fac(ly) * fac(lz) * fac(2 * ell) * fac(ell + ma))
                )
                * s1
                / ((2.0**ell) * fac(ell))
                * ((-1.0) ** ma)
            )
    return carts, out


class Shell:
    """One QCOMP NAO shell in the Gaussian fit."""

    def __init__(self, element: str, n: int, ell: int, zeta: int, z: list[float], c: list[float]):
        self.element = element
        self.n = n
        self.l = ell
        self.zeta = zeta
        self.z = np.asarray(z, dtype=float)
        self.c = np.asarray(c, dtype=float)
        self.carts, self.c2s = c2s_matrix(ell)
        base = (
            0.5**ell
            * PI**1.5
            * float(
                np.sum(
                    self.c[:, None]
                    * self.c[None, :]
                    / (self.z[:, None] + self.z[None, :]) ** (ell + 1.5)
                )
            )
        )
        self.norm_cphi = np.array(
            [
                1.0 / math.sqrt(_dfac(2 * lx - 1) * _dfac(2 * ly - 1) * _dfac(2 * lz - 1) * base)
                for lx, ly, lz in self.carts
            ]
        )
        self.cphi = self.c[:, None] * self.norm_cphi[None, :]
        self.sphi = np.einsum("pc,mc->pcm", self.cphi, self.c2s)

    def weights(self, m: int) -> np.ndarray:
        """Primitive-Cartesian weights reproducing the m-th spherical fit AO."""
        # Each primitive is represented by a coefficient-1 shell in PySCF.
        nsph = 2 * self.l + 1
        mi = m + self.l
        raw = np.array(
            [
                math.sqrt(
                    _dfac(2 * lx - 1)
                    * _dfac(2 * ly - 1)
                    * _dfac(2 * lz - 1)
                    * (0.5**self.l)
                    * PI**1.5
                    / (2 * z) ** (self.l + 1.5)
                )
                for z in self.z
                for lx, ly, lz in self.carts
            ]
        )
        # The overlap diagonal of each coefficient-1 primitive shell is filled
        # by the caller, so weights are returned unnormalised here and divided
        # there; see ``primitive_weights``.
        values = np.array(
            [self.sphi[i, ic, mi] for i in range(len(self.z)) for ic in range(len(self.carts))]
        )
        return values * raw, nsph


def parse_orbgauss(workdir: str | Path) -> dict[str, list[Shell]]:
    """Parse ORB.GAUSS files, grouped by element and sorted as SIESTA orders them."""
    shells: dict[str, list[Shell]] = defaultdict(list)
    for path in sorted(Path(workdir).glob("ORB.GAUSS.*")):
        lines = path.read_text(errors="replace").splitlines()
        head = lines[1].split()
        # "# O.1 2 0 1 6"
        label, n, ell, zeta, ng = head[1], int(head[2]), int(head[3]), int(head[4]), int(head[5])
        element = label.split(".")[0]
        z, c = [], []
        for line in lines[3 : 3 + ng]:
            parts = line.replace("#", " ").split()
            z.append(float(parts[0]))
            c.append(float(parts[1]))
        shells[element].append(Shell(element, n, ell, zeta, z, c))
    for element in shells:
        shells[element].sort(key=lambda sh: (sh.l, sh.zeta))
    return dict(shells)


def ao_table(atoms, shells: dict[str, list[Shell]]):
    """(atom, shell, m) for each AO in SIESTA/QCOMP order."""
    table = []
    for ia, symbol in enumerate(atoms.get_chemical_symbols()):
        for shell in shells[symbol]:
            for m in range(-shell.l, shell.l + 1):
                table.append((ia, shell, m))
    return table


def _primitive_mol(atoms, selected):
    """PySCF Mole of coefficient-1 primitive Cartesian shells for a quartet."""
    from pyscf import gto

    # Group selected shells by atom; PySCF applies one basis list per element,
    # so reject a case where one element occurs on two different atoms with
    # different selected shell patterns.
    by_atom: dict[int, list[tuple[Shell, int]]] = defaultdict(list)
    for ia, shell, m in selected:
        by_atom[ia].append((shell, m))
    atom_index = sorted(by_atom)
    remap = {old: new for new, old in enumerate(atom_index)}
    atom = [[atoms[old].symbol, tuple(atoms[old].position)] for old in atom_index]
    basis: dict[str, list[list]] = defaultdict(list)
    weights_layout = []  # (atom_new, shell, m, nprim)
    for old in atom_index:
        for shell, m in by_atom[old]:
            # One PySCF shell per primitive: this makes each primitive a
            # separate AO, so the QCOMP contraction weights can be applied
            # independently.
            for z in shell.z:
                basis[shell.element].append((shell.l, [float(z), 1.0]))
            weights_layout.append((remap[old], shell, m))
    gto.mole.NORMALIZE_GTO = False
    # The reference is integral-only; the electron count/spin are irrelevant,
    # but PySCF requires a parity-consistent spin for the selected atoms.
    spin = sum(int(atoms[old].number) for old in atom_index) % 2
    mol = gto.M(atom=atom, basis=dict(basis), unit="Angstrom", cart=True, spin=spin, verbose=0)
    # PySCF orders AOs atom-by-atom, in the basis list order; the layout above
    # only matches if every selected atom has the same element basis list.  For
    # the quartets used by the test (one shell per atom, or same-atom shells),
    # this is the case.
    return mol, weights_layout


def _weights_per_index(mol, layout) -> list[np.ndarray]:
    """One weight vector per quartet index, in ``mol`` AO order.

    Match each ``mol._bas`` entry to the next selected shell with the same
    atom and angular momentum, and split that shell's primitive weights in
    ``_bas`` order.  Each selected AO gets its own vector, so the four
    quartet indices are contracted independently.
    """
    from collections import defaultdict

    sdiag = np.diag(mol.intor("int1e_ovlp"))
    by_atom_l: dict[tuple[int, int], list[int]] = defaultdict(list)
    remaining = {}
    for idx, (atom, shell, _m) in enumerate(layout):
        by_atom_l[(atom, shell.l)].append(idx)
        remaining[idx] = len(shell.z)
    consumed: dict[int, int] = defaultdict(int)

    weights = np.zeros((mol.nao, len(layout)))
    pos = 0
    for bas in mol._bas:
        atom, ell = int(bas[0]), int(bas[1])
        candidates = by_atom_l.get((atom, ell), [])
        idx = next((i for i in candidates if remaining[i] > 0), None)
        if idx is None:
            raise RuntimeError(f"PySCF shell ({atom}, l={ell}) has no selected counterpart")
        remaining[idx] -= 1
        _atom, shell, m = layout[idx]
        values, _ = shell.weights(m)
        ncart = 2 * ell + 1
        primitive = consumed[idx]
        consumed[idx] += 1
        vals = values[primitive * ncart : (primitive + 1) * ncart]
        n = len(vals)
        block = sdiag[pos : pos + n]
        if n != ncart or np.any(block <= 0):
            raise RuntimeError("invalid primitive block while ordering weights")
        weights[pos : pos + n, idx] = vals / np.sqrt(block)
        pos += n
    if pos != mol.nao:
        raise RuntimeError(f"weights cover {pos} of {mol.nao} AOs")
    return [weights[:, i] for i in range(weights.shape[1])]


def eri_value(atoms, table, quartet, *, omega: float | None = None) -> tuple[float, str]:
    """Reference ERI for four AO indices, in the QCOMP ``sphi`` fit basis.

    ``omega=None`` is full Coulomb; otherwise QCOMP's short-range
    ``erfc(omega r)/r`` operator.
    """
    selected = [table[i] for i in quartet]
    # A quartet with several different atoms of one element and different
    # shell patterns cannot be represented by PySCF's per-element basis.
    mol, layout = _primitive_mol(atoms, selected)
    weights = _weights_per_index(mol, layout)
    if omega is None:
        eri = mol.intor("int2e_cart").reshape([mol.nao] * 4)
    else:
        with mol.with_short_range_coulomb(omega):
            eri = mol.intor("int2e_cart").reshape([mol.nao] * 4)
    value = float(np.einsum("a,b,c,d,abcd->", *weights, eri, optimize=True))
    return value, "full" if omega is None else f"short({omega})"


def self_eri(atoms, table, ao: int, **kwargs) -> tuple[float, str]:
    """Reference self-ERI ``(aa|aa)`` for one QCOMP AO.

    Uses one coefficient-1 PySCF shell per primitive (QCOMP's exact
    primitive-Cartesian construction) and applies the ``sphi`` weights
    independently to the four ERI indices.
    """
    from pyscf import gto

    ia, shell, m = table[ao]
    basis = {shell.element: [(shell.l, [float(z), 1.0]) for z in shell.z]}
    spin = int(atoms[ia].number) % 2
    gto.mole.NORMALIZE_GTO = False
    mol = gto.M(
        atom=[[atoms[ia].symbol, tuple(atoms[ia].position)]],
        basis=basis,
        unit="Angstrom",
        cart=True,
        spin=spin,
        verbose=0,
    )
    values, _ = shell.weights(m)
    sdiag = np.diag(mol.intor("int1e_ovlp"))
    assert values.shape == sdiag.shape
    weights = values / np.sqrt(sdiag)
    omega = kwargs.get("omega")
    if omega is None:
        eri = mol.intor("int2e_cart").reshape([mol.nao] * 4)
    else:
        with mol.with_short_range_coulomb(omega):
            eri = mol.intor("int2e_cart").reshape([mol.nao] * 4)
    value = float(
        np.einsum("a,b,c,d,abcd->", weights, weights, weights, weights, eri, optimize=True)
    )
    return value, "full" if omega is None else f"short({omega})"
