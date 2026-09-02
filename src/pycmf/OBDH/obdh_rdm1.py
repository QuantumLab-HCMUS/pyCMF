"""
Unrelaxed correlated 1-RDM for UOBMP2 / UOBDH in pyCMF.

--------------------------------------------------------------------------
THE THREE DENSITIES
--------------------------------------------------------------------------
Write the 1-RDM in the MO basis of the converged OBMP2 orbitals and split
the MO index set into occupied (o) and virtual (v).  Every density in this
story is the same matrix with different blocks filled in:

                       oo block       vv block      ov block
  determinant          I              0             0
  unrelaxed            I + d_oo       d_vv          0
  relaxed  = dE/dh     I + d_oo + D_oo   d_vv + D_vv   z

  d_oo[i,j] = -sum T*_imef T_jmef      (negative: occupied loses charge)
  d_vv[a,b] = +sum T*_mnae T_mnbe      (positive: virtual gains charge)
  Tr(d_oo) = -Tr(d_vv), so N is conserved.

  z          comes from the Z-vector equation, i.e. from dE/dkappa
             (orbital response).
  D_oo, D_vv come from dE/dT (amplitude response).

This module computes the UNRELAXED density -- the middle row.  It is the
unrestricted counterpart of `pycmf/OBDH/dft_obmp2.py:562`, which only
handles closed-shell systems, and it reproduces the recipe stated in
Tran, PCCP 24, 19393 (2022):

    "At the convergence, the OBMP2 electron density is evaluated using
     the T2 amplitude (eqn (4)) as in standard MP2."

--------------------------------------------------------------------------
WHY UNRELAXED IS NOT THE DIPOLE
--------------------------------------------------------------------------
mu = -dE/dF requires the RELAXED density.  Two response terms separate the
unrelaxed density from it:

  * z (ov block)        <- nonzero whenever dE/dkappa != 0
  * D_oo, D_vv          <- nonzero whenever dE/dT != 0

For standard MP2 the amplitudes ARE the stationary point of the Hylleraas
functional, so dE/dT = 0 and D_oo = D_vv = 0 exactly.  That is why the
textbook statement "relaxed = unrelaxed + Z-vector" holds for MP2: only
the ov block is missing.

For OBMP2 it does NOT hold.  Eqn (4) builds the amplitudes with
denominators taken from eigenvalues of the CORRELATED Fock Fbar = f + v,
not from f, so the amplitudes sit off the Hylleraas stationary point and
dE/dT != 0.  Measured on CN / cc-pVDZ by finite-differencing dE/dh:

      ||D_oo|| = 0.249    ||D_vv|| = 0.159    ||z|| = 0.057

The two "extra" blocks that MP2 does not have are three times larger than
the Z-vector block.  So for OBMP2 no purely-ov correction can recover the
dipole -- a correct Z-vector would still leave the oo/vv blocks wrong.

Dipoles on CN / cc-pVDZ, same converged solution:

      determinant  (solver.dip_mom)          1.2329 D
      unrelaxed    (this module)             1.0928 D
      relaxed = -dE/dF (finite field)        0.4141 D

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    from obmp2_rdm1 import attach

    s = OBMP2_CL(mf)
    s.second_order, s.ampf, s.shift = True, 1.0, 0.0
    s.run()
    attach(s)                    # adds attributes, never overwrites dip_mom

    s.dip_mom                    # 1.2329  determinant  (pyCMF, untouched)
    s.dip_mom_unrelaxed          # 1.0928  I + d_oo + d_vv
    s.natural_occupations()      # check 0 <= n <= 1 (N-representability)

For the hybrid (OBDH) branch, pass alpha_c = mp.alphaa[1] to scale the
correlation blocks, matching how the B2PLYP density is built elsewhere in
the benchmark.  Note this is an extrapolation: PCCP 2022 never applied
OBMP2 densities to a double hybrid, so state it explicitly if you use it.
"""

import numpy
from pyscf import ao2mo, lib, scf

__all__ = ['make_rdm1_unrelaxed', 'dipole_unrelaxed', 'natural_occupations',
           'mp1_amplitudes', 'gamma1_oo_vv', 'attach']


def _ovov(mol, co, cv, cO, cV):
    """Chemist-notation (ia|jb), shaped (nocc, nvir, noccB, nvirB)."""
    eri = ao2mo.general(mol, (co, cv, cO, cV), compact=False)
    return eri.reshape(co.shape[1], cv.shape[1], cO.shape[1], cV.shape[1])


def mp1_amplitudes(mp, ampf=None, css=1.0, cos=1.0, shift=None):
    """T2 of eqn (4), built from the CONVERGED OBMP2 orbitals and orbital
    energies.  Returned in pyscf index order t2aa[i,j,a,b], t2ab[i,J,a,B].

    Note the denominators use mp.mo_energy, which are eigenvalues of the
    correlated Fock Fbar -- this is exactly what makes dE/dT nonzero.
    """
    mol = mp._scf.mol
    mo, eps, occ = mp.mo_coeff, mp.mo_energy, mp._scf.mo_occ
    if ampf is None:
        ampf = getattr(mp, 'ampf', 1.0)
    if shift is None:
        shift = getattr(mp, 'shift', 0.0)

    noa = int(numpy.count_nonzero(occ[0] > 0))
    nob = int(numpy.count_nonzero(occ[1] > 0))
    coa, cva = mo[0][:, :noa], mo[0][:, noa:]
    cob, cvb = mo[1][:, :nob], mo[1][:, nob:]
    ea = eps[0][:noa, None] - eps[0][None, noa:]
    eb = eps[1][:nob, None] - eps[1][None, nob:]

    g = _ovov(mol, coa, cva, coa, cva)
    t = g / (ea[:, :, None, None] + ea[None, None, :, :] - shift)
    t2aa = (t - t.transpose(0, 3, 2, 1)).transpose(0, 2, 1, 3) * (css * ampf)

    g = _ovov(mol, cob, cvb, cob, cvb)
    t = g / (eb[:, :, None, None] + eb[None, None, :, :] - shift)
    t2bb = (t - t.transpose(0, 3, 2, 1)).transpose(0, 2, 1, 3) * (css * ampf)

    g = _ovov(mol, coa, cva, cob, cvb)
    t2ab = (g / (ea[:, :, None, None] + eb[None, None, :, :] - shift)
            ).transpose(0, 2, 1, 3) * (cos * ampf)

    return t2aa, t2ab, t2bb


def gamma1_oo_vv(t2aa, t2ab, t2bb):
    """The d_oo and d_vv blocks.  Matches pyscf.mp.ump2._gamma1_intermediates.

    d_oo is negative-definite, d_vv positive-definite, and their traces
    cancel -- this is the charge that correlation moves out of the occupied
    orbitals and into the virtual ones.
    """
    dooa = lib.einsum('imef,jmef->ij', t2aa.conj(), t2aa) * -.5
    dooa -= lib.einsum('imef,jmef->ij', t2ab.conj(), t2ab)
    doob = lib.einsum('imef,jmef->ij', t2bb.conj(), t2bb) * -.5
    doob -= lib.einsum('mief,mjef->ij', t2ab.conj(), t2ab)

    dvva = lib.einsum('mnae,mnbe->ba', t2aa.conj(), t2aa) * .5
    dvva += lib.einsum('mnae,mnbe->ba', t2ab.conj(), t2ab)
    dvvb = lib.einsum('mnae,mnbe->ba', t2bb.conj(), t2bb) * .5
    dvvb += lib.einsum('mnea,mneb->ba', t2ab.conj(), t2ab)
    return (dooa, doob), (dvva, dvvb)


def make_rdm1_unrelaxed(mp, ao_repr=True, alpha_c=1.0,
                        ampf=None, css=1.0, cos=1.0, shift=None):
    """gamma = I + d_oo + d_vv, with the ov block left at zero.

    Args:
        alpha_c: scale factor on d_oo and d_vv.  Leave at 1.0 for pure
            OBMP2.  For OBDH pass mp.alphaa[1] if you want the correlation
            blocks scaled the same way the double-hybrid energy is.

    Returns:
        (dm_alpha, dm_beta).  ao_repr=True gives the AO basis.
    """
    occ = mp._scf.mo_occ
    noa = int(numpy.count_nonzero(occ[0] > 0))
    nob = int(numpy.count_nonzero(occ[1] > 0))
    nma, nmb = mp.mo_coeff[0].shape[1], mp.mo_coeff[1].shape[1]

    t2 = mp1_amplitudes(mp, ampf, css, cos, shift)
    (dooa, doob), (dvva, dvvb) = gamma1_oo_vv(*t2)

    dm = []
    for n_o, n_m, doo, dvv, C in ((noa, nma, dooa, dvva, mp.mo_coeff[0]),
                                  (nob, nmb, doob, dvvb, mp.mo_coeff[1])):
        d = numpy.zeros((n_m, n_m))
        d[:n_o, :n_o] = .5 * alpha_c * (doo + doo.conj().T)   # d_oo
        d[n_o:, n_o:] = .5 * alpha_c * (dvv + dvv.conj().T)   # d_vv
        d[numpy.diag_indices(n_o)] += 1.0                     # the I of the determinant
        # ov block deliberately left at zero: no orbital response here.
        dm.append(C @ d @ C.conj().T if ao_repr else d)
    return tuple(dm)


def dipole_unrelaxed(mp, unit='Debye', alpha_c=1.0):
    """|mu| from the unrelaxed density.  Comparable with the OBMP2 column
    of Table 2 in Tran, PCCP 2022.  This is NOT -dE/dF."""
    dm = make_rdm1_unrelaxed(mp, ao_repr=True, alpha_c=alpha_c)
    return float(numpy.linalg.norm(
        scf.hf.dip_moment(mp._scf.mol, dm, unit=unit, verbose=0)))


def natural_occupations(mp, alpha_c=1.0):
    """Eigenvalues of the unrelaxed density, descending, per spin.

    A determinant gives exactly 1s and 0s.  Fractional values are the
    signature of correlation.  Any value outside [0, 1] means the density
    is not N-representable -- worth checking on difficult radicals.
    """
    dm = make_rdm1_unrelaxed(mp, ao_repr=False, alpha_c=alpha_c)
    return [numpy.sort(numpy.linalg.eigvalsh(d))[::-1] for d in dm]


def attach(mp, alpha_c=1.0):
    """Add .rdm1_unrelaxed / .dip_mom_unrelaxed / .natural_occupations to a
    solver instance that has already run.

    Does NOT overwrite .dip_mom or ._gamma.  Overwriting them would make
    every previously computed number irreproducible, and the two dipoles
    answer different questions anyway.
    """
    cls = type(mp)
    if not hasattr(cls, 'dip_mom_unrelaxed'):
        cls.rdm1_unrelaxed = make_rdm1_unrelaxed
        cls.dip_mom_unrelaxed = property(
            lambda self: dipole_unrelaxed(self, alpha_c=getattr(self, '_alpha_c', 1.0)))
        cls.natural_occupations = natural_occupations
    mp._alpha_c = alpha_c
    return mp