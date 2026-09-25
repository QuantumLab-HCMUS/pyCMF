"""
Unrelaxed correlated 1-RDM for UOBMP2 / UOBDH in pyCMF.

--------------------------------------------------------------------------
THE THREE DENSITIES
--------------------------------------------------------------------------
Write the 1-RDM in the MO basis of the converged OBMP2 orbitals and split
the MO index set into occupied (o) and virtual (v).  Every density in this
story is the same matrix with different blocks filled in:

                       oo block          vv block      ov block
  determinant          I                 0             0
  unrelaxed            I + d_oo          d_vv          0
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
dE/dT != 0.  Block decomposition of (relaxed - unrelaxed) on CN / 6-31G,
in Debye:

      z_ov  = +0.3761      (orbital response)
      D_oo  = -0.5185      (amplitude response)
      D_vv  = -0.0809      (amplitude response)

The two branches have OPPOSITE SIGN and the amplitude branch is 1.6x
larger, so a Z-vector-only correction makes the error WORSE.  No purely-ov
patch can recover the OBMP2 dipole.

--------------------------------------------------------------------------
REFERENCE VALUES
--------------------------------------------------------------------------
CN, r(C-N) = 1.1718 A, UHF reference, pure OBMP2 (is_hybrid=False,
ampf=1, shift=0).  Verified against the solver at pyCMF 6518b23 /
PySCF 2.14.0; finite field is a central difference at F = 1e-4 a.u.

                                       6-31G      cc-pVDZ
      determinant  (solver.dip_mom)    1.3602 D   1.2329 D
      unrelaxed    (this module)       1.2116 D   1.0928 D
      relaxed = -dE/dF (finite field)  0.9883 D   0.8751 D

The ordering det > unrelaxed > relaxed holds for CN and is a cheap check
that benchmark columns have not been mislabelled.

--------------------------------------------------------------------------
WHERE THE AMPLITUDES COME FROM
--------------------------------------------------------------------------
`make_rdm1_unrelaxed` calls the solver's OWN `make_amp` (uobdh_solver).
That solver is density-fitted -- BaseEmbedOBMP2 derives from DFOBMP2 and
make_amp contracts 3-centre qov blocks off `mp.with_df` -- and it applies
mp.css / mp.cos inside.  Rebuilding (ia|jb) here from exact 4-centre AO
integrals would give amplitudes differing from the converged ones by
~6e-4 relative, would silently drop css/cos, and would cost O(N^4)
memory.  Reusing make_amp removes all three problems by construction.

`ampf` is NOT applied inside make_amp -- the solver scales the barred
amplitudes afterwards (uobdh_solver.py:501-504) -- so this module applies
it, exactly once.

`mp1_amplitudes` is the independent rebuild path, kept as a cross-check
(it also accepts a plain pyscf UMP2 object, which is how this module is
validated against pyscf.mp.ump2.make_rdm1).  It is not used in production.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    from pycmf.OBDH.obdh_rdm1 import attach

    s = UOBMP2(mf)
    s.second_order, s.ampf, s.shift = True, 1.0, 0.0
    s.run()
    attach(s)                    # adds attributes, never overwrites dip_mom

    s.dip_mom                    # 1.2329  determinant  (pyCMF, untouched)
    s.dip_mom_unrelaxed          # 1.0928  I + d_oo + d_vv
    s.natural_occupations()      # check 0 <= n <= 1 (N-representability)

For the hybrid (OBDH) branch pass alpha_c = mp.alphaa[1] to attach(); it
scales the correlation blocks the way the double-hybrid energy scales the
MP2 term.  All three attached attributes then read that same value.  Note
this is an extrapolation: PCCP 2022 never applied OBMP2 densities to a
double hybrid, so state it explicitly if you use it.
"""

import io

import numpy
from pyscf import ao2mo, lib, scf

__all__ = ['make_rdm1_unrelaxed', 'dipole_unrelaxed', 'natural_occupations',
           'solver_amplitudes', 'mp1_amplitudes', 'gamma1_oo_vv', 'attach']


# --------------------------------------------------------------------------
# amplitudes
# --------------------------------------------------------------------------

def _ovov_to_oovv(t):
    """(i,a,j,b) -> (i,j,a,b), the pyscf ordering used by gamma1_oo_vv."""
    return numpy.ascontiguousarray(t.transpose(0, 2, 1, 3))


def solver_amplitudes(mp, ampf=None):
    """T2 of eqn (4) as the solver actually built it, returned in pyscf
    index order t2aa[i,j,a,b], t2ab[i,J,a,B], t2bb[I,J,A,B].

    Same integrals (DF), same denominators (eigenvalues of the correlated
    Fock Fbar), same css/cos as the converged energy.  `ampf` is applied
    here because the solver applies it after make_amp returns.
    """
    from .uobdh_solver import make_amp

    if ampf is None:
        ampf = getattr(mp, 'ampf', 1.0)

    # make_amp logs at verbose=5 unconditionally; mute it so a 152-molecule
    # sweep does not drown in integral-transform chatter.
    stdout, mp.stdout = mp.stdout, io.StringIO()
    try:
        _tmp1, tmp1_bar = make_amp(mp)
    finally:
        mp.stdout = stdout

    bar_aa, bar_bb, bar_ab, _bar_ba = tmp1_bar   # NOTE the order: aa, bb, ab, ba
    return (_ovov_to_oovv(bar_aa) * ampf,
            _ovov_to_oovv(bar_ab) * ampf,
            _ovov_to_oovv(bar_bb) * ampf)


def _ovov(mp, co, cv, cO, cV):
    """Chemist-notation (ia|jb), shaped (nocc, nvir, noccB, nvirB).

    Uses whatever integral engine the object carries, in the same order of
    preference as the solver: mp.with_df first (DFOBMP2 always has one),
    then the mean-field's stored _eri, then exact integrals from mol.
    """
    with_df = getattr(mp, 'with_df', None)
    if with_df is not None:
        eri = with_df.ao2mo((co, cv, cO, cV), compact=False)
    elif getattr(mp._scf, '_eri', None) is not None:
        eri = ao2mo.general(mp._scf._eri, (co, cv, cO, cV), compact=False)
    else:
        eri = ao2mo.general(mp._scf.mol, (co, cv, cO, cV), compact=False)
    return eri.reshape(co.shape[1], cv.shape[1], cO.shape[1], cV.shape[1])


def mp1_amplitudes(mp, ampf=None, css=None, cos=None, shift=None):
    """Independent rebuild of T2 from mp.mo_coeff / mp.mo_energy.

    Cross-check path only -- `make_rdm1_unrelaxed` uses `solver_amplitudes`.
    Also works on a plain pyscf UMP2 object (set mp.mo_energy first), which
    is how this module is validated against pyscf.mp.ump2.

    All four scale factors default to the values carried by `mp`, so this
    cannot silently disagree with the solver on css/cos the way a hardwired
    1.0 would.
    """
    if ampf is None:
        ampf = getattr(mp, 'ampf', 1.0)
    if shift is None:
        shift = getattr(mp, 'shift', 0.0)
    if css is None:
        css = getattr(mp, 'css', 1.0)
    if cos is None:
        cos = getattr(mp, 'cos', 1.0)

    mo = mp.mo_coeff
    eps = getattr(mp, 'mo_energy', None)
    if eps is None:
        eps = mp._scf.mo_energy
    noa, nob = _nocc(mp)

    coa, cva = mo[0][:, :noa], mo[0][:, noa:]
    cob, cvb = mo[1][:, :nob], mo[1][:, nob:]
    ea = eps[0][:noa, None] - eps[0][None, noa:]
    eb = eps[1][:nob, None] - eps[1][None, nob:]

    g = _ovov(mp, coa, cva, coa, cva)
    t = g / (ea[:, :, None, None] + ea[None, None, :, :] - shift)
    t2aa = (t - t.transpose(0, 3, 2, 1)).transpose(0, 2, 1, 3) * (css * ampf)

    g = _ovov(mp, cob, cvb, cob, cvb)
    t = g / (eb[:, :, None, None] + eb[None, None, :, :] - shift)
    t2bb = (t - t.transpose(0, 3, 2, 1)).transpose(0, 2, 1, 3) * (css * ampf)

    g = _ovov(mp, coa, cva, cob, cvb)
    t2ab = (g / (ea[:, :, None, None] + eb[None, None, :, :] - shift)
            ).transpose(0, 2, 1, 3) * (cos * ampf)

    return (numpy.ascontiguousarray(t2aa),
            numpy.ascontiguousarray(t2ab),
            numpy.ascontiguousarray(t2bb))


# --------------------------------------------------------------------------
# density
# --------------------------------------------------------------------------

def _nocc(mp):
    """Occupied counts, taken the way the solver takes them.

    Deliberately NOT mp._scf.mo_occ: the solver rewrites mp.mo_coeff,
    mp.mo_energy and mp.mo_occ (uobdh_solver.py:432-449, 644-645) while
    mp._scf keeps the starting HF occupancies.  Under CL truncation the
    two are not even the same length.
    """
    if hasattr(mp, 'get_nocc'):
        noa, nob = mp.get_nocc()
        return int(noa), int(nob)
    occ = mp.mo_occ
    return int(numpy.count_nonzero(occ[0] > 0)), int(numpy.count_nonzero(occ[1] > 0))


def _nmo(mp):
    if hasattr(mp, 'get_nmo'):
        nma, nmb = mp.get_nmo()
        return int(nma), int(nmb)
    return mp.mo_coeff[0].shape[1], mp.mo_coeff[1].shape[1]


def gamma1_oo_vv(t2aa, t2ab, t2bb):
    """The d_oo and d_vv blocks.  Byte-identical to
    pyscf.mp.ump2._gamma1_intermediates and to
    pycmf.OBMP.uobmp2._gamma1_intermediates.

    d_oo is negative-definite, d_vv positive-definite, and their traces
    cancel -- this is the charge that correlation moves out of the occupied
    orbitals and into the virtual ones.

    Expects pyscf index order (i,j,a,b).  Passing the solver's native ovov
    layout (i,a,j,b) here runs without error and returns garbage.
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


def make_rdm1_unrelaxed(mp, ao_repr=True, alpha_c=1.0, t2=None):
    """gamma = I + d_oo + d_vv, with the ov block left at zero.

    Args:
        alpha_c: scale factor on d_oo and d_vv.  Leave at 1.0 for pure
            OBMP2.  For OBDH pass mp.alphaa[1] to scale the correlation
            blocks the same way the double-hybrid energy is scaled.
        t2: (t2aa, t2ab, t2bb) in pyscf order, if you already have them.
            Defaults to the solver's own amplitudes.

    Returns:
        (dm_alpha, dm_beta).  ao_repr=True gives the AO basis.
    """
    noa, nob = _nocc(mp)
    nma, nmb = _nmo(mp)
    if mp.mo_coeff[0].shape[1] != nma or mp.mo_coeff[1].shape[1] != nmb:
        raise ValueError('mo_coeff has %s columns but get_nmo() says %s -- the '
                         'occupancy arrays and the orbitals are out of sync'
                         % ((mp.mo_coeff[0].shape[1], mp.mo_coeff[1].shape[1]),
                            (nma, nmb)))

    if t2 is None:
        t2 = solver_amplitudes(mp)
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


def dipole_unrelaxed(mp, unit='Debye', alpha_c=1.0, t2=None):
    """|mu| from the unrelaxed density.  Comparable with the OBMP2 column
    of Table 2 in Tran, PCCP 2022.  This is NOT -dE/dF."""
    dm = make_rdm1_unrelaxed(mp, ao_repr=True, alpha_c=alpha_c, t2=t2)
    return float(numpy.linalg.norm(
        scf.hf.dip_moment(mp._scf.mol, dm, unit=unit, verbose=0)))


def natural_occupations(mp, alpha_c=1.0, t2=None):
    """Eigenvalues of the unrelaxed density, descending, per spin.

    A determinant gives exactly 1s and 0s.  Fractional values are the
    signature of correlation.  Any value outside [0, 1] means the density
    is not N-representable -- worth checking on difficult radicals.
    """
    dm = make_rdm1_unrelaxed(mp, ao_repr=False, alpha_c=alpha_c, t2=t2)
    return [numpy.sort(numpy.linalg.eigvalsh(d))[::-1] for d in dm]


# --------------------------------------------------------------------------
# attachment
# --------------------------------------------------------------------------

def attach(mp, alpha_c=1.0):
    """Add .rdm1_unrelaxed / .dip_mom_unrelaxed / .natural_occupations to a
    solver instance that has already run.

    Does NOT overwrite .dip_mom or ._gamma.  Overwriting them would make
    every previously computed number irreproducible, and the two dipoles
    answer different questions anyway.

    All three attributes read the same per-instance _alpha_c, so they can
    never disagree.  Binding make_rdm1_unrelaxed directly would leave its
    own alpha_c default at 1.0 and silently ignore the value passed here --
    invisible for pure OBMP2 (alpha_c = 1) and a ~7% error for OBDH.
    """

    def _rdm1(self, ao_repr=True, alpha_c=None, t2=None):
        if alpha_c is None:
            alpha_c = getattr(self, '_alpha_c', 1.0)
        return make_rdm1_unrelaxed(self, ao_repr=ao_repr, alpha_c=alpha_c, t2=t2)

    def _natocc(self, alpha_c=None, t2=None):
        if alpha_c is None:
            alpha_c = getattr(self, '_alpha_c', 1.0)
        return natural_occupations(self, alpha_c=alpha_c, t2=t2)

    def _dip(self):
        return dipole_unrelaxed(self, alpha_c=getattr(self, '_alpha_c', 1.0))

    cls = type(mp)
    # Rebind every time: a hasattr() guard makes the bindings survive a
    # module reload with the old code still in place.
    cls.rdm1_unrelaxed = _rdm1
    cls.natural_occupations = _natocc
    cls.dip_mom_unrelaxed = property(_dip)
    mp._alpha_c = alpha_c
    return mp