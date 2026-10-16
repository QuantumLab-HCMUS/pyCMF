#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''
Experimental high-spin restricted-open-shell OB-MP2 (RO-OBMP2).

This implementation keeps one common spatial-orbital coefficient matrix for
alpha and beta spin channels.  The alpha/beta occupation spaces are generated
from a 1-D ROHF occupation vector (2=closed, 1=open, 0=virtual), and every
orbital update is performed with a single Roothaan effective Fock matrix.

'''

import time
import hashlib
from functools import reduce

import numpy
import scipy.linalg

from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf import __config__
from pyscf.scf.rohf import get_roothaan_fock
from pyscf.scf.uhf import spin_square as uhf_spin_square

# Base OBMP2 from:
# pycmf/obmp2/obmp2.py
from pycmf.obmp2.obmp2 import OBMP2
from pycmf.obmp2.obmp2 import _ChemistsERIs as _OBMP2ChemistsERIs

from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor


WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


# Iterative kernel


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=logger.NOTE):
    """Run RO-OBMP2, invalidating old results even if input checking fails."""
    _clear_results(mp)
    mp._kernel_run_id = getattr(mp, '_kernel_run_id', 0) + 1
    completed = False
    try:
        result = _kernel(mp, mo_energy, mo_coeff, eris, with_t2, verbose)
        completed = True
        return result
    finally:
        # Includes exceptions during validation, iteration or finalization,
        # and interruptions.  Never expose a partially committed result.
        if not completed:
            _clear_results(mp)


def _kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
            verbose=logger.NOTE):
    """Run the high-spin restricted-open-shell OBMP2 iteration.

    The UOBMP2-style helpers build HF, amplitudes, BCH1 and BCH2;
    the orbital representation and orbital update remain ROHF:
    one common spatial C matrix, alpha/beta occupations derived from 2/1/0
    ROHF occupations, and a single Roothaan effective-Fock diagonalization.
    """
    coeff, mo_occ_ro, eps_ro, eps_a, eps_b = _initial_rohf_orbitals(
        mp, mo_energy=mo_energy, mo_coeff=mo_coeff
    )

    # Keep a private 1-D occupation vector because the OBMP2 base class was
    # originally written for a different representation.
    mp._ro_mo_occ = mo_occ_ro.copy()

    # Internal spin-block algebra still uses alpha/beta channels.  The crucial
    # RO restriction is C_alpha == C_beta at all times.
    mo_coeff = [coeff.copy(), coeff.copy()]
    mo_energy = [eps_a.copy(), eps_b.copy()]
    occa, occb = _spin_occ_from_rohf(mo_occ_ro)
    mo_occ = [occa, occb]

    nocca, noccb = get_nocc(mp)
    nmoa, nmob = get_nmo(mp)
    nvira, nvirb = nmoa - nocca, nmob - noccb

    ncore = noccb
    nopen = nocca - noccb

    # Reference energy is the converged ROHF total energy.  Do not overwrite
    # this value later when forming the HF-like one-body contribution.
    nuc = mp._scf.energy_nuc()
    ene_hf_ref = mp._scf.energy_tot()

    shift = mp.shift


    niter = mp.niter
    if niter < 1:
        raise ValueError('niter must be at least one.')
    if not isinstance(mp.second_order, (bool, numpy.bool_)):
        raise TypeError('second_order must be True or False.')
    if mp.break_sym:
        raise NotImplementedError(
            'The legacy break_sym contractions are not supported for ROOBMP2.')
    ampf = _resolve_amplitude_factor(mp)
    if not numpy.isfinite(mp.denom_thresh) or mp.denom_thresh < 0:
        raise ValueError('denom_thresh must be finite and non-negative.')
    if not numpy.isfinite(shift):
        raise ValueError('shift must be finite.')
    mp.ampf_used = ampf
    mp.bch_order = 2 if mp.second_order else 1
    ene_old = None
    conv = False
    eri_ao = mp.mol.intor('int2e')

    print("shift = ", mp.shift)
    print ("thresh = ", mp.thresh)
    print ("niter = ", mp.niter)
    print('BCH order = %d; amplitude factor = %.8g' % (mp.bch_order, ampf))

    for it in range(niter):
        coeff = mo_coeff[0]
        if not numpy.allclose(coeff, mo_coeff[1], atol=1e-12, rtol=0.0):
            raise RuntimeError('ROOBMP2 internal error: alpha/beta spatial orbitals diverged.')

        h2mo = int_transform_ss(eri_ao, coeff)

        h1ao = mp._scf.get_hcore(mp.mol)
        h1mo = numpy.matmul(coeff.T, numpy.matmul(h1ao, coeff))
        h1mo_a = h1mo.copy()
        h1mo_b = h1mo.copy()

        #####################
        ### Hartree-Fock

        veffa, veffb, c0 = make_veff(mp, h2mo)
        fock_hfa = h1mo_a + veffa
        fock_hfb = h1mo_b + veffb

        if it == 0:
            fock_error_a, fock_error_b = _rohf_reference_fock_error(
                mp, coeff, mo_occ_ro, fock_hfa, fock_hfb)
            print('ROHF alpha-Fock check = %.6e' % fock_error_a)
            print('ROHF beta-Fock check  = %.6e' % fock_error_b)
            if max(fock_error_a, fock_error_b) > mp.fock_check_thresh:
                logger.warn(
                    mp,
                    'Inherited integral-loop Fock does not match the PySCF '
                    'ROHF spin Fock. Do not interpret correlated results until '
                    'the ERI convention is corrected.')

        fock_a = fock_hfa.copy()
        fock_b = fock_hfb.copy()

        ene_hf = 0.0
        for i in range(nocca):
            ene_hf += fock_a[i, i]
        for i in range(noccb):
            ene_hf += fock_b[i, i]
        c0 *= 0.5
        ene_hf += c0

        if it == 0:
            # Compare the SAME determinant, including a caller-supplied start.
            dm_check = mp._scf.make_rdm1(coeff, mo_occ_ro)
            hf_energy_check = mp._scf.energy_tot(dm=dm_check)
            reference_energy_error = abs((ene_hf + nuc) - hf_energy_check)
            print('ROHF reference-energy check = %.6e' % reference_energy_error)
            if reference_energy_error > mp.energy_check_thresh:
                logger.warn(
                    mp,
                    'Inherited spin-resolved HF energy reconstruction does '
                    'not match the ROHF reference energy.')

        # MP1 amplitudes: the same scale is applied to raw and barred T2.
        (tmp1_aa, tmp1_bb, tmp1_ab, tmp1_ba,
         tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab, tmp1_bar_ba,
         min_abs_denom) = make_amp(mp, mo_energy, h2mo, ampf=ampf)

        # BCH 1st order
        c0_hf = float(c0)
        c0, c1_a, c1_b = first_BCH(
            mp, fock_hfa, fock_hfb, tmp1_bar_aa, tmp1_bar_bb,
            tmp1_bar_ab, tmp1_bar_ba, c0, h2mo)
        fock_a += 0.5 * (c1_a + c1_a.T)
        fock_b += 0.5 * (c1_b + c1_b.T)

        delta_f1_a = fock_a - fock_hfa
        delta_f1_b = fock_b - fock_hfb
        c0_after_bch1 = float(c0)
        delta_f2_a = numpy.zeros_like(fock_a)
        delta_f2_b = numpy.zeros_like(fock_b)

        # BCH 2nd order
        if mp.second_order and ampf != 0.0:
            c0, c1_a_2nd, c1_b_2nd = second_BCH(
                mp, fock_hfa, fock_hfb, tmp1_aa, tmp1_bb, tmp1_ab,
                tmp1_ba, tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab,
                tmp1_bar_ba, c0)
            delta_f2_a = 0.5 * (c1_a_2nd + c1_a_2nd.T)
            delta_f2_b = 0.5 * (c1_b_2nd + c1_b_2nd.T)
            fock_a += delta_f2_a
            fock_b += delta_f2_b

        ene = c0
        for i in range(nocca):
            ene += 1. * fock_a[i,i]
        for i in range(noccb):
            ene += 1. * fock_b[i,i]

        ene_tot = ene + nuc
        if not numpy.isfinite(ene_tot):
            raise FloatingPointError('The OBMP2 total energy is NaN or infinite.')
        de = numpy.inf if ene_old is None else abs(ene_tot - ene_old)
        ene_old = ene_tot
        print()
        print("========================")
        print('iter = %d'%it, ' energy = %8.6f'%ene_tot, ' energy diff = %8.6f'%de, flush=True)
        print()

        if mp.eval_fc:
            print("Fermi contact using HF-like density")
            mp.mo_coeff = coeff.copy()
            mp.mo_occ = mo_occ_ro.copy()
            rdm1 = mp.make_rdm1()
            mp.make_fc(rdm1)

        # ------------------------------------------------------------
        # ROHF orbital update
        # ------------------------------------------------------------
        # UOBMP2 diagonalizes F_alpha and F_beta independently.  That would
        # immediately break the restricted-open-shell constraint.  Instead we
        # construct one Roothaan effective Fock and diagonalize it once.
        fock_eff = _roothaan_fock_mo(fock_a, fock_b, mo_occ_ro)
        orbital_residual = _roothaan_orbital_residual(fock_eff, mo_occ_ro)
        # Denominators use spin energies carried from the preceding update.
        # Check their fixed point as well as the density-changing RO blocks.
        denominator_residual = max(
            numpy.max(numpy.abs(numpy.diag(fock_a) - mo_energy[0])),
            numpy.max(numpy.abs(numpy.diag(fock_b) - mo_energy[1])))
        print('RO orbital residual = %.6e' % orbital_residual)
        print('Spin-energy residual = %.6e' % denominator_residual)
        if mp.second_order:
            print('BCH2 Fock norms (alpha, beta) = %.6e, %.6e' %
                  (numpy.linalg.norm(delta_f2_a), numpy.linalg.norm(delta_f2_b)))
        if numpy.isfinite(min_abs_denom):
            print('Minimum |MP denominator| = %.6e' % min_abs_denom)
        else:
            print('Minimum |MP denominator| = not formed (ampf=0 or empty channel)')

        # Keep the returned C, energy and Focks at the same evaluated iterate.
        # In particular, do not rotate C after the last permitted Fock build.
        conv = (de < mp.thresh and orbital_residual < mp.grad_thresh and
                denominator_residual < mp.energy_consistency_thresh)
        if conv or it == niter - 1:
            break

        eps_eff_raw, U_raw = scipy.linalg.eigh(fock_eff)

        # Spin-resolved diagonal Fock energies in the new common orbital basis
        # are retained for MP2 denominators.
        fa_rot = reduce(numpy.dot, (U_raw.conj().T, fock_a, U_raw))
        fb_rot = reduce(numpy.dot, (U_raw.conj().T, fock_b, U_raw))
        eps_a_raw = numpy.real(numpy.diag(fa_rot))
        eps_b_raw = numpy.real(numpy.diag(fb_rot))

        # Assign closed/open/virtual occupations with the ROHF rule, then
        # reorder into contiguous C/O/V blocks because the inherited tensor
        # contractions use occupied-first slicing.
        occ_raw = _fill_rohf_occ(eps_eff_raw, eps_a_raw, ncore, nopen)
        idx_c = numpy.where(occ_raw == 2)[0]
        idx_o = numpy.where(occ_raw == 1)[0]
        idx_v = numpy.where(occ_raw == 0)[0]
        order = numpy.concatenate((idx_c, idx_o, idx_v))

        U = U_raw[:, order]
        eps_ro = numpy.asarray(eps_eff_raw)[order].copy()
        eps_a = numpy.asarray(eps_a_raw)[order].copy()
        eps_b = numpy.asarray(eps_b_raw)[order].copy()
        mo_occ_ro = numpy.asarray(occ_raw)[order].copy()
        mp._ro_mo_occ = mo_occ_ro.copy()

        coeff = numpy.matmul(coeff, U)
        mo_coeff = [coeff.copy(), coeff.copy()]
        mo_energy = [eps_a.copy(), eps_b.copy()]
        occa, occb = _spin_occ_from_rohf(mo_occ_ro)
        mo_occ = [occa, occb]

    # Store an ROHF-like public representation: one spatial coefficient
    # matrix, with all Focks and their diagonals in that SAME final MO basis.
    # If unconverged, these diagonal expectations are not canonical eigenvalues.
    mp.mo_coeff = mo_coeff[0].copy()
    mp.denominator_energies = (mo_energy[0].copy(), mo_energy[1].copy())
    mp.mo_energy = lib.tag_array(
        numpy.diag(fock_eff).copy(), mo_ea=numpy.diag(fock_a).copy(),
        mo_eb=numpy.diag(fock_b).copy())
    mp._ro_mo_occ = mo_occ_ro.copy()
    mp.mo_occ = mo_occ_ro.copy()

    # PySCF MP2 convention: e_corr is relative to the original SCF reference.
    mp.e_corr = float(ene_tot - ene_hf_ref)
    mp.e_tot_obmp2 = float(ene_tot)
    mp.fock_corr_a = numpy.asarray(fock_a).copy()
    mp.fock_corr_b = numpy.asarray(fock_b).copy()
    mp.fock_ro = numpy.asarray(fock_eff).copy()
    mp.fock_hf_a = fock_hfa.copy()
    mp.fock_hf_b = fock_hfb.copy()
    mp.bch1_fock_correction = (delta_f1_a.copy(), delta_f1_b.copy())
    mp.bch2_fock_correction = (delta_f2_a.copy(), delta_f2_b.copy())
    mp.bch1_scalar_correction = float(c0_after_bch1 - c0_hf)
    mp.bch2_scalar_correction = float(c0 - c0_after_bch1)
    mp.scalar_constant = float(c0)
    mp.orbital_residual = float(orbital_residual)
    mp.denominator_residual = float(denominator_residual)
    mp.cycles = it + 1
    mp.min_abs_denom = float(min_abs_denom)
    mp.reference_fock_error = (fock_error_a, fock_error_b)
    mp.reference_energy_error = float(reference_energy_error)
    # Save the SAME scaled amplitudes used for this evaluated energy/Fock.
    # (i,a,j,b) -> (i,j,a,b); aa/bb are antisymmetrized.  Counting a separate
    # ba density contribution as well as ab would double count opposite spin.
    if with_t2:
        mp.t2 = (tmp1_bar_aa.transpose(0, 2, 1, 3).copy(),
                 tmp1_bar_ab.transpose(0, 2, 1, 3).copy(),
                 tmp1_bar_bb.transpose(0, 2, 1, 3).copy())
        mp.t2_metadata = {
            'run_id': mp._kernel_run_id,
            'bch_order': mp.bch_order,
            'amplitude_factor': float(ampf),
            'shift': float(shift),
            'converged': bool(conv),
            'cycle': it + 1,
            'index_order': '(i,j,a,b)',
            'spin_blocks': ('aa antisymmetrized', 'ab', 'bb antisymmetrized'),
        }
        mp._t2_signature = _amplitude_signature(mp)

    occupied = (mp.mo_coeff[:, mo_occ_ro > 0], mp.mo_coeff[:, mo_occ_ro == 2])
    mp.ss_det, mp.multiplicity_det = uhf_spin_square(occupied, mp._scf.get_ovlp())

    # A common spatial-orbital ROHF determinant with n_alpha >= n_beta is an
    # eigenfunction of S^2 for the high-spin occupation pattern.
    s_target = 0.5 * (nocca - noccb)
    ss_target = s_target * (s_target + 1.0)
    print('ROOBMP2 target <S^2> = %.8g  2S+1 = %.8g' %
          (ss_target, 2.0 * s_target + 1.0))
    print('RO-OBMP2 <S^2>det = %.10f  2S+1 = %.10f' %
          (mp.ss_det, mp.multiplicity_det))
    print('source: method determinant only')

    print()
    if conv:
        print('RO-OBMP2 has converged')
    else:
        print('RO-OBMP2 has not converged')
    print('RO-OBMP2 total energy = ', ene_tot)
    print('RO-OBMP2 correlation energy = ', mp.e_corr)

    # Commit validity only after every finalization step has succeeded.
    mp.converged = bool(conv)
    mp._final_state_available = True
    return mp.e_corr


# Spin-resolved HF, amplitudes and BCH contractions


def make_veff(mp, h2mo):
    """Return the spin HF potentials and the *unhalved* HF scalar constant.

    ``h2mo[p,q,r,s]`` is the current common-orbital Coulomb tensor (pq|rs).
    The kernel applies c0 *= 0.5, as in the UOBMP2 organization.  Passing the
    tensor explicitly avoids repeating AO-to-MO transforms or using old MOs.
    """
    nocca, noccb = mp.get_nocc()
    oa, ob = slice(None, nocca), slice(None, noccb)

    veffa = (numpy.einsum('pqkk->pq', h2mo[:, :, oa, oa])
             - numpy.einsum('pkkq->pq', h2mo[:, oa, oa, :])
             + numpy.einsum('pqkk->pq', h2mo[:, :, ob, ob]))
    veffb = (numpy.einsum('pqkk->pq', h2mo[:, :, ob, ob])
             - numpy.einsum('pkkq->pq', h2mo[:, ob, ob, :])
             + numpy.einsum('pqkk->pq', h2mo[:, :, oa, oa]))

    c0 = 0.0
    c0 -= numpy.einsum('iijj->', h2mo[oa, oa, oa, oa])
    c0 += numpy.einsum('ijji->', h2mo[oa, oa, oa, oa])
    c0 -= numpy.einsum('iijj->', h2mo[oa, oa, ob, ob])
    c0 -= numpy.einsum('iijj->', h2mo[ob, ob, ob, ob])
    c0 += numpy.einsum('ijji->', h2mo[ob, ob, ob, ob])
    c0 -= numpy.einsum('iijj->', h2mo[ob, ob, oa, oa])
    return veffa, veffb, c0


def make_amp(mp, mo_energy, h2mo, ampf=None):
    """Build scaled MP1 doubles in the UOBMP2 (i,a,j,b) index convention.

    Return aa, bb, ab, ba, their four barred tensors, and min_abs_denom.
    Both raw and barred amplitudes include ampf exactly once.  The supplied
    spin energies are those used by the current iteration, not mp.mo_energy
    (which has the public ROHF representation).  ampf=0 skips denominators.
    """
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa - nocca, nmob - noccb
    if ampf is None:
        ampf = _resolve_amplitude_factor(mp)
    tmp1_aa = numpy.zeros((nocca, nvira, nocca, nvira))
    tmp1_bb = numpy.zeros((noccb, nvirb, noccb, nvirb))
    tmp1_ab = numpy.zeros((nocca, nvira, noccb, nvirb))
    tmp1_ba = numpy.zeros((noccb, nvirb, nocca, nvira))
    min_abs_denom = numpy.inf

    if ampf != 0.0:
        mo_ea, mo_eb = mo_energy
        channels = (
            ('alpha-alpha', tmp1_aa, mo_ea, mo_ea, nocca, nocca),
            ('beta-beta', tmp1_bb, mo_eb, mo_eb, noccb, noccb),
            ('alpha-beta', tmp1_ab, mo_ea, mo_eb, nocca, noccb),
            ('beta-alpha', tmp1_ba, mo_eb, mo_ea, noccb, nocca),
        )
        for channel, amplitude, ei, ej, ni, nj in channels:
            # Keep the original order ei + ej - ea - eb - shift.  The four
            # singleton axes broadcast directly to (i,a,j,b), without tile.
            denominator = ei[:ni, None, None, None] + ej[None, None, :nj, None]
            denominator = denominator - ei[None, ni:, None, None]
            denominator = denominator - ej[None, None, None, nj:] - mp.shift
            numerator = h2mo[:ni, ni:, :nj, nj:]
            bad = (~numpy.isfinite(denominator)
                   | (numpy.abs(denominator) <= mp.denom_thresh))
            if numpy.any(bad):
                index = tuple(numpy.argwhere(bad)[0])
                _checked_amplitude(numerator[index], denominator[index],
                                   mp.denom_thresh, channel)
            if denominator.size:
                min_abs_denom = min(min_abs_denom,
                                    float(numpy.min(numpy.abs(denominator))))
            amplitude[:] = numerator / denominator
            if not numpy.all(numpy.isfinite(amplitude)):
                raise FloatingPointError('%s amplitude is NaN or infinite.' % channel)

    amplitudes = (tmp1_aa, tmp1_bb, tmp1_ab, tmp1_ba)
    for amplitude in amplitudes:
        amplitude *= ampf
    if not all(numpy.all(numpy.isfinite(t)) for t in amplitudes):
        raise FloatingPointError('MP1 doubles contain NaN or infinity.')

    tmp1_bar_aa = tmp1_aa - tmp1_aa.transpose(0, 3, 2, 1)
    tmp1_bar_bb = tmp1_bb - tmp1_bb.transpose(0, 3, 2, 1)
    tmp1_bar_ab = tmp1_ab.copy()
    tmp1_bar_ba = tmp1_ba.copy()
    return (tmp1_aa, tmp1_bb, tmp1_ab, tmp1_ba,
            tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab, tmp1_bar_ba,
            min_abs_denom)


def first_BCH(mp, fock_hfa, fock_hfb, tmp1_bar_aa, tmp1_bar_bb,
              tmp1_bar_ab, tmp1_bar_ba, c0, h2mo):
    """Return the BCH1 scalar and unsymmetrized one-body corrections.

    Same spin contractions as the original RO loops; same function layout as
    UOBMP2.  h2mo is passed in the current common MO basis.  The kernel applies
    0.5 * (c1 + c1.T) once when adding each correction to the spin Fock.
    """
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    oa, va = slice(None, nocca), slice(nocca, None)
    ob, vb = slice(None, noccb), slice(noccb, None)
    c1_a = numpy.zeros((nmoa, nmoa), dtype=fock_hfa.dtype)
    c1_b = numpy.zeros((nmob, nmob), dtype=fock_hfb.dtype)

    c0 -= numpy.einsum('iajb,iajb->', h2mo[oa, va, oa, va], tmp1_bar_aa)
    c0 -= numpy.einsum('iajb,iajb->', h2mo[ob, vb, oa, va], tmp1_bar_ba)
    c0 -= numpy.einsum('iajb,iajb->', h2mo[ob, vb, ob, vb], tmp1_bar_bb)
    c0 -= numpy.einsum('iajb,iajb->', h2mo[oa, va, ob, vb], tmp1_bar_ab)

    # Alpha: [j,B], [p,j], [p,B].  i/a belong to the first spin channel;
    # j/b belong to the second.  Lower-case a/b are virtual-space offsets.
    c1_a[oa, va] += 2.0 * numpy.einsum('ia,iajb->jb', fock_hfa[oa, va], tmp1_bar_aa)
    c1_a[oa, va] += 2.0 * numpy.einsum('ia,iajb->jb', fock_hfb[ob, vb], tmp1_bar_ba)
    c1_a[:, oa] += 2.0 * numpy.einsum('iapb,iajb->pj', h2mo[oa, va, :, va], tmp1_bar_aa)
    c1_a[:, oa] += 2.0 * numpy.einsum('iapb,iajb->pj', h2mo[ob, vb, :, va], tmp1_bar_ba)
    c1_a[:, va] -= 2.0 * numpy.einsum('iajp,iajb->pb', h2mo[oa, va, oa, :], tmp1_bar_aa)
    c1_a[:, va] -= 2.0 * numpy.einsum('iajp,iajb->pb', h2mo[ob, vb, oa, :], tmp1_bar_ba)

    # Beta: the same three blocks with the beta occupied/virtual partition.
    c1_b[ob, vb] += 2.0 * numpy.einsum('ia,iajb->jb', fock_hfb[ob, vb], tmp1_bar_bb)
    c1_b[ob, vb] += 2.0 * numpy.einsum('ia,iajb->jb', fock_hfa[oa, va], tmp1_bar_ab)
    c1_b[:, ob] += 2.0 * numpy.einsum('iapb,iajb->pj', h2mo[ob, vb, :, vb], tmp1_bar_bb)
    c1_b[:, ob] += 2.0 * numpy.einsum('iapb,iajb->pj', h2mo[oa, va, :, vb], tmp1_bar_ab)
    c1_b[:, vb] -= 2.0 * numpy.einsum('iajp,iajb->pb', h2mo[ob, vb, ob, :], tmp1_bar_bb)
    c1_b[:, vb] -= 2.0 * numpy.einsum('iajp,iajb->pb', h2mo[oa, va, ob, :], tmp1_bar_ab)
    return c0, c1_a, c1_b


def second_BCH(mp, fock_hfa, fock_hfb, tmp1_aa, tmp1_bb, tmp1_ab,
               tmp1_ba, tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab,
               tmp1_bar_ba, c0):
    """Return the BCH2 scalar and unsymmetrized one-body corrections.

    These are the original RO contractions for 1/2 * [[F_HF,A2],A2]_(0,1).
    Their numerical prefactors already include the BCH factor: do not apply
    another overall 1/2.  Keep bare HF spin Focks here, and scaled amplitudes
    in every factor.  Labels [1]-[8] match the original RO/UOBMP2 blocks.
    """
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    oa, va = slice(None, nocca), slice(nocca, None)
    ob, vb = slice(None, noccb), slice(noccb, None)
    c1_a = numpy.zeros((nmoa, nmoa), dtype=fock_hfa.dtype)
    c1_b = numpy.zeros((nmob, nmob), dtype=fock_hfb.dtype)

    # [1] F_ov contracted with two barred amplitudes -> occupied/virtual.
    y1_a = numpy.einsum('ia,iajb->jb', fock_hfa[oa, va], tmp1_bar_aa)
    y1_a += numpy.einsum('ia,iajb->jb', fock_hfb[ob, vb], tmp1_bar_ba)
    y1_b = numpy.einsum('ia,iajb->jb', fock_hfa[oa, va], tmp1_bar_ab)
    y1_b += numpy.einsum('ia,iajb->jb', fock_hfb[ob, vb], tmp1_bar_bb)
    c1_a[oa, va] += numpy.einsum('jb,iajb->ia', y1_a, tmp1_bar_aa)
    c1_a[oa, va] += numpy.einsum('jb,iajb->ia', y1_b, tmp1_bar_ab)
    c1_b[ob, vb] += numpy.einsum('jb,iajb->ia', y1_a, tmp1_bar_ba)
    c1_b[ob, vb] += numpy.einsum('jb,iajb->ia', y1_b, tmp1_bar_bb)

    # [2] y[i,a,j,b] = sum_c F[A,C] T_bar[i,c,j,b].
    y1_aa = numpy.einsum('ac,icjb->iajb', fock_hfa[va, va], tmp1_bar_aa)
    y1_ab = numpy.einsum('ac,icjb->iajb', fock_hfa[va, va], tmp1_bar_ab)
    y1_bb = numpy.einsum('ac,icjb->iajb', fock_hfb[vb, vb], tmp1_bar_bb)
    y1_ba = numpy.einsum('ac,icjb->iajb', fock_hfb[vb, vb], tmp1_bar_ba)
    c1_a[oa, oa] += numpy.einsum('iajb,iakb->jk', tmp1_aa, y1_aa)
    c1_a[oa, oa] += numpy.einsum('iajb,iakb->jk', tmp1_ba, y1_ba)
    c1_b[ob, ob] += numpy.einsum('iajb,iakb->jk', tmp1_bb, y1_bb)
    c1_b[ob, ob] += numpy.einsum('iajb,iakb->jk', tmp1_ab, y1_ab)
    c0 -= numpy.einsum('iajb,iajb->', tmp1_aa, y1_aa)
    c0 -= numpy.einsum('iajb,iajb->', tmp1_ab, y1_ab)
    c0 -= numpy.einsum('iajb,iajb->', tmp1_bb, y1_bb)
    c0 -= numpy.einsum('iajb,iajb->', tmp1_ba, y1_ba)

    # [3] Same F_vv intermediates, with the first occupied index left free.
    c1_a[oa, oa] += numpy.einsum('iajb,kajb->ik', tmp1_aa, y1_aa)
    c1_a[oa, oa] += numpy.einsum('iajb,kajb->ik', tmp1_ab, y1_ab)
    c1_b[ob, ob] += numpy.einsum('iajb,kajb->ik', tmp1_bb, y1_bb)
    c1_b[ob, ob] += numpy.einsum('iajb,kajb->ik', tmp1_ba, y1_ba)
    c1_a[va, va] -= numpy.einsum('iajb,iajc->bc', tmp1_aa, y1_aa)
    c1_a[va, va] -= numpy.einsum('iajb,iajc->bc', tmp1_ba, y1_ba)
    c1_b[vb, vb] -= numpy.einsum('iajb,iajc->bc', tmp1_bb, y1_bb)
    c1_b[vb, vb] -= numpy.einsum('iajb,iajc->bc', tmp1_ab, y1_ab)

    # [4] y[i,a,j,b] = sum_k F[i,k] T_bar[k,a,j,b].
    y1_aa = numpy.einsum('ik,kajb->iajb', fock_hfa[oa, oa], tmp1_bar_aa)
    y1_ab = numpy.einsum('ik,kajb->iajb', fock_hfa[oa, oa], tmp1_bar_ab)
    y1_bb = numpy.einsum('ik,kajb->iajb', fock_hfb[ob, ob], tmp1_bar_bb)
    y1_ba = numpy.einsum('ik,kajb->iajb', fock_hfb[ob, ob], tmp1_bar_ba)
    c1_a[oa, oa] -= numpy.einsum('iajb,iakb->jk', tmp1_aa, y1_aa)
    c1_a[oa, oa] -= numpy.einsum('iajb,iakb->jk', tmp1_ba, y1_ba)
    c1_b[ob, ob] -= numpy.einsum('iajb,iakb->jk', tmp1_bb, y1_bb)
    c1_b[ob, ob] -= numpy.einsum('iajb,iakb->jk', tmp1_ab, y1_ab)
    c0 += numpy.einsum('iajb,iajb->', tmp1_aa, y1_aa)
    c0 += numpy.einsum('iajb,iajb->', tmp1_ab, y1_ab)
    c0 += numpy.einsum('iajb,iajb->', tmp1_bb, y1_bb)
    c0 += numpy.einsum('iajb,iajb->', tmp1_ba, y1_ba)

    # [5] Occupied density-like intermediates -> general/occupied.
    y1_a = numpy.einsum('iajb,kajb->ik', tmp1_aa, tmp1_bar_aa)
    y1_a += numpy.einsum('iajb,kajb->ik', tmp1_ab, tmp1_bar_ab)
    y1_b = numpy.einsum('iajb,kajb->ik', tmp1_bb, tmp1_bar_bb)
    y1_b += numpy.einsum('iajb,kajb->ik', tmp1_ba, tmp1_bar_ba)
    c1_a[:, oa] -= numpy.einsum('ip,ik->pk', fock_hfa[oa, :], y1_a)
    c1_b[:, ob] -= numpy.einsum('ip,ik->pk', fock_hfb[ob, :], y1_b)

    # [6] Reuse the F_oo intermediates from [4]; free second virtual index.
    c1_a[va, va] += numpy.einsum('iajb,iajc->bc', tmp1_aa, y1_aa)
    c1_a[va, va] += numpy.einsum('iajb,iajc->bc', tmp1_ba, y1_ba)
    c1_b[vb, vb] += numpy.einsum('iajb,iajc->bc', tmp1_bb, y1_bb)
    c1_b[vb, vb] += numpy.einsum('iajb,iajc->bc', tmp1_ab, y1_ab)

    # [7] Same F_oo intermediates; free first virtual index.
    c1_a[va, va] += numpy.einsum('iajb,icjb->ac', tmp1_aa, y1_aa)
    c1_a[va, va] += numpy.einsum('iajb,icjb->ac', tmp1_ab, y1_ab)
    c1_b[vb, vb] += numpy.einsum('iajb,icjb->ac', tmp1_bb, y1_bb)
    c1_b[vb, vb] += numpy.einsum('iajb,icjb->ac', tmp1_ba, y1_ba)

    # # [8] F_vv intermediates -> virtual/virtual, followed by general/virtual.
    # y1_aa = numpy.einsum('ac,icjb->iajb', fock_hfa[va, va], tmp1_bar_aa)
    # y1_ab = numpy.einsum('ac,icjb->iajb', fock_hfa[va, va], tmp1_bar_ab)
    # y1_bb = numpy.einsum('ac,icjb->iajb', fock_hfb[vb, vb], tmp1_bar_bb)
    # y1_ba = numpy.einsum('ac,icjb->iajb', fock_hfb[vb, vb], tmp1_bar_ba)

    y1_a = numpy.einsum('iajb,icjb->ac', tmp1_aa, tmp1_bar_aa)
    y1_a += numpy.einsum('iajb,icjb->ac', tmp1_ab, tmp1_bar_ab)
    y1_b = numpy.einsum('iajb,icjb->ac', tmp1_bb, tmp1_bar_bb)
    y1_b += numpy.einsum('iajb,icjb->ac', tmp1_ba, tmp1_bar_ba)
    c1_a[:, va] -= numpy.einsum('ap,ac->pc', fock_hfa[va, :], y1_a)
    c1_b[:, vb] -= numpy.einsum('ap,ac->pc', fock_hfb[vb, :], y1_b)
    return c0, c1_a, c1_b


# Restricted-open-shell setup and state validation

def _is_spin_energy_pair(mo_energy):
    """Return True for an explicit ``(epsilon_alpha, epsilon_beta)`` pair."""
    return (isinstance(mo_energy, (tuple, list)) and
            len(mo_energy) == 2 and
            numpy.asarray(mo_energy[0]).ndim == 1 and
            numpy.asarray(mo_energy[1]).ndim == 1)


def _as_rohf_occ(mp):
    """Return the 1-D ROHF spatial-orbital occupation vector.

    High-spin convention used here:
      occupation 2 -> alpha + beta occupied (closed)
      occupation 1 -> alpha occupied only (open)
      occupation 0 -> unoccupied in alpha and beta
    """
    occ = getattr(mp, 'mo_occ', None)
    if occ is None:
        occ = mp._scf.mo_occ
    occ = numpy.asarray(occ, dtype=float)
    if occ.ndim != 1:
        raise TypeError(
            'ROOBMP2 requires an ROHF-like 1-D mo_occ array with values 2, 1, 0; '
            f'got shape {occ.shape}.'
        )
    if not numpy.all(numpy.isclose(occ[:, None], [0.0, 1.0, 2.0],
                                  rtol=0.0, atol=1e-10).any(axis=1)):
        raise ValueError('ROOBMP2 mo_occ must contain only 0, 1, or 2.')
    return numpy.rint(occ)


def _spin_occ_from_rohf(mo_occ):
    """Convert ROHF spatial occupations to alpha/beta 0/1 occupations."""
    mo_occ = numpy.asarray(mo_occ)
    occa = (mo_occ > 0).astype(float)
    occb = (mo_occ == 2).astype(float)
    return occa, occb


def _ordered_rohf_orbitals(mo_coeff, mo_occ, *energies):
    """Order orbitals as closed, open, virtual.

    The inherited spin-block algebra assumes that occupied orbitals are
    contiguous.  ROHF itself does not require the occupations to be stored in
    this order, so we make the ordering explicit before entering the kernel.
    """
    idx_c = numpy.where(mo_occ == 2)[0]
    idx_o = numpy.where(mo_occ == 1)[0]
    idx_v = numpy.where(mo_occ == 0)[0]
    order = numpy.concatenate((idx_c, idx_o, idx_v))
    coeff = numpy.asarray(mo_coeff)[:, order].copy()
    occ = numpy.asarray(mo_occ)[order].copy()
    eps = [numpy.asarray(e)[order].copy() for e in energies]
    return (coeff, occ, *eps)


def _fill_rohf_occ(mo_energy, mo_energy_a, ncore, nopen):
    """Assign ROHF occupations following the PySCF ROHF selection logic."""
    mo_energy = numpy.asarray(mo_energy)
    mo_energy_a = numpy.asarray(mo_energy_a)
    occ = numpy.zeros(mo_energy.size, dtype=float)

    core_sort = numpy.argsort(mo_energy)
    core_idx = core_sort[:ncore]
    occ[core_idx] = 2.0

    if nopen:
        remaining = core_sort[ncore:]
        open_sort = numpy.argsort(mo_energy_a[remaining])
        open_idx = remaining[open_sort[:nopen]]
        occ[open_idx] = 1.0
    return occ


def _roothaan_fock_mo(fock_a, fock_b, mo_occ):
    """Build Roothaan's effective ROHF Fock matrix in an orthonormal MO basis.

    Block prescription (C=closed, O=open, V=virtual), with
    Fc=(Fa+Fb)/2:

        C-C Fc, C-O Fb, C-V Fc
        O-C Fb, O-O Fc, O-V Fa
        V-C Fc, V-O Fa, V-V Fc

    This is the restricted-open-shell constraint: only one spatial-orbital
    rotation is generated for both spins.
    """
    fa = numpy.asarray(fock_a)
    fb = numpy.asarray(fock_b)
    occ = numpy.asarray(mo_occ)

    if fa.ndim != 2 or fa.shape[0] != fa.shape[1] or fb.shape != fa.shape:
        raise ValueError('Alpha and beta Focks must be equal-sized square matrices.')
    if occ.ndim != 1 or occ.size != fa.shape[0]:
        raise ValueError('ROHF occupation and Fock dimensions are inconsistent.')
    if not numpy.all(numpy.isfinite(fa)) or not numpy.all(numpy.isfinite(fb)):
        raise FloatingPointError('A correlated spin Fock contains NaN or infinity.')

    # The inherited BCH implementation targets real orbitals, but explicitly
    # Hermitize here to remove harmless numerical asymmetry before ``eigh``.
    fa = 0.5 * (fa + fa.conj().T)
    fb = 0.5 * (fb + fb.conj().T)

    occa, occb = _spin_occ_from_rohf(occ)
    dma_mo = numpy.diag(occa)
    dmb_mo = numpy.diag(occb)
    s_mo = numpy.eye(occ.size)

    # All arguments are in the current orthonormal MO basis.  The occupation
    # matrices are determinant projectors, not correlated fractional RDMs.
    fock_ro = get_roothaan_fock((fa, fb), (dma_mo, dmb_mo), s_mo)
    return numpy.asarray(fock_ro)


def _roothaan_orbital_residual(fock_ro, mo_occ):
    """Frobenius norm of the independent C-O, C-V, and O-V blocks."""
    fock_ro = numpy.asarray(fock_ro)
    occ = numpy.asarray(mo_occ)
    c = numpy.flatnonzero(occ == 2)
    o = numpy.flatnonzero(occ == 1)
    v = numpy.flatnonzero(occ == 0)

    residual2 = 0.0
    for rows, cols in ((c, o), (c, v), (o, v)):
        if rows.size and cols.size:
            block = fock_ro[numpy.ix_(rows, cols)]
            residual2 += numpy.linalg.norm(block)**2
    return float(numpy.sqrt(residual2))


def _rohf_spin_energies(mp, coeff, occ):
    """Compute alpha/beta diagonal Fock energies for common ROHF orbitals."""
    dm = mp._scf.make_rdm1(coeff, occ)
    hcore = mp._scf.get_hcore(mp.mol)
    vhf = mp._scf.get_veff(mp.mol, dm)
    fock_a_ao = hcore + vhf[0]
    fock_b_ao = hcore + vhf[1]
    eps_a = numpy.einsum(
        'pi,pi->i', coeff.conj(), fock_a_ao.dot(coeff)).real
    eps_b = numpy.einsum(
        'pi,pi->i', coeff.conj(), fock_b_ao.dot(coeff)).real
    return eps_a, eps_b


def _rohf_reference_fock_error(mp, coeff, occ, fock_a_mo, fock_b_mo):
    """Compare inherited integral-loop Focks with PySCF ROHF spin Focks."""
    dm = mp._scf.make_rdm1(coeff, occ)
    hcore = mp._scf.get_hcore(mp.mol)
    vhf = mp._scf.get_veff(mp.mol, dm)
    fock_a_ref = coeff.conj().T.dot(hcore + vhf[0]).dot(coeff)
    fock_b_ref = coeff.conj().T.dot(hcore + vhf[1]).dot(coeff)
    return (float(numpy.linalg.norm(fock_a_mo - fock_a_ref)),
            float(numpy.linalg.norm(fock_b_mo - fock_b_ref)))


def _checked_amplitude(numerator, denominator, threshold, channel):
    """Safely form one MP1 amplitude and report a singular denominator."""
    if not numpy.isfinite(denominator) or abs(denominator) <= threshold:
        raise FloatingPointError(
            '%s denominator is singular or too small: %.16e '
            '(threshold %.3e).' % (channel, denominator, threshold))
    value = numerator / denominator
    if not numpy.isfinite(value):
        raise FloatingPointError('%s amplitude is NaN or infinite.' % channel)
    return value


def _resolve_amplitude_factor(mp):
    """Choose the amplitude factor once, without mutating the user's setting."""
    factor = mp.ampf
    if factor is None:
        return 1.0 if mp.second_order else 0.5
    if (not numpy.isscalar(factor) or not numpy.isrealobj(factor) or
            not numpy.isfinite(factor)):
        raise ValueError('ampf must be None or a finite real scalar.')
    return float(factor)


def _initial_rohf_orbitals(mp, mo_energy=None, mo_coeff=None):
    """Prepare a common ROHF orbital set and spin-resolved denominators."""
    if getattr(mp, 'mom', False):
        raise NotImplementedError('MOM/root-following is not implemented for ROOBMP2 yet.')

    frozen = getattr(mp, 'frozen', None)
    frozen_is_zero = isinstance(frozen, (int, numpy.integer)) and frozen == 0
    if frozen is not None and not frozen_is_zero:
        raise NotImplementedError(
            'This ROOBMP2 prototype currently supports frozen=None or frozen=0 only. '
            'Shared frozen-core support should be added after the basic method is validated.'
        )

    if mo_coeff is None:
        candidate = getattr(mp, 'mo_coeff', None)
        mo_coeff = mp._scf.mo_coeff if candidate is None else candidate
    coeff = numpy.asarray(mo_coeff)
    if coeff.ndim != 2:
        raise TypeError(
            'ROOBMP2 requires one common 2-D spatial MO coefficient matrix from ROHF; '
            f'got shape {coeff.shape}.'
        )

    occ = _as_rohf_occ(mp)
    if coeff.shape[1] != occ.size:
        raise ValueError('mo_coeff and mo_occ have inconsistent numbers of orbitals.')
    if numpy.iscomplexobj(coeff) and numpy.max(numpy.abs(coeff.imag)) > 1e-12:
        raise NotImplementedError(
            'Complex orbitals are not supported by the inherited BCH contractions.')

    overlap = mp._scf.get_ovlp()
    metric = coeff.conj().T.dot(overlap).dot(coeff)
    orth_error = numpy.linalg.norm(metric - numpy.eye(occ.size))
    if orth_error > 1e-8:
        raise ValueError(
            'The common ROHF orbitals are not S-orthonormal: '
            '||C^H S C - I|| = %.3e' % orth_error)

    # This file implements the alpha-majority high-spin convention.
    nalpha = int(numpy.count_nonzero(occ > 0))
    nbeta = int(numpy.count_nonzero(occ == 2))
    nopen = int(numpy.count_nonzero(occ == 1))
    if (nalpha, nbeta) != mp.mol.nelec:
        raise ValueError('ROHF occupations must match mol.nelec.')
    if mp.mol.spin < 0:
        raise NotImplementedError(
            'Only alpha-majority high-spin ROHF references are supported.')
    if nalpha - nbeta != nopen or nopen != mp.mol.spin:
        raise ValueError(
            'In the high-spin convention, number of singly occupied orbitals '
            'must equal mol.spin = N_alpha - N_beta; got nopen=%d, spin=%d.' %
            (nopen, mp.mol.spin))

    # An explicit spin-energy pair changes MP denominators only.  Roothaan
    # effective energies are still taken from the ROHF reference for ordering.
    if _is_spin_energy_pair(mo_energy):
        source_e = mp._scf.mo_energy
        eps_a = numpy.asarray(mo_energy[0], dtype=float).copy()
        eps_b = numpy.asarray(mo_energy[1], dtype=float).copy()
    else:
        if mo_energy is None:
            candidate_e = getattr(mp, 'mo_energy', None)
            source_e = mp._scf.mo_energy if candidate_e is None else candidate_e
        else:
            source_e = mo_energy
        eps_a = eps_b = None

    eps_ro = numpy.asarray(source_e, dtype=float).copy()
    if eps_ro.ndim != 1:
        raise TypeError('ROOBMP2 requires a 1-D ROHF mo_energy array.')
    if eps_ro.size != occ.size:
        raise ValueError('ROHF energy and occupation dimensions are inconsistent.')

    # PySCF ROHF may tag the Roothaan energies with spin-resolved diagonal
    # Fock energies.  If tags are unavailable, explicitly construct Fa/Fb;
    # using the same Roothaan eigenvalue for both spins is not valid in general.
    if eps_a is None:
        eps_a_src = getattr(source_e, 'mo_ea', None)
        eps_b_src = getattr(source_e, 'mo_eb', None)
        if eps_a_src is not None and eps_b_src is not None:
            eps_a = numpy.asarray(eps_a_src, dtype=float).copy()
            eps_b = numpy.asarray(eps_b_src, dtype=float).copy()
        else:
            eps_a, eps_b = _rohf_spin_energies(mp, coeff, occ)

    if eps_a.shape != eps_ro.shape or eps_b.shape != eps_ro.shape:
        raise ValueError('Alpha/beta orbital energies have inconsistent dimensions.')
    if not (numpy.all(numpy.isfinite(eps_ro)) and
            numpy.all(numpy.isfinite(eps_a)) and
            numpy.all(numpy.isfinite(eps_b))):
        raise FloatingPointError('Initial ROHF orbital energies contain NaN or infinity.')

    coeff, occ, eps_ro, eps_a, eps_b = _ordered_rohf_orbitals(
        coeff, occ, eps_ro, eps_a, eps_b
    )
    return coeff, occ, eps_ro, eps_a, eps_b


def _clear_results(mp):
    """Invalidate computed outputs, keeping orbital guesses for a restart."""
    mp.converged = False
    mp._final_state_available = False
    mp.cycles = 0
    for name in (
            't2', 't2_metadata', '_t2_signature', 'rdm1_source',
            'e_corr', 'e_tot_obmp2', 'denominator_energies', 'ampf_used',
            'ss_det', 'multiplicity_det', 'fock_corr_a', 'fock_corr_b',
            'fock_ro', 'fock_hf_a', 'fock_hf_b', 'bch1_fock_correction',
            'bch2_fock_correction', 'bch1_scalar_correction',
            'bch2_scalar_correction', 'scalar_constant', 'orbital_residual',
            'denominator_residual', 'min_abs_denom', 'reference_fock_error',
            'reference_energy_error'):
        setattr(mp, name, None)


def _amplitude_signature(mp):
    """Bind saved T2 to its MO basis, occupations and actual denominators.

    Hashing the tensor values also catches replacement with an old, same-shaped
    T2 or in-place edits.  This is an in-memory consistency check, not a
    portable checkpoint or a complete Hamiltonian fingerprint.
    """
    digest = hashlib.sha256()
    arrays = (mp.mo_coeff, mp.mo_occ, *mp.denominator_energies, *mp.t2)
    for value in arrays:
        array = numpy.ascontiguousarray(value)
        digest.update(repr((array.shape, array.dtype.str)).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def int_transform_ss(eri_ao, mo_coeff):
    nao = mo_coeff.shape[0]
    nmo = mo_coeff.shape[1]
    eri_mo = numpy.dot(mo_coeff.T, eri_ao.reshape(nao,-1))
    eri_mo = numpy.dot(eri_mo.reshape(-1,nao), mo_coeff)
    eri_mo = eri_mo.reshape(nmo,nao,nao,nmo).transpose(1,0,3,2)
    eri_mo = numpy.dot(mo_coeff.T, eri_mo.reshape(nao,-1))
    eri_mo = numpy.dot(eri_mo.reshape(-1,nao), mo_coeff)
    eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
    return eri_mo

def int_transform_os(eri_ao, mo_coeff_s1, mo_coeff_s2):
    nao = mo_coeff_s1.shape[0]
    nmo = mo_coeff_s1.shape[1]
    eri_mo = numpy.dot(mo_coeff_s1.T, eri_ao.reshape(nao,-1))
    eri_mo = numpy.dot(eri_mo.reshape(-1,nao), mo_coeff_s2)
    eri_mo = eri_mo.reshape(nmo,nao,nao,nmo).transpose(1,0,3,2)
    eri_mo = numpy.dot(mo_coeff_s1.T, eri_mo.reshape(nao,-1))
    eri_mo = numpy.dot(eri_mo.reshape(-1,nao), mo_coeff_s2)
    eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
    return eri_mo

def get_nocc(mp):
    """Return (n_alpha_occ, n_beta_occ) from a 1-D ROHF occupation vector."""
    occ = _as_rohf_occ(mp)
    nocca = int(numpy.count_nonzero(occ > 0))
    noccb = int(numpy.count_nonzero(occ == 2))

    frozen = getattr(mp, 'frozen', None)
    if frozen is None or (isinstance(frozen, (int, numpy.integer)) and frozen == 0):
        return nocca, noccb
    raise NotImplementedError('ROOBMP2 currently supports frozen=None or frozen=0 only.')


def get_nmo(mp):
    """Return equal alpha/beta MO dimensions for common ROHF spatial orbitals."""
    occ = _as_rohf_occ(mp)
    nmo = int(occ.size)
    frozen = getattr(mp, 'frozen', None)
    if frozen is None or (isinstance(frozen, (int, numpy.integer)) and frozen == 0):
        return nmo, nmo
    raise NotImplementedError('ROOBMP2 currently supports frozen=None or frozen=0 only.')


def get_frozen_mask(mp):
    """Return shared alpha/beta masks for the common spatial orbital set."""
    occ = _as_rohf_occ(mp)
    frozen = getattr(mp, 'frozen', None)
    if frozen is None or (isinstance(frozen, (int, numpy.integer)) and frozen == 0):
        mask = numpy.ones(occ.size, dtype=bool)
        return mask.copy(), mask.copy()
    raise NotImplementedError('ROOBMP2 currently supports frozen=None or frozen=0 only.')


def mom_reorder(mp, mo_coeff):
    raise NotImplementedError('MOM/root-following is not implemented for ROOBMP2 yet.')


def mom_select(mp, mo_coeff_init, mo_coeff_new):
    raise NotImplementedError('MOM/root-following is not implemented for ROOBMP2 yet.')


def make_rdm1(mp, use_t2=False, use_ao=True, allow_unconverged=False, **kwargs):
    mp.rdm1_source = None
    if 'ao_repr' in kwargs:
        use_ao = kwargs.pop('ao_repr')
    if kwargs:
        raise TypeError('Unsupported make_rdm1 keyword(s): %s' %
                        ', '.join(sorted(kwargs)))

    coeff = numpy.asarray(mp.mo_coeff)
    occ = _as_rohf_occ(mp)
    if coeff.ndim != 2 or coeff.shape[1] != occ.size:
        raise TypeError('ROOBMP2 requires one common 2-D C and one 1-D mo_occ.')
    occa, occb = _spin_occ_from_rohf(occ)
    dtype = numpy.result_type(coeff.dtype, numpy.float64)
    dm_a = numpy.diag(occa).astype(dtype)
    dm_b = numpy.diag(occb).astype(dtype)
    source = 'method determinant only'

    if use_t2:
        if not getattr(mp, '_final_state_available', False):
            raise RuntimeError('Run ROOBMP2.kernel(with_t2=True) before the T2 density.')
        t2 = getattr(mp, 't2', None)
        if t2 is None:
            raise RuntimeError(
                'Final T2 amplitudes were not retained. Run kernel(with_t2=True).')
        nocca, noccb = mp.get_nocc()
        nmoa, nmob = mp.get_nmo()
        nvira, nvirb = nmoa - nocca, nmob - noccb
        if nmoa != occ.size or nmob != occ.size:
            raise ValueError('T2 density requires the full common MO space.')
        if (not numpy.array_equal(occa, (numpy.arange(occ.size) < nocca)) or
                not numpy.array_equal(occb, (numpy.arange(occ.size) < noccb))):
            raise ValueError('Stored T2 requires the final closed/open/virtual ordering.')
        shapes = ((nocca, nocca, nvira, nvira),
                  (nocca, noccb, nvira, nvirb),
                  (noccb, noccb, nvirb, nvirb))
        if not isinstance(t2, (tuple, list)) or len(t2) != 3:
            raise ValueError('Expected T2 = (t2aa, t2ab, t2bb) in (i,j,a,b) order.')
        t2 = tuple(numpy.asarray(t) for t in t2)
        for channel, amplitude, shape in zip(('aa', 'ab', 'bb'), t2, shapes):
            if amplitude.shape != shape:
                raise ValueError('T2 %s has shape %s; expected %s.' %
                                 (channel, amplitude.shape, shape))
            if not numpy.all(numpy.isfinite(amplitude)):
                raise FloatingPointError('T2 %s contains NaN or infinity.' % channel)

        metadata = getattr(mp, 't2_metadata', None)
        if (metadata is None or
                metadata.get('run_id') != getattr(mp, '_kernel_run_id', None) or
                getattr(mp, 'denominator_energies', None) is None or
                getattr(mp, '_t2_signature', None) != _amplitude_signature(mp)):
            raise RuntimeError(
                'Stored T2 does not match this run/orbital basis/occupations/'
                'denominators, or its values were changed. Run kernel(with_t2=True) again.')
        if not metadata['converged'] and not allow_unconverged:
            raise RuntimeError(
                'ROOBMP2 did not converge. Rerun to convergence, or explicitly '
                'use allow_unconverged=True for a last-iterate diagnostic only.')

        doo, dvv = _gamma1_intermediates(mp, t2)
        dtype = numpy.result_type(dtype, *(t.dtype for t in t2))
        dm_a = dm_a.astype(dtype)
        dm_b = dm_b.astype(dtype)
        # Alpha: occupied C+O, virtual V.  Beta: occupied C, virtual O+V.
        # The 1/2 here Hermitizes each density block; the same-spin contraction
        # factors are already in _gamma1_intermediates.  Do not apply them twice.
        for dm, nocc, oo, vv in zip((dm_a, dm_b), (nocca, noccb), doo, dvv):
            dm[:nocc, :nocc] += 0.5 * (oo + oo.conj().T)
            dm[nocc:, nocc:] += 0.5 * (vv + vv.conj().T)
        source = 'unrelaxed O(T2^2) density from final RO-OBMP2 amplitudes'
        if not metadata['converged']:
            source += ' (UNCONVERGED last iterate)'

    mp.rdm1_source = source
    if use_ao:
        dm_a = coeff @ dm_a @ coeff.conj().T
        dm_b = coeff @ dm_b @ coeff.conj().T
    return numpy.asarray((dm_a, dm_b))


def _gamma1_intermediates(mp, t2):
    t2aa, t2ab, t2bb = t2
    dooa  = lib.einsum('imef,jmef->ij', t2aa.conj(), t2aa) *-.5
    dooa -= lib.einsum('imef,jmef->ij', t2ab.conj(), t2ab)
    doob  = lib.einsum('imef,jmef->ij', t2bb.conj(), t2bb) *-.5
    doob -= lib.einsum('mief,mjef->ij', t2ab.conj(), t2ab)

    dvva  = lib.einsum('mnae,mnbe->ba', t2aa.conj(), t2aa) * .5
    dvva += lib.einsum('mnae,mnbe->ba', t2ab.conj(), t2ab)
    dvvb  = lib.einsum('mnae,mnbe->ba', t2bb.conj(), t2bb) * .5
    dvvb += lib.einsum('mnea,mneb->ba', t2ab.conj(), t2ab)
    return ((dooa, doob), (dvva, dvvb))

def make_fc(mp, dm0, it=None, R_reslv=None, hfc_nuc=None, verbose=None):
    '''The contribution of Fermi-contact term and dipole-dipole interactions'''
    #log = logger.new_logger(hfcobj, verbose)
    mol = mp.mol
    if hfc_nuc is None:
        hfc_nuc = range(mol.natm)
    if isinstance(dm0, numpy.ndarray) and dm0.ndim == 2: # RHF DM
        return numpy.zeros((3,3))

    dma, dmb = dm0
    spindm = dma - dmb
    effspin = mol.spin * .5

    #if R_reslv is not None:
    #    mo_coeff = mp.mo_coeff
    #    nocca, noccb = mp.get_nocc()
    #    dma_mo, dmb_mo = mp.make_rdm1(use_t2=True,use_ao=False)
    #    spinnocca, U = scipy.linalg.eigh(dma_mo)
    #    spinmoa = numpy.matmul(mo_coeff[0], U)
    #    nao = mo_coeff[0].shape[0]
    #    tmp = numpy.zeros((nao,nao))
    #    for mu in range(nao):
    #        for nu in range(nao):
    #            tmp[mu,nu] = spinmoa[] * spinmoa[mu,nocca-1] * spinmoa[nu,nocca-1]
    #    np = 1000
    #    dz = (R_reslv[1] - R_reslv[0])/np
    #    fname = "spinden_somo"+str(it)+".dat"
    #    with open(fname, 'w') as f:
    #        for i in range(np):
    #            r = i*dz + R_reslv[0]
    #            coords = [[0,0,r]]
    #            h1fc = _get_integrals_fc_Rreslv(mol, coords)
    #            fc = numpy.einsum('ij,ji', h1fc, tmp)
    #            f.write(" %8.6f %8.6f \n"  %(r, fc))


    e_gyro = .5 * nist.G_ELECTRON
    nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
    au2MHz = nist.HARTREE2J / nist.PLANCK * 1e-6
    fac = nist.ALPHA**2 / 2 / effspin * e_gyro * au2MHz

    hfc = []
    for i, atm_id in enumerate(hfc_nuc):
        nuc_gyro = get_nuc_g_factor(mol.atom_symbol(atm_id)) * nuc_mag
        #h1 = _get_integrals_fcdip(mol, atm_id)
        #fcsd = numpy.einsum('xyij,ji->xy', h1, spindm)

        h1fc = _get_integrals_fc(mol, atm_id)
        fc = numpy.einsum('ij,ji', h1fc, spindm)

        #sd = fcsd + numpy.eye(3) * fc

        print('FC of atom %d :'%atm_id, '%8.6f (in MHz)' %(2*fac * nuc_gyro * fc))
        #if hfcobj.verbose >= logger.INFO:
        #    _write(hfcobj, align(fac*nuc_gyro*sd)[0], 'SD of atom %d (in MHz)' % atm_id)
        #hfc.append(fac * nuc_gyro * fcsd)
    #return numpy.asarray(hfc)

def _get_integrals_fcdip(mol, atm_id):
    '''AO integrals for FC + Dipole-dipole'''
    nao = mol.nao
    with mol.with_rinv_origin(mol.atom_coord(atm_id)):
        # Note the fermi-contact part is different to the fermi-contact
        # operator in SSC.  FC here is associated to the the integrals of
        # (\nabla \nabla 1/r), which includes the contribution of Poisson
        # equation, 4\pi rho.  Factor 4.\pi/3 is used in the Fermi contact
        # contribution.  In SSC, the factor of FC part is -8\pi/3.
        ipipv = mol.intor('int1e_ipiprinv', 9).reshape(3,3,nao,nao)
        ipvip = mol.intor('int1e_iprinvip', 9).reshape(3,3,nao,nao)
        h1ao = ipipv + ipvip  # (nabla i | r/r^3 | j)
        h1ao = h1ao + h1ao.transpose(0,1,3,2)
        trace = h1ao[0,0] + h1ao[1,1] + h1ao[2,2]
        idx = numpy.arange(3)
        h1ao[idx,idx] -= trace
    return h1ao

def _get_integrals_fc(mol, atm_id):
    '''AO integrals for Fermi contact term'''
    coords = mol.atom_coord(atm_id).reshape(1, 3)
    ao = mol.eval_gto('GTOval', coords)
    return 4*numpy.pi/3 * numpy.einsum('ip,iq->pq', ao, ao)

def _get_integrals_fc_Rreslv(mol, coords):
    '''AO integrals for Fermi contact term'''
    ao = mol.eval_gto('GTOval', coords)
    return 4*numpy.pi/3 * numpy.einsum('ip,iq->pq', ao, ao)


class ROOBMP2(OBMP2):
    """High-spin RO-constrained OBMP2 with optional second BCH.

    Example::

        pt = ROOBMP2(rohf_reference)
        pt.second_order = True
        pt.ampf = 1.0   # or None for the order-dependent default
        e_corr = pt.kernel()

    ``ampf`` scales T consistently everywhere.  The default ``None`` means
    0.5 for BCH1 and 1.0 for BCH1+BCH2; an explicit 0.0 tests the ROHF limit.
    BCH2 adds the scalar and one-body double-commutator contractions.
    kernel(with_t2=True) retains the final scaled T2 for make_rdm1(use_t2=True),
    an unrelaxed O(T2^2) property density.  No T1, fully relaxed response density,
    or general RO perturbation-theory derivation is supplied.
    """

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
        self.ampf = None
        self._kernel_run_id = 0
        _clear_results(self)
        self._keys.update({
            'grad_thresh', 'denom_thresh', 'fock_check_thresh',
            'energy_check_thresh', 'energy_consistency_thresh', 'break_sym',
            'eval_fc', 'converged', 'bch_order', 'spin_square_source', 'rdm1_source',
            't2_metadata',
        })

    @property
    def bch_order(self):
        return 2 if self.second_order else 1

    @bch_order.setter
    def bch_order(self, value):
        if isinstance(value, (bool, numpy.bool_)) or value not in (1, 2):
            raise ValueError('bch_order must be 1 or 2.')
        self.second_order = (value == 2)

    @property
    def e_tot(self):
        return self.e_tot_obmp2

    spin_square_source = 'method determinant only'

    def spin_square(self):
        """Spin of the final common-orbital determinant, not correlated spin."""
        if not self._final_state_available:
            raise RuntimeError('Run ROOBMP2.kernel() before requesting final determinant spin.')
        c = numpy.asarray(self.mo_coeff)
        occ = _as_rohf_occ(self)
        occupied = (c[:,occ>0], c[:,occ==2])
        return uhf_spin_square(occupied, self._scf.get_ovlp())

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    int_transform_ss = int_transform_ss
    int_transform_os = int_transform_os
    mom_select = mom_select
    mom_reorder = mom_reorder
    break_sym = False
    grad_thresh = 1e-6
    denom_thresh = 1e-12
    fock_check_thresh = 1e-8
    energy_check_thresh = 1e-8
    energy_consistency_thresh = 1e-6
    converged = False
    #use_t2 = False

    @lib.with_doc(OBMP2.kernel.__doc__)
    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2, self.verbose)

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return _make_eris(self, mo_coeff, verbose=self.verbose)

    make_rdm1 = make_rdm1
    #make_rdm2 = make_rdm2
    make_fc = make_fc
    eval_fc = False

    def nuc_grad_method(self):
        raise NotImplementedError(
            'Analytic nuclear gradients are not implemented for ROOBMP2. '
            'UMP2 gradients are not valid for this restricted-open-shell method.'
        )

OBMP2 = ROOBMP2

#from pyscf import scf
#scf.uhf.UHF.MP2 = lib.class_as_method(MP2)


class _ChemistsERIs(_OBMP2ChemistsERIs):
    def __init__(self, mp, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        coeff = numpy.asarray(mo_coeff)
        if coeff.ndim != 2:
            raise TypeError('ROOBMP2 ERIs require one common 2-D spatial MO matrix.')
        moidxa, moidxb = mp.get_frozen_mask()
        if not numpy.array_equal(moidxa, moidxb):
            raise RuntimeError('ROOBMP2 requires a shared frozen-orbital mask.')
        coeff = coeff[:, moidxa]
        self.mo_coeff = (coeff.copy(), coeff.copy())

def _make_eris(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    log = logger.new_logger(mp, verbose)
    time0 = (time.process_time(), time.perf_counter())
    eris = _ChemistsERIs(mp, mo_coeff)

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb
    nao = eris.mo_coeff[0].shape[0]
    nmo_pair = nmoa * (nmoa+1) // 2
    nao_pair = nao * (nao+1) // 2
    mem_incore = (nao_pair**2 + nmo_pair**2) * 8/1e6
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mp.max_memory-mem_now)

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    orboa = moa[:,:nocca]
    orbob = mob[:,:noccb]
    orbva = moa[:,nocca:]
    orbvb = mob[:,noccb:]

    if (mp.mol.incore_anyway or
        (mp._scf._eri is not None and mem_incore+mem_now < mp.max_memory)):
        log.debug('transform (ia|jb) incore')
        if callable(ao2mofn):
            eris.ovov = ao2mofn((orboa,orbva,orboa,orbva)).reshape(nocca*nvira,nocca*nvira)
            eris.ovOV = ao2mofn((orboa,orbva,orbob,orbvb)).reshape(nocca*nvira,noccb*nvirb)
            eris.OVOV = ao2mofn((orbob,orbvb,orbob,orbvb)).reshape(noccb*nvirb,noccb*nvirb)
        else:
            eris.ovov = ao2mo.general(mp._scf._eri, (orboa,orbva,orboa,orbva))
            eris.ovOV = ao2mo.general(mp._scf._eri, (orboa,orbva,orbob,orbvb))
            eris.OVOV = ao2mo.general(mp._scf._eri, (orbob,orbvb,orbob,orbvb))

    elif getattr(mp._scf, 'with_df', None):
        logger.warn(mp, 'UMP2 detected DF being used in the HF object. '
                    'MO integrals are computed based on the DF 3-index tensors.\n'
                    'It\'s recommended to use DF-UMP2 module.')
        log.debug('transform (ia|jb) with_df')
        eris.ovov = mp._scf.with_df.ao2mo((orboa,orbva,orboa,orbva))
        eris.ovOV = mp._scf.with_df.ao2mo((orboa,orbva,orbob,orbvb))
        eris.OVOV = mp._scf.with_df.ao2mo((orbob,orbvb,orbob,orbvb))

    else:
        log.debug('transform (ia|jb) outcore')
        eris.feri = lib.H5TmpFile()
        _ao2mo_ovov(mp, (orboa,orbva,orbob,orbvb), eris.feri,
                    max(2000, max_memory), log)
        eris.ovov = eris.feri['ovov']
        eris.ovOV = eris.feri['ovOV']
        eris.OVOV = eris.feri['OVOV']

    time1 = log.timer('Integral transformation', *time0)
    return eris

def _ao2mo_ovov(mp, orbs, feri, max_memory=2000, verbose=None):
    time0 = (time.process_time(), time.perf_counter())
    log = logger.new_logger(mp, verbose)
    orboa = numpy.asarray(orbs[0], order='F')
    orbva = numpy.asarray(orbs[1], order='F')
    orbob = numpy.asarray(orbs[2], order='F')
    orbvb = numpy.asarray(orbs[3], order='F')
    nao, nocca = orboa.shape
    noccb = orbob.shape[1]
    nvira = orbva.shape[1]
    nvirb = orbvb.shape[1]

    mol = mp.mol
    int2e = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, int2e, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    nbas = mol.nbas
    assert(nvira <= nao)
    assert(nvirb <= nao)

    ao_loc = mol.ao_loc_nr()
    dmax = max(4, min(nao/3, numpy.sqrt(max_memory*.95e6/8/(nao+nocca)**2)))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, dmax)
    dmax = max(x[2] for x in sh_ranges)
    eribuf = numpy.empty((nao,dmax,dmax,nao))
    ftmp = lib.H5TmpFile()
    disk = (nocca**2*(nao*(nao+dmax)/2+nvira**2) +
            noccb**2*(nao*(nao+dmax)/2+nvirb**2) +
            nocca*noccb*(nao**2+nvira*nvirb))
    log.debug('max_memory %s MB (dmax = %s) required disk space %g MB',
              max_memory, dmax, disk*8/1e6)

    fint = gto.moleintor.getints4c
    aa_blk_slices = []
    ab_blk_slices = []
    count_ab = 0
    count_aa = 0
    time1 = time0
    with lib.call_in_background(ftmp.__setitem__) as save:
        for ish0, ish1, ni in sh_ranges:
            for jsh0, jsh1, nj in sh_ranges:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]

                eri = fint(int2e, mol._atm, mol._bas, mol._env,
                           shls_slice=(0,nbas,ish0,ish1, jsh0,jsh1,0,nbas),
                           aosym='s1', ao_loc=ao_loc, cintopt=ao2mopt._cintopt,
                           out=eribuf)
                tmp_i = lib.ddot(orboa.T, eri.reshape(nao,(i1-i0)*(j1-j0)*nao))
                tmp_li = lib.ddot(orbob.T, tmp_i.reshape(nocca*(i1-i0)*(j1-j0),nao).T)
                tmp_li = tmp_li.reshape(noccb,nocca,(i1-i0),(j1-j0))
                save('ab/%d'%count_ab, tmp_li.transpose(1,0,2,3))
                ab_blk_slices.append((i0,i1,j0,j1))
                count_ab += 1

                if ish0 >= jsh0:
                    tmp_li = lib.ddot(orboa.T, tmp_i.reshape(nocca*(i1-i0)*(j1-j0),nao).T)
                    tmp_li = tmp_li.reshape(nocca,nocca,(i1-i0),(j1-j0))
                    save('aa/%d'%count_aa, tmp_li.transpose(1,0,2,3))

                    tmp_i = lib.ddot(orbob.T, eri.reshape(nao,(i1-i0)*(j1-j0)*nao))
                    tmp_li = lib.ddot(orbob.T, tmp_i.reshape(noccb*(i1-i0)*(j1-j0),nao).T)
                    tmp_li = tmp_li.reshape(noccb,noccb,(i1-i0),(j1-j0))
                    save('bb/%d'%count_aa, tmp_li.transpose(1,0,2,3))
                    aa_blk_slices.append((i0,i1,j0,j1))
                    count_aa += 1

                time1 = log.timer_debug1('partial ao2mo [%d:%d,%d:%d]' %
                                         (ish0,ish1,jsh0,jsh1), *time1)
    time1 = time0 = log.timer('mp2 ao2mo_ovov pass1', *time0)
    eri = eribuf = tmp_i = tmp_li = None

    fovov = feri.create_dataset('ovov', (nocca*nvira,nocca*nvira), 'f8',
                                chunks=(nvira,nvira))
    fovOV = feri.create_dataset('ovOV', (nocca*nvira,noccb*nvirb), 'f8',
                                chunks=(nvira,nvirb))
    fOVOV = feri.create_dataset('OVOV', (noccb*nvirb,noccb*nvirb), 'f8',
                                chunks=(nvirb,nvirb))
    occblk = int(min(max(nocca,noccb),
                     max(4, 250/nocca, max_memory*.9e6/8/(nao**2*nocca)/5)))

    def load_aa(h5g, nocc, i0, eri):
        if i0 < nocc:
            i1 = min(i0+occblk, nocc)
            for k, (p0,p1,q0,q1) in enumerate(aa_blk_slices):
                eri[:i1-i0,:,p0:p1,q0:q1] = h5g[str(k)][i0:i1]
                if p0 != q0:
                    dat = numpy.asarray(h5g[str(k)][:,i0:i1])
                    eri[:i1-i0,:,q0:q1,p0:p1] = dat.transpose(1,0,3,2)

    def load_ab(h5g, nocca, i0, eri):
        if i0 < nocca:
            i1 = min(i0+occblk, nocca)
            for k, (p0,p1,q0,q1) in enumerate(ab_blk_slices):
                eri[:i1-i0,:,p0:p1,q0:q1] = h5g[str(k)][i0:i1]

    def save(h5dat, nvir, i0, i1, dat):
        for i in range(i0, i1):
            h5dat[i*nvir:(i+1)*nvir] = dat[i-i0].reshape(nvir,-1)

    with lib.call_in_background(save) as bsave:
        with lib.call_in_background(load_aa) as prefetch:
            buf_prefecth = numpy.empty((occblk,nocca,nao,nao))
            buf = numpy.empty_like(buf_prefecth)
            load_aa(ftmp['aa'], nocca, 0, buf_prefecth)
            for i0, i1 in lib.prange(0, nocca, occblk):
                buf, buf_prefecth = buf_prefecth, buf
                prefetch(ftmp['aa'], nocca, i1, buf_prefecth)
                eri = buf[:i1-i0].reshape((i1-i0)*nocca,nao,nao)
                dat = _ao2mo.nr_e2(eri, orbva, (0,nvira,0,nvira), 's1', 's1')
                bsave(fovov, nvira, i0, i1,
                      dat.reshape(i1-i0,nocca,nvira,nvira).transpose(0,2,1,3))
                time1 = log.timer_debug1('pass2 ao2mo for aa [%d:%d]' % (i0,i1), *time1)

            buf_prefecth = numpy.empty((occblk,noccb,nao,nao))
            buf = numpy.empty_like(buf_prefecth)
            load_aa(ftmp['bb'], noccb, 0, buf_prefecth)
            for i0, i1 in lib.prange(0, noccb, occblk):
                buf, buf_prefecth = buf_prefecth, buf
                prefetch(ftmp['bb'], noccb, i1, buf_prefecth)
                eri = buf[:i1-i0].reshape((i1-i0)*noccb,nao,nao)
                dat = _ao2mo.nr_e2(eri, orbvb, (0,nvirb,0,nvirb), 's1', 's1')
                bsave(fOVOV, nvirb, i0, i1,
                      dat.reshape(i1-i0,noccb,nvirb,nvirb).transpose(0,2,1,3))
                time1 = log.timer_debug1('pass2 ao2mo for bb [%d:%d]' % (i0,i1), *time1)

        orbvab = numpy.asarray(numpy.hstack((orbva, orbvb)), order='F')
        with lib.call_in_background(load_ab) as prefetch:
            load_ab(ftmp['ab'], nocca, 0, buf_prefecth)
            for i0, i1 in lib.prange(0, nocca, occblk):
                buf, buf_prefecth = buf_prefecth, buf
                prefetch(ftmp['ab'], nocca, i1, buf_prefecth)
                eri = buf[:i1-i0].reshape((i1-i0)*noccb,nao,nao)
                dat = _ao2mo.nr_e2(eri, orbvab, (0,nvira,nvira,nvira+nvirb), 's1', 's1')
                bsave(fovOV, nvira, i0, i1,
                      dat.reshape(i1-i0,noccb,nvira,nvirb).transpose(0,2,1,3))
                time1 = log.timer_debug1('pass2 ao2mo for ab [%d:%d]' % (i0,i1), *time1)

    time0 = log.timer('mp2 ao2mo_ovov pass2', *time0)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.ROHF(mol).run()
    mp = ROOBMP2(mf)
    mp.verbose = 5

    #pt = OBMP2(mf)
    #emp2, t2 = pt.kernel()
    #print(emp2 - -0.204019967288338)
    #pt.max_memory = 1
    #emp2, t2 = pt.kernel()
    #print(emp2 - -0.204019967288338)
    #
    #pt = MP2(scf.density_fit(mf, 'weigend'))
    #print(pt.kernel()[0] - -0.204254500454)
