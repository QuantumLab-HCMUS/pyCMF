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


"""
Periodic OB-MP2
"""

import time, logging, tracemalloc
from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.lib import kpts_helper, kpts
from pyscf import __config__
from pyscf.pbc.mp import kmp2

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)
LARGE_DENOM = getattr(__config__, 'LARGE_DENOM', 1e14)


def kernel(mp, mo_energy, mo_coeff, mo_occ, with_t2=WITH_T2, verbose=logger.NOTE):

    nuc = mp._scf.energy_nuc()
    nmo = mp.nmo
    nkpts = numpy.shape(mo_energy)[0]
    nocc = mp.nocc
    niter = mp.niter
    ene_old = 0.0
    # dm = mp._scf.make_rdm1(mo_coeff, mo_occ)
    # print('===========dm========')
    # print(dm)
    DIIS_RESID = [[] for _ in range(nkpts)]
    F_list = [[] for _ in range(nkpts)]
    coeff = [[] for _ in range(nkpts)]
    """
    h1ao = mp._scf.get_hcore()
    veffao = mp._scf.get_veff(mp._scf.cell, dm)
    veff= [reduce(numpy.dot, (mo.T.conj(), veffao[k], mo))
                            for k, mo in enumerate(mo_coeff)]
    fock_hf = numpy.zeros((nkpts, nmo, nmo), dtype=complex)
    fock_hf += veff
    fock_hf += [reduce(numpy.dot, (mo.T.conj(), h1ao[k], mo))
                            for k, mo in enumerate(mo_coeff)]
    c0_hf = 0
    for kp in range(nkpts):
        for i in range(nocc):
            c0_hf -=  veff[kp][i,i].real
    c0_hf/= nkpts
    """
    print(mo_energy)
    print()
    print('**********************************')
    print('************** OBMP2 *************')
    # sort_idx = numpy.argsort(mo_energy)
    f = numpy.empty_like(mo_coeff)
    for it in range(niter):
        dm = mp._scf.make_rdm1(mo_coeff, mo_occ)
        # print("dm", dm)
        h1ao = mp._scf.get_hcore()
        veffao = mp._scf.get_veff(mp._scf.cell, dm)
        sort_idx = numpy.argsort(mo_energy)

        #####################
        ### Hartree-Fock
        veff = [reduce(numpy.dot, (mo.T.conj(), veffao[k], mo)) for k, mo in enumerate(mo_coeff)]

        c0_hf = 0
        for kp in range(nkpts):
            for i in range(nocc):
                c0_hf -= veff[kp][i, i].real
        c0_hf /= nkpts
        fock_hf = numpy.zeros((nkpts, nmo, nmo), dtype=complex)
        fock_hf += veff
        fock_hf += [reduce(numpy.dot, (mo.T.conj(), h1ao[k], mo)) for k, mo in enumerate(mo_coeff)]
        numpy.set_printoptions(precision=6)

        # initializing w/ HF
        fock = 0
        fock += fock_hf
        c0 = c0_hf

        ene_hf = 0
        for k in range(nkpts):
            for i in range(nocc):
                ene_hf += 2 * fock[k][i, i].real / nkpts

        ene_hf += c0_hf + nuc

        if mp.second_order:
            mp.ampf = 1.0
        #####################
        ### OBMP2
        c0_1st, c1 = first_BCH(mp, mo_energy, mo_coeff, fock_hf)
        for k in range(nkpts):
            fock[k] += c1[k] + c1[k].T.conj()

        #####################
        ene = 0
        ene0 = 0

        for k in range(nkpts):
            for i in range(nocc):
                ene += 2 * fock[k][i, i].real / nkpts
        ene_tot = ene + c0 + c0_1st + nuc  # + c0_2nd
        # ene_tot = ene_hf
        print('e_corr = ', ene_tot - ene_hf)
        de = abs(ene_tot - ene_old)
        ene_old = ene_tot
        tracemalloc.start(25)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        stat = top_stats[:10]
        total_mem = sum(stat.size for stat in top_stats)
        print()
        print('i/ter = %d' % it, ' ene = %8.8f' % ene_tot, ' ene diff = %8.8f' % de, flush=True)
        print()
        print()
        nk = 0
        if de < mp.thresh:
            break
        ## diagonalizing correlated Fock
        new_mo_coeff = numpy.empty_like(mo_coeff, dtype=complex)
        U = numpy.empty_like(mo_coeff)
        new_mo_energy = numpy.empty_like(mo_energy, dtype=complex)
        for k in range(nkpts):
            new_mo_energy[k], U = scipy.linalg.eigh(fock[k])
            new_mo_coeff[k] = numpy.dot(mo_coeff[k], U)

            mo_energy[k] = new_mo_energy[k].real
            mo_coeff[k] = new_mo_coeff[k]
            print()
            # print("mo_energy")
            # print(mo_coeff[k])
            # mo_energy[k] = mp.mo_energy[k][sort_idx[k]].real
    IP, EA = make_IPEA(mp, mo_energy, mo_coeff, fock_hf)
    print('IP_v1 = ', IP[0] - mo_energy[nk][nocc - 1])
    print('EA_c1 = ', EA[0] - mo_energy[nk][nocc])
    print('IP_v2 = ', IP[1] - mo_energy[nk][nocc - 2])
    print('EA_c2 = ', EA[1] - mo_energy[nk][nocc + 1])
    print('IP_v3 = ', IP[2] - mo_energy[nk][nocc - 3])
    print('EA_c3 = ', EA[2] - mo_energy[nk][nocc + 2])
    return ene_tot, mo_energy


#################################################################################################################


def make_veff(mp, mo_coeff, mo_energy):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = numpy.shape(mo_energy)[0]

    kpts = mp.kpts
    dm = mp._scf.make_rdm1()
    veff_ao = mp._scf.get_veff(mp._scf.cell, dm)

    veff = numpy.zeros((nkpts, nmo, nmo), dtype=complex)

    for kp in range(nkpts):
        veff[kp] = numpy.matmul(mo_coeff[kp].T.conj(), numpy.matmul(veff_ao[kp], mo_coeff[kp]))

    c0_hf = 0
    for kp in range(nkpts):
        for i in range(nocc):
            c0_hf -= veff[kp][i, i].real
    c0_hf /= nkpts

    return veff_ao, veff, c0_hf


def ene_denom(mp, mo_energy, ki, ka, kj, kb):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = numpy.shape(mo_energy)[0]

    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind='split')
    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    eia = LARGE_DENOM * numpy.ones((nocc, nvir), dtype=mo_energy[0].dtype)
    n0_ovp_ia = numpy.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
    eia[n0_ovp_ia] = (mo_e_o[ki][:, None] - mo_e_v[ka])[n0_ovp_ia]

    ejb = LARGE_DENOM * numpy.ones((nocc, nvir), dtype=mo_energy[0].dtype)
    n0_ovp_jb = numpy.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
    ejb[n0_ovp_jb] = (mo_e_o[kj][:, None] - mo_e_v[kb])[n0_ovp_jb]

    ejh = LARGE_DENOM * numpy.ones((nocc, nocc), dtype=mo_energy[0].dtype)
    n0_ovp_jh = numpy.ix_(nonzero_opadding[kj], nonzero_opadding[kb])
    ejh[n0_ovp_jh] = (mo_e_o[kj][:, None] - mo_e_o[kb])[n0_ovp_jh]

    eah = LARGE_DENOM * numpy.ones((nvir, nocc), dtype=mo_energy[0].dtype)
    n0_ovp_ah = numpy.ix_(nonzero_opadding[ka], nonzero_opadding[kb])
    eah[n0_ovp_ah] = (mo_e_v[ka][:, None] - mo_e_o[kb])[n0_ovp_ah]

    eij = LARGE_DENOM * numpy.ones((nocc, nocc), dtype=mo_energy[0].dtype)
    n0_ovp_ij = numpy.ix_(nonzero_opadding[ki], nonzero_opadding[kj])
    eij[n0_ovp_ij] = (mo_e_o[ki][:, None] - mo_e_o[kj])[n0_ovp_ij]

    elb = LARGE_DENOM * numpy.ones((nvir, nvir), dtype=mo_energy[0].dtype)
    n0_ovp_lb = numpy.ix_(nonzero_vpadding[kj], nonzero_vpadding[kb])
    elb[n0_ovp_lb] = (mo_e_v[kj][:, None] - mo_e_v[kb])[n0_ovp_lb]

    e_iajb = lib.direct_sum('ia,jb -> iajb', eia, ejb)
    e_ialb = lib.direct_sum('ia,lb -> ialb', eia, elb)
    e_iajh = lib.direct_sum('ia,jh -> iajh', eia, ejh)
    e_ahij = lib.direct_sum('ah,ij -> ahij', eah, eij)

    return e_iajb, e_ialb, e_iajh, e_ahij


def first_BCH(mp, mo_energy, mo_coeff, fock_hf):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts
    kd = mp.kpts
    kconserv = mp.khelper.kconserv
    fao2mo = mp._scf.with_df.ao2mo
    kd = kpts.KPoints(mp._scf.cell, mp.kpts)
    kd.build(space_group_symmetry=True)
    k1 = [1]
    k2 = [2]
    ovlp = mp._scf.get_ovlp()

    """
    kijab, weight, k4_bz2ibz = kd.make_k4_ibz(sym='s2')
    _, igroup = numpy.unique(kijab[:,:2], axis=0, return_index=True)
    igroup = igroup.ravel()
    igroup = list(igroup) + [len(kijab)]
    edi = 0
    exi = 0
    
    print("kijab",kijab)
    print("igroup", igroup)
    print("k4_bz2ibz", k4_bz2ibz)
    """
    tmp1 = numpy.zeros((nkpts, nocc, nvir, nocc, nvir), dtype=complex)
    tmp1_bar = numpy.zeros((nkpts, nocc, nvir, nocc, nvir), dtype=complex)
    tmp_bar = numpy.zeros((nkpts, nkpts, nocc, nvir, nocc, nvir), dtype=complex)
    h2mo_ovgg = numpy.zeros((nkpts, nocc, nvir, nmo, nmo), dtype=complex)
    h2mo_ovov = numpy.zeros((nkpts, nocc, nvir, nocc, nvir), dtype=complex)
    h2mo_ovog = numpy.zeros((nkpts, nocc, nvir, nocc, nmo), dtype=complex)
    h2mo_ovgv = numpy.zeros((nkpts, nocc, nvir, nmo, nvir), dtype=complex)
    c1 = numpy.zeros((nkpts, nmo, nmo), dtype=complex)
    c2 = numpy.zeros((nkpts, nmo, nmo), dtype=complex)
    y1 = numpy.zeros((nkpts, nvir, nocc), dtype=complex)
    y2 = numpy.zeros((nkpts, nocc, nvir, nocc, nvir), dtype=complex)
    y3 = numpy.zeros((nkpts, nocc, nvir, nocc, nvir), dtype=complex)
    y4 = numpy.zeros((nkpts, nocc, nocc), dtype=complex)
    y5 = numpy.zeros((nkpts, nvir, nvir), dtype=complex)
    c0 = 0
    IP_v1 = 0
    IP_v2 = 0
    IP_v3 = 0
    EA_c1 = 0
    EA_c2 = 0
    EA_c3 = 0
    print('mo energy', mo_energy)
    nk = 0
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                kp = kj
                kq = kb
                o_i = mo_coeff[ki][:, :nocc]
                o_a = mo_coeff[ka][:, nocc:]
                o_p = mo_coeff[kp]
                o_q = mo_coeff[kq]
                h2mo_ovgg[ka] = (
                    fao2mo(
                        (o_i, o_a, o_p, o_q), (mp.kpts[ki], mp.kpts[ka], mp.kpts[kp], mp.kpts[kq]), compact=False
                    ).reshape(nocc, nvir, nmo, nmo)
                    / nkpts
                )
                h2mo_ovov[ka] = h2mo_ovgg[ka][:, :, :nocc, nocc:]
                h2mo_ovgv[ka] = h2mo_ovgg[ka][:, :, :, nocc:]
                h2mo_ovog[ka] = h2mo_ovgg[ka][:, :, :nocc, :]
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                e_iajb, e_ialb, e_iajh, e_ahij = ene_denom(mp, mo_energy, ki, ka, kj, kb)
                w_iajb = h2mo_ovov[ka] - 0.5 * h2mo_ovov[kb].transpose(0, 3, 2, 1)
                tmp1[ka] = (h2mo_ovov[ka] / e_iajb).conj()
                tmp1_bar[ka] = (w_iajb / e_iajb).conj()

                # 1st order
                ### Tb_iajb * g_iajp -> c1_pb
                c1[kb, nocc:, :] -= 2 * numpy.einsum('iajb, iajp -> bp', tmp1_bar[ka], h2mo_ovog[ka])
                ### Tb_iajb * g_iapb -> c1_jp
                c1[kj, :, :nocc] += 2 * numpy.einsum('iajb, iapb -> pj', tmp1_bar[ka], h2mo_ovgv[ka])
                ### Tb_iajb * f_ia   -> c1_jb
                if ki == ka:
                    c1[kj, nocc:, :nocc] += 2 * numpy.einsum('iajb, ia -> bj', tmp1_bar[ka], fock_hf[ka, :nocc, nocc:])
                ### c0_1st
                c0 -= 4 * numpy.einsum('iajb, iajb', tmp1_bar[ka], h2mo_ovov[ka]).real
                ### 2nd order
                if ki == ka:
                    y1[kj] += 2 * numpy.einsum('ia, iajb -> bj', fock_hf[ka][:nocc, nocc:], tmp1_bar[ka])
                    tmp_bar[ki, kj] = tmp1_bar[ka]
                    print(ki, kj)
                y2[ka] = numpy.einsum('ca, iclb -> ialb', fock_hf[ka, nocc:, nocc:], tmp1_bar[ka].conj())
                y3[ka] = numpy.einsum('ik, kalb -> ialb', fock_hf[ki, :nocc, :nocc], tmp1_bar[ka].conj())
                y4[ki, :, :] += numpy.einsum('iajb, kajb -> ki', tmp1[ka], tmp1_bar[ka].conj())
                y5[ka, :, :] += numpy.einsum('iajb, icjb -> ac', tmp1[ka], tmp1_bar[ka].conj())

                c2[kj, :nocc, :nocc] += numpy.einsum('ialb, iajb -> lj', y2[ka], tmp1[ka])  # [2]
                c2[ki, :nocc, :nocc] += numpy.einsum('kajb, iajb -> ki', y2[ka], tmp1[ka])  # [3]
                c2[kb, nocc:, nocc:] -= numpy.einsum('iajd, iajb -> bd', y2[ka], tmp1[ka])  # [8]
                c2[kj, :nocc, :nocc] -= numpy.einsum('ialb, iajb -> lj', y3[ka], tmp1[ka])  # [4]
                c2[ka, nocc:, nocc:] += numpy.einsum('icjb, iajb -> ac', y3[ka], tmp1[ka])  # [7]
                c2[kb, nocc:, nocc:] += numpy.einsum('iajd, iajb -> bd', y3[ka], tmp1[ka])  # [6]
                c0 -= 4 * numpy.einsum('iajb,iajb ->', y2[ka], tmp1[ka]).real  # [11]
                c0 += 4 * numpy.einsum('iajb,iajb ->', y3[ka], tmp1[ka]).real  # [10]

    for ki in range(nkpts):
        c2[ki, :nocc, :] -= numpy.einsum('ip, ki -> kp', fock_hf[ki, :nocc, :], y4[ki])  # [5]
        c2[ki, :, nocc:] -= numpy.einsum('pa, ac -> pc', fock_hf[ki, :, nocc:], y5[ki])  # [9]
        c2[ki, :nocc, nocc:] += numpy.einsum('qbj, qjbkc -> kc', y1, tmp_bar[:, ki, :, :, :, :].conj())  # [1]

    c0 /= nkpts
    c1 += c2
    return c0, c1


def make_IPEA(mp, mo_energy, mo_coeff, fock_hf):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts
    kd = mp.kpts
    kconserv = mp.khelper.kconserv
    fao2mo = mp._scf.with_df.ao2mo

    IP_v1 = 0
    IP_v2 = 0
    IP_v3 = 0
    EA_c1 = 0
    EA_c2 = 0
    EA_c3 = 0
    nk = 0

    tmp1_bar_iajh = numpy.zeros((nkpts, nkpts, nocc, nvir, nocc, nocc), dtype=complex)
    w_iajh = numpy.zeros((nkpts, nkpts, nocc, nvir, nocc, nocc), dtype=complex)
    tmp1_bar_ialb = numpy.zeros((nkpts, nkpts, nocc, nvir, nvir, nvir), dtype=complex)
    w_ialb = numpy.zeros((nkpts, nkpts, nocc, nvir, nvir, nvir), dtype=complex)
    h2mo_ovoo = numpy.zeros((nkpts, nkpts, nocc, nvir, nocc, nocc), dtype=complex)
    h2mo_ovvv = numpy.zeros((nkpts, nkpts, nocc, nvir, nvir, nvir), dtype=complex)

    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                o_i = mo_coeff[ki][:, :nocc]
                o_a = mo_coeff[ka][:, nocc:]
                o_j = mo_coeff[kj]
                o_b = mo_coeff[kb]
                if kb == nk:
                    h2mo_ovoo[ki, ka] = (
                        fao2mo(
                            (o_i, o_a, o_j[:, :nocc], o_b[:, :nocc]),
                            (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]),
                            compact=False,
                        ).reshape(nocc, nvir, nocc, nocc)
                        / nkpts
                    )
                if kj == nk:
                    h2mo_ovvv[ki, ka] = (
                        fao2mo(
                            (o_i, o_a, o_j[:, nocc:], o_b[:, nocc:]),
                            (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]),
                            compact=False,
                        ).reshape(nocc, nvir, nvir, nvir)
                        / nkpts
                    )
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                e_iajb, e_ialb, e_iajh, e_ahij = ene_denom(mp, mo_energy, ki, ka, kj, kb)
                if kb == nk:
                    w_iajh[ki, ka] = h2mo_ovoo[ki, ka] - 0.5 * h2mo_ovoo[kj, ka].transpose(2, 1, 0, 3)
                    tmp1_bar_iajh[ki, ka] = (w_iajh[ki, ka] / e_iajh).conj()
                    IP_v1 += (
                        2
                        * numpy.einsum(
                            'iaj, iaj -> ',
                            tmp1_bar_iajh[ki, ka][:, :, :, nocc - 1],
                            h2mo_ovoo[ki, ka][:, :, :, nocc - 1],
                        ).real
                    )
                    IP_v2 += (
                        2
                        * numpy.einsum(
                            'iaj, iaj -> ',
                            tmp1_bar_iajh[ki, ka][:, :, :, nocc - 2],
                            h2mo_ovoo[ki, ka][:, :, :, nocc - 2],
                        ).real
                    )
                    IP_v3 += (
                        2
                        * numpy.einsum(
                            'iaj, iaj -> ',
                            tmp1_bar_iajh[ki, ka][:, :, :, nocc - 3],
                            h2mo_ovoo[ki, ka][:, :, :, nocc - 3],
                        ).real
                    )
                if kj == nk:
                    w_ialb = h2mo_ovvv[ki, ka] - 0.5 * h2mo_ovvv[ki, kb].transpose(0, 3, 2, 1)
                    tmp1_bar_ialb = (w_ialb / e_ialb).conj()
                    EA_c1 -= (
                        2 * numpy.einsum('iab, iab -> ', tmp1_bar_ialb[:, :, 0, :], h2mo_ovvv[ki, ka][:, :, 0, :]).real
                    )
                    EA_c2 -= (
                        2 * numpy.einsum('iab, iab -> ', tmp1_bar_ialb[:, :, 1, :], h2mo_ovvv[ki, ka][:, :, 1, :]).real
                    )
                    EA_c3 -= (
                        2 * numpy.einsum('iab, iab -> ', tmp1_bar_ialb[:, :, 2, :], h2mo_ovvv[ki, ka][:, :, 2, :]).real
                    )

    IP = [IP_v1, IP_v2, IP_v3]
    EA = [EA_c1, EA_c2, EA_c3]

    return IP, EA


"""
def make_rdm1(mp): # , t2=None, eris=None, verbose=logger.NOTE, ao_repr=False):
    '''Spin-traced one-particle density matrix.
    The occupied-virtual orbital response != included.x

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)

    Kwargs:
        ao_repr : boolean
            Whether to transfrom 1-particle density matrix to AO
            representation.
    '''
    from pyscf.cc import ccsd_rdm

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia = mp.mo_energy[:nocc,None] - mp.mo_energy[None,nocc:] 
    eris = mp.ao2mo(mp.mo_coeff)
    
    t2 = mp.tmp_dip
    doo, dvv = _gamma1_intermediates(mp, t2, eris)
    nocc = doo.shape[0]
    nvir = dvv.shape[0]
    dov = numpy.zeros((nocc,nvir), dtype=doo.dtype)
    dvo = dov.T
    return ccsd_rdm._make_rdm1(mp, (doo, dov, dvo, dvv), with_frozen=True,
                               ao_repr=False)

def _gamma1_intermediates(mp, t2):
    if t2 is None: t2 = mp.t2
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    if t2 is None:
        if eris is None: eris = mp.ao2mo()
        mo_energy = _mo_energy_without_core(mp, mp.mo_energy)
        eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
        dtype = eris.ovov.dtype
    else:
        dtype = t2.dtype

    dm1occ = numpy.zeros((nocc,nocc), dtype=dtype)
    dm1vir = numpy.zeros((nvir,nvir), dtype=dtype)
    for i in range(nocc):
        if t2 is None:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
            t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
        else:
            t2i = t2[i]
        l2i = t2i.conj()
        dm1vir += numpy.einsum('jca,jcb->ba', l2i, t2i) * 2 \
                - numpy.einsum('jca,jbc->ba', l2i, t2i)
        dm1occ += numpy.einsum('iab,jab->ij', l2i, t2i) * 2 \
                - numpy.einsum('iab,jba->ij', l2i, t2i)
    return -dm1occ, dm1vir
"""


def _add_padding(mp, mo_coeff, mo_energy, mo_occ):
    from pyscf.pbc import tools
    from pyscf.pbc.cc.ccsd import _adjust_occ

    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts

    # Check if these are padded mo coefficients and energies
    if not numpy.all([x.shape[0] == nmo for x in mo_coeff]):
        mo_coeff = padded_mo_coeff(mp, mo_coeff)

    if not numpy.all([x.shape[0] == nmo for x in mo_energy]):
        mo_energy = padded_mo_energy(mp, mo_energy)

    if not numpy.all([x.shape[0] == nmo for x in mo_occ]):
        mo_occ = padded_mo_occ(mp, mo_occ)

    return mo_coeff, mo_energy, mo_occ


def _padding_k_idx(nmo, nocc, kind='split'):
    """A convention used for padding vectors, matrices and tensors in case when occupation numbers depend on the
    k-point index.
    Args:
        nmo (Iterable): k-dependent orbital number;
        nocc (Iterable): k-dependent occupation numbers;
        kind (str): either "split" (occupied and virtual spaces are split) or "joint" (occupied and virtual spaces are
        the joint;

    Returns:
        Two lists corresponding to the occupied and virtual spaces for kind="split". Each list contains integer arrays
        with indexes pointing to actual non-zero entries in the padded vector/matrix/tensor. If kind="joint", a single
        list of arrays is returned corresponding to the entire MO space.
    """
    if kind not in ('split', 'joint'):
        raise ValueError("The 'kind' argument must be one of 'split', 'joint'")

    if kind == 'split':
        indexes_o = []
        indexes_v = []
    else:
        indexes = []

    nocc = numpy.array(nocc)
    nmo = numpy.array(nmo)
    nvirt = nmo - nocc
    dense_o = numpy.amax(nocc)
    dense_v = numpy.amax(nvirt)
    dense_nmo = dense_o + dense_v

    for k_o, k_nmo in zip(nocc, nmo):
        k_v = k_nmo - k_o
        if kind == 'split':
            indexes_o.append(numpy.arange(k_o))
            indexes_v.append(numpy.arange(dense_v - k_v, dense_v))
        else:
            indexes.append(
                numpy.concatenate(
                    (
                        numpy.arange(k_o),
                        numpy.arange(dense_nmo - k_v, dense_nmo),
                    )
                )
            )

    if kind == 'split':
        return indexes_o, indexes_v

    else:
        return indexes


def padding_k_idx(mp, kind='split'):
    """A convention used for padding vectors, matrices and tensors in case when occupation numbers depend on the
    k-point index.

    This implementation stores k-dependent Fock and other matrix in dense arrays with additional dimensions
    corresponding to k-point indexes. In case when the occupation numbers depend on the k-point index (i.e. a metal) or
    when some k-points have more Bloch basis functions than others the corresponding data structure has to be padded
    with entries that are not used (fictitious occupied and virtual degrees of freedom). Current convention stores these
    states at the Fermi level as shown in the following example.

    +----+--------+--------+--------+
    |    |  k=0   |  k=1   |  k=2   |
    |    +--------+--------+--------+
    |    | nocc=2 | nocc=3 | nocc=2 |
    |    | nvir=4 | nvir=3 | nvir=3 |
    +====+========+========+========+
    | v3 |  k0v3  |  k1v2  |  k2v2  |
    +----+--------+--------+--------+
    | v2 |  k0v2  |  k1v1  |  k2v1  |
    +----+--------+--------+--------+
    | v1 |  k0v1  |  k1v0  |  k2v0  |
    +----+--------+--------+--------+
    | v0 |  k0v0  |        |        |
    +====+========+========+========+
    |          Fermi level          |
    +====+========+========+========+
    | o2 |        |  k1o2  |        |
    +----+--------+--------+--------+
    | o1 |  k0o1  |  k1o1  |  k2o1  |
    +----+--------+--------+--------+
    | o0 |  k0o0  |  k1o0  |  k2o0  |
    +----+--------+--------+--------+

    In the above example, `get_nmo(mp, per_kpoint=True) == (6, 6, 5)`, `get_nocc(mp, per_kpoint) == (2, 3, 2)`. The
    resulting dense `get_nmo(mp) == 7` and `get_nocc(mp) == 3` correspond to padded dimensions. This function will
    return the following indexes corresponding to the filled entries of the above table:

    >>> padding_k_idx(mp, kind="split")
    ([(0, 1), (0, 1, 2), (0, 1)], [(0, 1, 2, 3), (1, 2, 3), (1, 2, 3)])

    >>> padding_k_idx(mp, kind="joint")
    [(0, 1, 3, 4, 5, 6), (0, 1, 2, 4, 5, 6), (0, 1, 4, 5, 6)]

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        kind (str): either "split" (occupied and virtual spaces are split) or "joint" (occupied and virtual spaces are
        the joint;

    Returns:
        Two lists corresponding to the occupied and virtual spaces for kind="split". Each list contains integer arrays
        with indexes pointing to actual non-zero entries in the padded vector/matrix/tensor. If kind="joint", a single
        list of arrays is returned corresponding to the entire MO space.
    """
    return _padding_k_idx(mp.get_nmo(per_kpoint=True), mp.get_nocc(per_kpoint=True), kind=kind)


def padded_mo_occ(mp, mo_occ):
    """
    Pads occupancy of active MOs.

    Returns:
        Padded molecular occupancy.
    """
    frozen_mask = get_frozen_mask(mp)
    padding_convention = padding_k_idx(mp, kind='joint')
    nkpts = mp.nkpts

    result = numpy.zeros((nkpts, mp.nmo), dtype=mo_occ[0].dtype)
    for k in range(nkpts):
        result[numpy.ix_([k], padding_convention[k])] = mo_occ[k][frozen_mask[k]]

    return result


def padded_mo_energy(mp, mo_energy):
    """
    Pads energies of active MOs.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        mo_energy (ndarray): original non-padded molecular energies;

    Returns:
        Padded molecular energies.
    """
    frozen_mask = get_frozen_mask(mp)
    padding_convention = padding_k_idx(mp, kind='joint')
    nkpts = mp.nkpts

    result = numpy.zeros((nkpts, mp.nmo), dtype=mo_energy[0].dtype)
    for k in range(nkpts):
        result[numpy.ix_([k], padding_convention[k])] = mo_energy[k][frozen_mask[k]]

    return result


def padded_mo_coeff(mp, mo_coeff):
    """
    Pads coefficients of active MOs.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        mo_coeff (ndarray): original non-padded molecular coefficients;

    Returns:
        Padded molecular coefficients.
    """
    frozen_mask = get_frozen_mask(mp)
    padding_convention = padding_k_idx(mp, kind='joint')
    nkpts = mp.nkpts

    result = numpy.zeros((nkpts, mo_coeff[0].shape[0], mp.nmo), dtype=mo_coeff[0].dtype)
    for k in range(nkpts):
        result[numpy.ix_([k], numpy.arange(result.shape[1]), padding_convention[k])] = mo_coeff[k][:, frozen_mask[k]]

    return result


def _frozen_sanity_check(frozen, mo_occ, kpt_idx):
    """Performs a few sanity checks on the frozen array and mo_occ.

    Specific tests include checking for duplicates within the frozen array.

    Args:
        frozen (array_like of int): The orbital indices that will be frozen.
        mo_occ (:obj:`ndarray` of int): The occupuation number for each orbital
            resulting from a mean-field-like calculation.
        kpt_idx (int): The k-point that `mo_occ` and `frozen` belong to.

    """
    frozen = numpy.array(frozen)
    nocc = numpy.count_nonzero(mo_occ > 0)
    nvir = len(mo_occ) - nocc
    assert nocc, 'No occupied orbitals?\n\nnocc = %s\nmo_occ = %s' % (nocc, mo_occ)
    all_frozen_unique = (len(frozen) - len(numpy.unique(frozen))) == 0
    if not all_frozen_unique:
        raise RuntimeError('Frozen orbital list contains duplicates!\n\nkpt_idx %s\nfrozen %s' % (kpt_idx, frozen))
    if len(frozen) > 0 and numpy.max(frozen) > len(mo_occ) - 1:
        raise RuntimeError(
            'Freezing orbital not in MO list!\n\nkpt_idx %s\n'
            'frozen %s\nmax orbital idx %s' % (kpt_idx, frozen, len(mo_occ) - 1)
        )


def get_nocc(mp, per_kpoint=False):
    """Number of occupied orbitals for k-point calculations.

    Number of occupied orbitals for use in a calculation with k-points, taking into
    account frozen orbitals.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        per_kpoint (bool, optional): True returns the number of occupied
            orbitals at each k-point.  False gives the max of this list.

    Returns:
        nocc (int, list of int): Number of occupied orbitals. For return type, see description of arg
            `per_kpoint`.

    """
    for i, moocc in enumerate(mp.mo_occ):
        if numpy.any(moocc % 1 != 0):
            raise RuntimeError(
                'Fractional occupation numbers encountered @ kp={:d}: {}. This may have been caused by '
                'smearing of occupation numbers in the mean-field calculation. If so, consider '
                'executing mf.smearing_method = False; mf.mo_occ = mf.get_occ() prior to calling '
                'this'.format(i, moocc)
            )
    if mp._nocc is not None:
        return mp._nocc
    if isinstance(mp.frozen, (int, numpy.integer)):
        nocc = [(numpy.count_nonzero(mp.mo_occ[ikpt]) - mp.frozen) for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        [_frozen_sanity_check(mp.frozen, mp.mo_occ[ikpt], ikpt) for ikpt in range(mp.nkpts)]
        nocc = []
        for ikpt in range(mp.nkpts):
            max_occ_idx = numpy.max(numpy.where(mp.mo_occ[ikpt] > 0))
            frozen_nocc = numpy.sum(numpy.array(mp.frozen) <= max_occ_idx)
            nocc.append(numpy.count_nonzero(mp.mo_occ[ikpt]) - frozen_nocc)
    elif isinstance(mp.frozen[0], (list, numpy.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError(
                'Frozen list has a different number of k-points (length) than passed in mean-field/'
                'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                '(length = %d)' % (mp.nkpts, mp.frozen, nkpts)
            )
        [_frozen_sanity_check(frozen, mo_occ, ikpt) for ikpt, frozen, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]

        nocc = []
        for ikpt, frozen in enumerate(mp.frozen):
            max_occ_idx = numpy.max(numpy.where(mp.mo_occ[ikpt] > 0))
            frozen_nocc = numpy.sum(numpy.array(frozen) <= max_occ_idx)
            nocc.append(numpy.count_nonzero(mp.mo_occ[ikpt]) - frozen_nocc)
    else:
        raise NotImplementedError

    assert any(numpy.array(nocc) > 0), 'Must have occupied orbitals! \n\nnocc %s\nfrozen %s\nmo_occ %s' % (
        nocc,
        mp.frozen,
        mp.mo_occ,
    )

    if not per_kpoint:
        nocc = numpy.amax(nocc)

    return nocc


def get_nmo(mp, per_kpoint=False):
    """Number of orbitals for k-point calculations.

    Number of orbitals for use in a calculation with k-points, taking into account
    frozen orbitals.

    Note:
        If `per_kpoint` is False, then the number of orbitals here is equal to max(nocc) + max(nvir),
        where each max is done over all k-points.  Otherwise the number of orbitals is returned
        as a list of number of orbitals at each k-point.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        per_kpoint (bool, optional): True returns the number of orbitals at each k-point.
            For a description of False, see Note.

    Returns:
        nmo (int, list of int): Number of orbitals. For return type, see description of arg
            `per_kpoint`.

    """
    if mp._nmo is not None:
        return mp._nmo

    if isinstance(mp.frozen, (int, numpy.integer)):
        nmo = [len(mp.mo_occ[ikpt]) - mp.frozen for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        [_frozen_sanity_check(mp.frozen, mp.mo_occ[ikpt], ikpt) for ikpt in range(mp.nkpts)]
        nmo = [len(mp.mo_occ[ikpt]) - len(mp.frozen) for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen, (list, numpy.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError(
                'Frozen list has a different number of k-points (length) than passed in mean-field/'
                'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                '(length = %d)' % (mp.nkpts, mp.frozen, nkpts)
            )
        [_frozen_sanity_check(fro, mo_occ, ikpt) for ikpt, fro, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]

        nmo = [len(mp.mo_occ[ikpt]) - len(mp.frozen[ikpt]) for ikpt in range(nkpts)]
    else:
        raise NotImplementedError

    assert all(numpy.array(nmo) > 0), 'Must have a positive number of orbitals!\n\nnmo %s\nfrozen %s\nmo_occ %s' % (
        nmo,
        mp.frozen,
        mp.mo_occ,
    )

    if not per_kpoint:
        # Depending on whether there are more occupied bands, we want to make sure that
        # nmo has enough room for max(nocc) + max(nvir) number of orbitals for occupied
        # and virtual space
        nocc = mp.get_nocc(per_kpoint=True)
        nmo = numpy.max(nocc) + numpy.max(numpy.array(nmo) - numpy.array(nocc))

    return nmo


def get_frozen_mask(mp):
    """Boolean mask for orbitals in k-point post-HF method.

    Creates a boolean mask to remove frozen orbitals and keep other orbitals for post-HF
    calculations.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.

    Returns:
        moidx (list of :obj:`ndarray` of `numpy.bool`): Boolean mask of orbitals to include.

    """
    moidx = [numpy.ones(x.size, dtype=bool) for x in mp.mo_occ]
    if isinstance(mp.frozen, (int, numpy.integer)):
        for idx in moidx:
            idx[: mp.frozen] = False
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        frozen = list(mp.frozen)
        for idx in moidx:
            idx[frozen] = False
    elif isinstance(mp.frozen[0], (list, numpy.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError(
                'Frozen list has a different number of k-points (length) than passed in mean-field/'
                'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                '(length = %d)' % (mp.nkpts, mp.frozen, nkpts)
            )
        [_frozen_sanity_check(fro, mo_occ, ikpt) for ikpt, fro, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]
        for ikpt, kpt_occ in enumerate(moidx):
            kpt_occ[mp.frozen[ikpt]] = False
    else:
        raise NotImplementedError

    return moidx


class OBMP2(lib.StreamObject):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):

        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        self.thresh = 1e-08
        self.shift = 0.0
        self.niter = 100
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen

        self.mom = False
        self.occ_exc = [None, None]
        self.vir_exc = [None, None]

        self.second_order = True
        self.ampf = 1

        ##################################################
        # don't modify the following attributes, they are not input options
        self.kpts = mf.kpts
        self.nkpts = numpy.shape(mf.mo_energy)[0]
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.e_corr = None
        self.t2 = None
        self._keys = set(self.__dict__.keys())

    @property
    def nocc(self):
        return self.get_nocc()

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    # int_transform = int_transform

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen != 0:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
        return self

    @property
    def emp2(self):
        return self.e_corr

    @property
    def e_tot(self):
        return self.ene_tot  # + self._scf.e_tot

    def kernel(self, shift=0.0, mo_energy=None, mo_coeff=None, mo_occ=None, with_t2=WITH_T2, _kern=kernel):
        """
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.
        """
        if mo_occ is None:
            mo_occ = self.mo_occ
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None or mo_occ is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff, mo_energy are not given.\nYou may need to call mf.kernel() to generate them.')

        mo_coeff, mo_energy, mo_occ = _add_padding(self, mo_coeff, mo_energy, mo_occ)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        # self.dump_flags()
        # _kern(self, mo_energy, mo_coeff, eris, with_t2, self.verbose)
        self.ene_tot, self.mo_energy = _kern(self, mo_energy, mo_coeff, mo_occ, with_t2, self.verbose)
        self._finalize()
        return self.ene_tot, self.mo_energy

    def _finalize(self):
        """Hook for dumping results and clearing up the object."""
        logger.note(self, 'E(%s) = %.15g', self.__class__.__name__, self.e_tot)
        return self

    make_veff = make_veff
    # make_amp  = make_amp
    first_BCH = first_BCH
    # second_BCH = second_BCH
    # make_rdm1 = make_rdm1
    # make_rdm2 = make_rdm2

    # as_scanner = as_scanner

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.mp import dfmp2

        mymp = dfmp2.DFMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
        if with_df is not None:
            mymp.with_df = with_df
        if mymp.with_df.auxbasis != auxbasis:
            mymp.with_df = copy.copy(mymp.with_df)
            mymp.with_df.auxbasis = auxbasis
        return mymp

    def nuc_grad_method(self):
        from pyscf.grad import mp2

        return mp2.Gradients(self)


del WITH_T2
