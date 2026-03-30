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
Periodic OB-MP2
''' 

import time, logging, tracemalloc
from functools import reduce
import copy
import numpy 
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.lib import kpts_helper
from pyscf import __config__
from pyscf.pbc.mp import kmp2

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)
LARGE_DENOM = getattr(__config__, 'LARGE_DENOM', 1e14)


def kernel(mp, mo_energy, mo_coeff, mo_occ, with_t2=WITH_T2,
           verbose=logger.NOTE):
    mo_ea, mo_eb = mo_energy
    mo_coeff_a, mo_coeff_b = mo_coeff
    mo_occ_a, mo_occ_b = mo_occ
    nuc = mp._scf.energy_nuc()
    #nmoa, nmob = mp.get_nmo()
    nmoa, nmob = mp.nmo
    #nkpts = len(mo_energy[0])
    nkpts = numpy.shape(mo_ea)[0]
    #nocca, noccb = mp.get_nocc()
    nocca, noccb = mp.nocc
    
    niter = mp.niter
    ene_old = 0.


    #dm = mp._scf.make_rdm1(mo_coeff, mo_occ)
    

    print("shift = ", mp.shift)
    print ("thresh = ", mp.thresh)
    print ("niter = ", mp.niter)
    print("nkpts  =", nkpts, "nmoa =", nmoa, "nmob =", nmob,
          "nocca =", nocca, "noccb =", noccb)
    
    DIIS_RESID_a = [[] for _ in range(nkpts)]
    DIIS_RESID_b = [[] for _ in range(nkpts)]
    
    F_list_a = [[] for _ in range(nkpts)]
    F_list_b = [[] for _ in range(nkpts)]
    
    coeff_a = [[] for _ in range(nkpts)]
    coeff_b = [[] for _ in range(nkpts)]

    print(mo_energy)
    print()
    print('**********************************')
    print('************** OBMP2 *************')
    #sort_idx_a = numpy.argsort(mo_ea)
    #sort_idx_b = numpy.argsort(mo_eb)
    #print("sort_idx alpha:", sort_idx_a)
    #print("sort_idx beta :", sort_idx_b)
    #print(sort_idx) 
    for it in range(niter):
        dm = mp._scf.make_rdm1(mo_coeff, mo_occ)
        h1ao = mp._scf.get_hcore()
        veffao_a, veffao_b = mp._scf.get_veff(mp._scf.cell, dm)
        veff_a = [reduce(numpy.dot, (mo.T.conj(), veffao_a[k], mo))
              for k, mo in enumerate(mo_coeff_a)]
        veff_b = [reduce(numpy.dot, (mo.T.conj(), veffao_b[k], mo))
              for k, mo in enumerate(mo_coeff_b)]
        c0_hf_a = 0
        c0_hf_b = 0
        for kp in range(nkpts):
            for i in range(nocca):
                c0_hf_a -= veff_a[kp][i, i].real
            for i in range(noccb):
                c0_hf_b -= veff_b[kp][i, i].real
        c0_hf_a /= nkpts
        c0_hf_b /= nkpts
        fock_hf_a = numpy.zeros((nkpts, nmoa, nmoa), dtype=complex)
        fock_hf_b = numpy.zeros((nkpts, nmob, nmob), dtype=complex)
        fock_hf_a += veff_a
        fock_hf_b += veff_b
        fock_hf_a += [reduce(numpy.dot, (mo.T.conj(), h1ao[k], mo))
                  for k, mo in enumerate(mo_coeff_a)]
        fock_hf_b += [reduce(numpy.dot, (mo.T.conj(), h1ao[k], mo))
                  for k, mo in enumerate(mo_coeff_b)]
        numpy.set_printoptions(precision=6)
        
        fock_a = fock_hf_a.copy()
        fock_b = fock_hf_b.copy()
        
        c0_a = c0_hf_a
        c0_b = c0_hf_b

        ene_hf = 0

        for k in range(nkpts):
            for i in range(nocca):
                ene_hf += fock_a[k][i, i].real / nkpts
            for i in range(noccb):
                ene_hf += fock_b[k][i, i].real / nkpts

        c0_hf = c0_hf_a + c0_hf_b
        ene_hf += c0_hf + nuc

        if  mp.second_order:
            mp.ampf = 1.0
        
        #####################
        ### MP1 amplitude
        #tmp1, tmp1_bar, h2mo_ovgg = (mp, mo_energy, mo_coeff)
        
        #####################
        ### BCH 1st order  
        c0_1st_a, c0_1st_b, c1a, c1b = first_BCH(mp, (mo_ea, mo_eb), 
                                                 (mo_coeff_a, mo_coeff_b), 
                                                 fock_hf_a, fock_hf_b)
        
        print("c1a + c1a.T.conj():")
        print(c1a[0] + c1a[0].T.conj())
        print("c1b + c1b.T.conj():")
        print(c1b[0] + c1b[0].T.conj())   
        for k in range(nkpts):
            fock_a[k] += 0.5*(c1a[k] + c1a[k].T.conj())
            fock_b[k] += 0.5*(c1b[k] + c1b[k].T.conj())
            #print(abs(fock[k] - fock[k].T.conj()) < 1e-15)
        #####################
        ### BCH 2nd order  
        enea = 0
        eneb = 0

        for k in range(nkpts):
            for i in range(nocca):
                enea += fock_a[k][i, i].real / nkpts
            for i in range(noccb):
                eneb += fock_b[k][i, i].real / nkpts
        ene = enea + eneb
        ene_tot = ene + c0_a + c0_b + c0_1st_a + c0_1st_b + nuc
        print('e_corr = ',ene_tot - ene_hf) 
        de = abs(ene_tot - ene_old)
        ene_old = ene_tot
        tracemalloc.start(25)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        stat = top_stats[:10]
        total_mem = sum(stat.size for stat in top_stats)
        print()
        print('iter = %d'%it, ' ene = %8.8f'%ene_tot, ' ene diff = %8.8f'%de, flush=True)
        #print("Total allocated size: %.3f Mb" % (total_mem / 1024**2))
        print()
        nk = 0
        if de < mp.thresh:
            break

        ## diagonalizing correlated Fock 
        new_mo_coeff_a = numpy.empty_like(mo_coeff[0])
        new_mo_coeff_b = numpy.empty_like(mo_coeff[1])
        new_mo_energy_a = numpy.empty_like(mo_ea, dtype=complex)
        new_mo_energy_b = numpy.empty_like(mo_eb, dtype=complex)
        Ua = numpy.empty_like(mo_coeff[0])
        Ub = numpy.empty_like(mo_coeff[1])
        for k in range(nkpts):
            # Alpha spin
            new_mo_energy_a[k], Ua = scipy.linalg.eigh(fock_a[k])
            new_mo_coeff_a[k] = numpy.dot(mo_coeff[0][k], Ua)

            mo_ea[k] = new_mo_energy_a[k].real
            mo_coeff[0][k] = new_mo_coeff_a[k]
            
            # Beta spin
            new_mo_energy_b[k], Ub = scipy.linalg.eigh(fock_b[k])
            new_mo_coeff_b[k] = numpy.dot(mo_coeff[1][k], Ub)

            mo_eb[k] = new_mo_energy_b[k].real
            mo_coeff[1][k] = new_mo_coeff_b[k]
            #mp.mo_coeff  = mo_coeff
    IPa, EAa, IPb, EAb = make_IPEA(mp, (mo_ea, mo_eb), (mo_coeff_a, mo_coeff_b))
    print("\n=== Alpha Spin ===")
    print(f"IPa_v1 (HOMO)   = {IPa[0] - mo_ea[nk][nocca-1]:.8f}")
    print(f"EAa_c1 (LUMO)   = {EAa[0] - mo_ea[nk][nocca]:.8f}")
    print(f"IPa_v2 (HOMO-1) = {IPa[1] - mo_ea[nk][nocca-2]:.8f}")
    print(f"EAa_c2 (LUMO+1) = {EAa[1] - mo_ea[nk][nocca+1]:.8f}")
    print(f"IPa_v3 (HOMO-2) = {IPa[2] - mo_ea[nk][nocca-3]:.8f}")
    print(f"EAa_c3 (LUMO+2) = {EAa[2] - mo_ea[nk][nocca+2]:.8f}")

    print("\n=== Beta Spin ===")
    print(f"IPb_v1 (HOMO)   = {IPb[0] - mo_eb[nk][noccb-1]:.8f}")
    print(f"EAb_c1 (LUMO)   = {EAb[0] - mo_eb[nk][noccb]:.8f}")
    print(f"IPb_v2 (HOMO-1) = {IPb[1] - mo_eb[nk][noccb-2]:.8f}")
    print(f"EAb_c2 (LUMO+1) = {EAb[1] - mo_eb[nk][noccb+1]:.8f}")
    print(f"IPb_v3 (HOMO-2) = {IPb[2] - mo_eb[nk][noccb-3]:.8f}")
    print(f"EAb_c3 (LUMO+2) = {EAb[2] - mo_eb[nk][noccb+2]:.8f}")
    
    return ene_tot, mo_ea, mo_eb, IPa, EAa, IPb, EAb

#################################################################################################################


def make_veff(mp, mo_coeff, mo_energy):
    mo_coeff_a, mo_coeff_b = mo_coeff
    mo_ea, mo_eb = mo_energy
    nmoa, nmob = mp.get_nmo()
    nocca, noccb = mp.get_nocc()
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    nkpts = len(mo_ea)

    dm = mp._scf.make_rdm1()
    veff_ao_a, veff_ao_b = mp._scf.get_veff(mp._scf.cell, dm)

    veff_a = numpy.zeros((nkpts, nmoa, nmoa), dtype=complex)
    veff_b = numpy.zeros((nkpts, nmob, nmob), dtype=complex)

    for kp in range(nkpts):
        veff_a[kp] = numpy.matmul(
            mo_coeff_a[kp].T.conj(), numpy.matmul(veff_ao_a[kp], mo_coeff_a[kp])
        )
        veff_b[kp] = numpy.matmul(
            mo_coeff_b[kp].T.conj(), numpy.matmul(veff_ao_b[kp], mo_coeff_b[kp])
        )
    
    c0_hf_a = 0
    c0_hf_b = 0
    for kp in range(nkpts):
        for i in range(nocca):
            c0_hf_a -= 0.5 * veff_a[kp][i, i].real
        for i in range(noccb):
            c0_hf_b -= 0.5 * veff_b[kp][i, i].real
            
    c0_hf_a /= nkpts
    c0_hf_b /= nkpts
    
    return veff_ao_a, veff_ao_b, veff_a, veff_b, c0_hf_a, c0_hf_b


def ene_denom(mp, mo_energy, ki, ka, kj, kb):
    mo_ea, mo_eb = mo_energy
    nmoa, nmob = mp.get_nmo()
    nocca, noccb = mp.get_nocc()
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    nkpts = len(mo_ea)

    (nonzero_opadding_a, nonzero_vpadding_a), (nonzero_opadding_b, nonzero_vpadding_b) = padding_k_idx(mp, kind="split")
    mo_e_oa = [mo_ea[k][:nocca] for k in range(nkpts)]
    mo_e_va = [mo_ea[k][nocca:] for k in range(nkpts)]
    mo_e_ob = [mo_eb[k][:noccb] for k in range(nkpts)]
    mo_e_vb = [mo_eb[k][noccb:] for k in range(nkpts)]

    # 1. SPIN ALPHA - ALPHA 
    eia_a = LARGE_DENOM * numpy.ones((nocca, nvira), dtype=mo_ea[0].dtype)
    n0_ovp_ia = numpy.ix_(nonzero_opadding_a[ki], nonzero_vpadding_a[ka])
    eia_a[n0_ovp_ia] = (mo_e_oa[ki][:,None] - mo_e_va[ka])[n0_ovp_ia]
    
    ejb_a = LARGE_DENOM * numpy.ones((nocca, nvira), dtype=mo_ea[0].dtype)
    n0_ovp_jb = numpy.ix_(nonzero_opadding_a[kj], nonzero_vpadding_a[kb])
    ejb_a[n0_ovp_jb] = (mo_e_oa[kj][:,None] - mo_e_va[kb])[n0_ovp_jb]
    
    ejh_a = LARGE_DENOM * numpy.ones((nocca, nocca), dtype=mo_ea[0].dtype)
    n0_ovp_jh = numpy.ix_(nonzero_opadding_a[kj], nonzero_opadding_a[kb])
    ejh_a[n0_ovp_jh] = (mo_e_oa[kj][:,None] - mo_e_oa[kb])[n0_ovp_jh]
    
    elb_a = LARGE_DENOM * numpy.ones((nvira, nvira), dtype=mo_ea[0].dtype)
    n0_ovp_lb = numpy.ix_(nonzero_vpadding_a[kj], nonzero_vpadding_a[kb]) 
    elb_a[n0_ovp_lb] = (mo_e_va[kj][:,None] - mo_e_va[kb])[n0_ovp_lb]

    e_iajb_a = lib.direct_sum('ia,jb -> iajb', eia_a, ejb_a)
    e_iajh_a = lib.direct_sum('ia,jh -> iajh', eia_a, ejh_a)
    e_ialb_a = lib.direct_sum('ia,lb -> ialb', eia_a, elb_a)

   
    # 2. SPIN BETA - BETA 
    eia_b = LARGE_DENOM * numpy.ones((noccb, nvirb), dtype=mo_eb[0].dtype)
    n0_ovp_ia = numpy.ix_(nonzero_opadding_b[ki], nonzero_vpadding_b[ka])
    eia_b[n0_ovp_ia] = (mo_e_ob[ki][:,None] - mo_e_vb[ka])[n0_ovp_ia]
    
    ejb_b = LARGE_DENOM * numpy.ones((noccb, nvirb), dtype=mo_eb[0].dtype)
    n0_ovp_jb = numpy.ix_(nonzero_opadding_b[kj], nonzero_vpadding_b[kb])
    ejb_b[n0_ovp_jb] = (mo_e_ob[kj][:,None] - mo_e_vb[kb])[n0_ovp_jb]
    
    ejh_b = LARGE_DENOM * numpy.ones((noccb, noccb), dtype=mo_eb[0].dtype)
    n0_ovp_jh = numpy.ix_(nonzero_opadding_b[kj], nonzero_opadding_b[kb])
    ejh_b[n0_ovp_jh] = (mo_e_ob[kj][:,None] - mo_e_ob[kb])[n0_ovp_jh]
    
    elb_b = LARGE_DENOM * numpy.ones((nvirb, nvirb), dtype=mo_eb[0].dtype) 
    n0_ovp_lb = numpy.ix_(nonzero_vpadding_b[kj], nonzero_vpadding_b[kb]) 
    elb_b[n0_ovp_lb] = (mo_e_vb[kj][:,None] - mo_e_vb[kb])[n0_ovp_lb]

    e_iajb_b = lib.direct_sum('ia,jb -> iajb', eia_b, ejb_b)
    e_iajh_b = lib.direct_sum('ia,jh -> iajh', eia_b, ejh_b)
    e_ialb_b = lib.direct_sum('ia,lb -> ialb', eia_b, elb_b)

    # ALPHA - BETA VÀ BETA - ALPHA 
    # Electron 1 là alpha (eia_a), Electron 2 là beta (ejb_b, ejh_b, elb_b)
    e_iajb_ab = lib.direct_sum('ia,jb -> iajb', eia_a, ejb_b)
    e_iajh_ab = lib.direct_sum('ia,jh -> iajh', eia_a, ejh_b)
    e_ialb_ab = lib.direct_sum('ia,lb -> ialb', eia_a, elb_b)

    # Electron 1 là beta (eia_b), Electron 2 là alpha (ejb_a, ejh_a, elb_a)
    e_iajb_ba = lib.direct_sum('ia,jb -> iajb', eia_b, ejb_a)
    e_iajh_ba = lib.direct_sum('ia,jh -> iajh', eia_b, ejh_a)
    e_ialb_ba = lib.direct_sum('ia,lb -> ialb', eia_b, elb_a)

    return e_iajb_a, e_iajh_a, e_ialb_a, e_iajb_b, e_iajh_b, e_ialb_b, e_iajb_ab, e_iajh_ab, e_ialb_ab, e_iajb_ba, e_iajh_ba, e_ialb_ba

def first_BCH(mp, mo_energy, mo_coeff, fock_hf_a, fock_hf_b):
    mo_ea, mo_eb = mo_energy
    mo_coeff_a, mo_coeff_b = mo_coeff
    
    nmoa, nmob = mp.get_nmo()
    nocca, noccb = mp.get_nocc()
    
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    nkpts = len(mo_ea)

    homo_a_idx = nocca - 1
    lumo_a_idx = 0
    homo_b_idx = noccb - 1
    lumo_b_idx = 0
    kpts = mp.kpts
    if hasattr(mp, 'khelper'):
        kconserv = mp.khelper.kconserv
    else:
        kconserv = kpts_helper.get_kconserv(mp._scf.cell, kpts)
        
    fao2mo = mp._scf.with_df.ao2mo
    
    # 1. SPIN ALPHA - ALPHA (aa)
    # Electron 1: alpha (a), Electron 2: alpha (a)
    tmp1_aa = numpy.zeros((nkpts, nocca, nvira, nocca, nvira), dtype=complex)
    tmp1_bar_aa = numpy.zeros((nkpts, nocca, nvira, nocca, nvira), dtype=complex)
    tmp1_bar_iajh_aa = numpy.zeros((nkpts, nkpts, nocca, nvira, nocca, nocca), dtype=complex)
    w_iajh_aa = numpy.zeros((nkpts, nkpts, nocca, nvira, nocca, nocca), dtype=complex)
    tmp1_bar_ialb_aa = numpy.zeros((nkpts, nkpts, nocca, nvira, nvira, nvira), dtype=complex)
    w_ialb_aa = numpy.zeros((nkpts, nkpts, nocca, nvira, nvira, nvira), dtype=complex)
    
    h2mo_ovgg_aa = numpy.zeros((nkpts, nocca, nvira, nmoa, nmoa), dtype=complex)
    h2mo_gggg_aa = numpy.zeros((nkpts, nmoa, nmoa, nmoa, nmoa), dtype=complex)
    h2mo_ovov_aa = numpy.zeros((nkpts, nocca, nvira, nocca, nvira), dtype=complex)
    h2mo_ovog_aa = numpy.zeros((nkpts, nocca, nvira, nocca, nmoa), dtype=complex)
    h2mo_ovgv_aa = numpy.zeros((nkpts, nocca, nvira, nmoa, nvira), dtype=complex)
    h2mo_ovoo_aa = numpy.zeros((nkpts, nkpts, nocca, nvira, nocca, nocca), dtype=complex)
    h2mo_ovvv_aa = numpy.zeros((nkpts, nkpts, nocca, nvira, nvira, nvira), dtype=complex)
    
    h2mo_vooo_aa = numpy.zeros((nkpts, nvira, nocca, nocca, nocca), dtype=complex)
    tmp1_bar_ahij_aa = numpy.zeros((nkpts, nvira, nocca, nocca, nocca), dtype=complex)
    w_ahij_aa = numpy.zeros((nkpts, nvira, nocca, nocca, nocca), dtype=complex)
    h2mo_jbia_aa = numpy.zeros((nocca, nvira, nocca, nvira), dtype=complex)
    h2mo_jaib_aa = numpy.zeros((nocca, nvira, nocca, nvira), dtype=complex)

    # 2. SPIN BETA - BETA (bb)
    # Electron 1: beta (b), Electron 2: beta (b)
    tmp1_bb = numpy.zeros((nkpts, noccb, nvirb, noccb, nvirb), dtype=complex)
    tmp1_bar_bb = numpy.zeros((nkpts, noccb, nvirb, noccb, nvirb), dtype=complex)
    tmp1_bar_iajh_bb = numpy.zeros((nkpts, nkpts, noccb, nvirb, noccb, noccb), dtype=complex)
    w_iajh_bb = numpy.zeros((nkpts, nkpts, noccb, nvirb, noccb, noccb), dtype=complex)
    tmp1_bar_ialb_bb = numpy.zeros((nkpts, nkpts, noccb, nvirb, nvirb, nvirb), dtype=complex)
    w_ialb_bb = numpy.zeros((nkpts, nkpts, noccb, nvirb, nvirb, nvirb), dtype=complex)
    
    h2mo_ovgg_bb = numpy.zeros((nkpts, noccb, nvirb, nmob, nmob), dtype=complex)
    h2mo_gggg_bb = numpy.zeros((nkpts, nmob, nmob, nmob, nmob), dtype=complex)
    h2mo_ovov_bb = numpy.zeros((nkpts, noccb, nvirb, noccb, nvirb), dtype=complex)
    h2mo_ovog_bb = numpy.zeros((nkpts, noccb, nvirb, noccb, nmob), dtype=complex)
    h2mo_ovgv_bb = numpy.zeros((nkpts, noccb, nvirb, nmob, nvirb), dtype=complex)
    h2mo_ovoo_bb = numpy.zeros((nkpts, nkpts, noccb, nvirb, noccb, noccb), dtype=complex)
    h2mo_ovvv_bb = numpy.zeros((nkpts, nkpts, noccb, nvirb, nvirb, nvirb), dtype=complex)
    
    h2mo_vooo_bb = numpy.zeros((nkpts, nvirb, noccb, noccb, noccb), dtype=complex)
    tmp1_bar_ahij_bb = numpy.zeros((nkpts, nvirb, noccb, noccb, noccb), dtype=complex)
    w_ahij_bb = numpy.zeros((nkpts, nvirb, noccb, noccb, noccb), dtype=complex)
    h2mo_jbia_bb = numpy.zeros((noccb, nvirb, noccb, nvirb), dtype=complex)
    h2mo_jaib_bb = numpy.zeros((noccb, nvirb, noccb, nvirb), dtype=complex)

    # 3. SPIN ALPHA - BETA (ab)
    # Electron 1: alpha (a), Electron 2: beta (b)
    tmp1_ab = numpy.zeros((nkpts, nocca, nvira, noccb, nvirb), dtype=complex)
    tmp1_bar_ab = numpy.zeros((nkpts, nocca, nvira, noccb, nvirb), dtype=complex)
    tmp1_bar_iajh_ab = numpy.zeros((nkpts, nkpts, nocca, nvira, noccb, noccb), dtype=complex)
    w_iajh_ab = numpy.zeros((nkpts, nkpts, nocca, nvira, noccb, noccb), dtype=complex)
    tmp1_bar_ialb_ab = numpy.zeros((nkpts, nkpts, nocca, nvira, nvirb, nvirb), dtype=complex)
    w_ialb_ab = numpy.zeros((nkpts, nkpts, nocca, nvira, nvirb, nvirb), dtype=complex)
    
    h2mo_ovgg_ab = numpy.zeros((nkpts, nocca, nvira, nmob, nmob), dtype=complex)
    h2mo_gggg_ab = numpy.zeros((nkpts, nmoa, nmoa, nmob, nmob), dtype=complex)
    h2mo_ovov_ab = numpy.zeros((nkpts, nocca, nvira, noccb, nvirb), dtype=complex)
    h2mo_ovog_ab = numpy.zeros((nkpts, nocca, nvira, noccb, nmob), dtype=complex)
    h2mo_ovgv_ab = numpy.zeros((nkpts, nocca, nvira, nmob, nvirb), dtype=complex)
    h2mo_ovoo_ab = numpy.zeros((nkpts, nkpts, nocca, nvira, noccb, noccb), dtype=complex)
    h2mo_ovvv_ab = numpy.zeros((nkpts, nkpts, nocca, nvira, nvirb, nvirb), dtype=complex)
    
    h2mo_vooo_ab = numpy.zeros((nkpts, nvira, nocca, noccb, noccb), dtype=complex)
    tmp1_bar_ahij_ab = numpy.zeros((nkpts, nvira, nocca, noccb, noccb), dtype=complex)
    w_ahij_ab = numpy.zeros((nkpts, nvira, nocca, noccb, noccb), dtype=complex)
    h2mo_jbia_ab = numpy.zeros((nocca, nvira, noccb, nvirb), dtype=complex)
    h2mo_jaib_ab = numpy.zeros((nocca, nvira, noccb, nvirb), dtype=complex)

    # 4. SPIN BETA - ALPHA (ba)
    # Electron 1: beta (b), Electron 2: alpha (a)
    tmp1_ba = numpy.zeros((nkpts, noccb, nvirb, nocca, nvira), dtype=complex)
    tmp1_bar_ba = numpy.zeros((nkpts, noccb, nvirb, nocca, nvira), dtype=complex)
    tmp1_bar_iajh_ba = numpy.zeros((nkpts, nkpts, noccb, nvirb, nocca, nocca), dtype=complex)
    w_iajh_ba = numpy.zeros((nkpts, nkpts, noccb, nvirb, nocca, nocca), dtype=complex)
    tmp1_bar_ialb_ba = numpy.zeros((nkpts, nkpts, noccb, nvirb, nvira, nvira), dtype=complex)
    w_ialb_ba = numpy.zeros((nkpts, nkpts, noccb, nvirb, nvira, nvira), dtype=complex)
    
    h2mo_ovgg_ba = numpy.zeros((nkpts, noccb, nvirb, nmoa, nmoa), dtype=complex)
    h2mo_gggg_ba = numpy.zeros((nkpts, nmob, nmob, nmoa, nmoa), dtype=complex)
    h2mo_ovov_ba = numpy.zeros((nkpts, noccb, nvirb, nocca, nvira), dtype=complex)
    h2mo_ovog_ba = numpy.zeros((nkpts, noccb, nvirb, nocca, nmoa), dtype=complex)
    h2mo_ovgv_ba = numpy.zeros((nkpts, noccb, nvirb, nmoa, nvira), dtype=complex)
    h2mo_ovoo_ba = numpy.zeros((nkpts, nkpts, noccb, nvirb, nocca, nocca), dtype=complex)
    h2mo_ovvv_ba = numpy.zeros((nkpts, nkpts, noccb, nvirb, nvira, nvira), dtype=complex)
    
    h2mo_vooo_ba = numpy.zeros((nkpts, nvirb, noccb, nocca, nocca), dtype=complex)
    tmp1_bar_ahij_ba = numpy.zeros((nkpts, nvirb, noccb, nocca, nocca), dtype=complex)
    w_ahij_ba = numpy.zeros((nkpts, nvirb, noccb, nocca, nocca), dtype=complex)
    h2mo_jbia_ba = numpy.zeros((noccb, nvirb, nocca, nvira), dtype=complex)
    h2mo_jaib_ba = numpy.zeros((noccb, nvirb, nocca, nvira), dtype=complex)
    
    c1a = numpy.zeros((nkpts, nmoa, nmoa), dtype=complex)
    c1b = numpy.zeros((nkpts, nmob, nmob), dtype=complex)
    c2a = numpy.zeros((nkpts, nmoa, nmoa), dtype=complex)
    c2b = numpy.zeros((nkpts, nmob, nmob), dtype=complex)
    
    y1_a = numpy.zeros((nkpts, nvira, nocca), dtype=complex)
    y1_b = numpy.zeros((nkpts, nvirb, noccb), dtype=complex)

    # y2, y3: 
    y2_aa = numpy.zeros((nkpts, nocca, nvira, nocca, nvira), dtype=complex)
    y2_bb = numpy.zeros((nkpts, noccb, nvirb, noccb, nvirb), dtype=complex)
    y2_ab = numpy.zeros((nkpts, nocca, nvira, noccb, nvirb), dtype=complex)
    y2_ba = numpy.zeros((nkpts, noccb, nvirb, nocca, nvira), dtype=complex)

    y3_aa = numpy.zeros((nkpts, nocca, nvira, nocca, nvira), dtype=complex)
    y3_bb = numpy.zeros((nkpts, noccb, nvirb, noccb, nvirb), dtype=complex)
    y3_ab = numpy.zeros((nkpts, nocca, nvira, noccb, nvirb), dtype=complex)
    y3_ba = numpy.zeros((nkpts, noccb, nvirb, nocca, nvira), dtype=complex)

    # y4: 
    y4_a = numpy.zeros((nkpts, nocca, nocca), dtype=complex)
    y4_b = numpy.zeros((nkpts, noccb, noccb), dtype=complex)

    # y5: 
    y5_a = numpy.zeros((nkpts, nvira, nvira), dtype=complex)
    y5_b = numpy.zeros((nkpts, nvirb, nvirb), dtype=complex)
    
    c0_1st_a = 0
    c0_1st_b = 0
    c0_2nd_a = 0
    c0_2nd_b = 0
    IPa_v1 = IPa_v2 = IPa_v3 = 0
    EAa_c1 = EAa_c2 = EAa_c3 = 0
    IPb_v1 = IPb_v2 = IPb_v3 = 0
    EAb_c1 = EAb_c2 = EAb_c3 = 0
    print("mo energy alpha", mo_ea)
    print("mo energy beta", mo_eb)
    nk = 0
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                kp = kj
                kq = kb
                
                o_i_a = mo_coeff_a[ki][:, :nocca]
                v_a_a = mo_coeff_a[ka][:, nocca:]
                g_p_a = mo_coeff_a[kp]
                g_q_a = mo_coeff_a[kq]

                o_i_b = mo_coeff_b[ki][:, :noccb]
                v_a_b = mo_coeff_b[ka][:, noccb:]
                g_p_b = mo_coeff_b[kp]
                g_q_b = mo_coeff_b[kq]

                # --- aa ---
                eri_aa = fao2mo((o_i_a, v_a_a, g_p_a, g_q_a),
                                (mp.kpts[ki], mp.kpts[ka], mp.kpts[kp], mp.kpts[kq]),
                                compact=False).reshape(nocca, nvira, nmoa, nmoa) / nkpts
                h2mo_ovov_aa[ka] = eri_aa[:, :, :nocca, nocca:]
                h2mo_ovgv_aa[ka] = eri_aa[:, :, :, nocca:]
                h2mo_ovog_aa[ka] = eri_aa[:, :, :nocca, :]
                h2mo_ovvv_aa[ka] = eri_aa[:, :, nocca:, nocca:]
                del eri_aa

                # --- bb ---
                eri_bb = fao2mo((o_i_b, v_a_b, g_p_b, g_q_b),
                                (mp.kpts[ki], mp.kpts[ka], mp.kpts[kp], mp.kpts[kq]),
                                compact=False).reshape(noccb, nvirb, nmob, nmob) / nkpts
                h2mo_ovov_bb[ka] = eri_bb[:, :, :noccb, noccb:]
                h2mo_ovgv_bb[ka] = eri_bb[:, :, :, noccb:]
                h2mo_ovog_bb[ka] = eri_bb[:, :, :noccb, :]
                h2mo_ovvv_bb[ka] = eri_bb[:, :, noccb:, noccb:]
                del eri_bb

                # --- ab (Electron 1: alpha, Electron 2: beta) ---
                eri_ab = fao2mo((o_i_a, v_a_a, g_p_b, g_q_b),
                                (mp.kpts[ki], mp.kpts[ka], mp.kpts[kp], mp.kpts[kq]),
                                compact=False).reshape(nocca, nvira, nmob, nmob) / nkpts
                h2mo_ovov_ab[ka] = eri_ab[:, :, :noccb, noccb:]
                h2mo_ovgv_ab[ka] = eri_ab[:, :, :, noccb:]
                h2mo_ovog_ab[ka] = eri_ab[:, :, :noccb, :]
                h2mo_ovvv_ab[ka] = eri_ab[:, :, noccb:, noccb:]
                del eri_ab

                # --- ba (Electron 1: beta, Electron 2: alpha) ---
                eri_ba = fao2mo((o_i_b, v_a_b, g_p_a, g_q_a),
                                (mp.kpts[ki], mp.kpts[ka], mp.kpts[kp], mp.kpts[kq]),
                                compact=False).reshape(noccb, nvirb, nmoa, nmoa) / nkpts
                h2mo_ovov_ba[ka] = eri_ba[:, :, :nocca, nocca:]
                h2mo_ovgv_ba[ka] = eri_ba[:, :, :, nocca:]
                h2mo_ovog_ba[ka] = eri_ba[:, :, :nocca, :]
                h2mo_ovvv_ba[ka] = eri_ba[:, :, nocca:, nocca:]
                del eri_ba

            # tmp1
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                
                (e_iajb_a, e_iajh_a, e_ialb_a,
                 e_iajb_b, e_iajh_b, e_ialb_b,
                 e_iajb_ab, e_iajh_ab, e_ialb_ab,
                 e_iajb_ba, e_iajh_ba, e_ialb_ba) = ene_denom(mp, (mo_ea, mo_eb), ki, ka, kj, kb)
                
                w_iajb_aa = h2mo_ovov_aa[ka] - h2mo_ovov_aa[kb].transpose(0, 3, 2, 1)
                tmp1_aa[ka] = (h2mo_ovov_aa[ka] / e_iajb_a).conj()
                tmp1_bar_aa[ka] = (w_iajb_aa / e_iajb_a).conj()

                w_iajb_bb = h2mo_ovov_bb[ka] - h2mo_ovov_bb[kb].transpose(0, 3, 2, 1)
                tmp1_bb[ka] = (h2mo_ovov_bb[ka] / e_iajb_b).conj()
                tmp1_bar_bb[ka] = (w_iajb_bb / e_iajb_b).conj()

                w_iajb_ab = h2mo_ovov_ab[ka]
                tmp1_ab[ka] = (h2mo_ovov_ab[ka] / e_iajb_ab).conj()
                tmp1_bar_ab[ka] = tmp1_ab[ka]

                w_iajb_ba = h2mo_ovov_ba[ka]
                tmp1_ba[ka] = (h2mo_ovov_ba[ka] / e_iajb_ba).conj()
                tmp1_bar_ba[ka] = tmp1_ba[ka]

                c1a[kb, nocca:, :] -= 2.0*numpy.einsum('iajb, iajp -> bp', tmp1_bar_aa[ka], h2mo_ovog_aa[ka])
                c1a[kj, :, :nocca] += 2.0*numpy.einsum('iajb, iapb -> pj', tmp1_bar_aa[ka], h2mo_ovgv_aa[ka])

                c1b[kb, noccb:, :] -= 2.0*numpy.einsum('iajb, iajp -> bp', tmp1_bar_bb[ka], h2mo_ovog_bb[ka])
                c1b[kj, :, :noccb] += 2.0*numpy.einsum('iajb, iapb -> pj', tmp1_bar_bb[ka], h2mo_ovgv_bb[ka])

                c1b[kb, noccb:, :] -= 2.0*numpy.einsum('iajb, iajp -> bp', tmp1_bar_ab[ka], h2mo_ovog_ab[ka])
                c1b[kj, :, :noccb] += 2.0*numpy.einsum('iajb, iapb -> pj', tmp1_bar_ab[ka], h2mo_ovgv_ab[ka])

                c1a[kb, nocca:, :] -= 2.0*numpy.einsum('iajb, iajp -> bp', tmp1_bar_ba[ka], h2mo_ovog_ba[ka])
                c1a[kj, :, :nocca] += 2.0*numpy.einsum('iajb, iapb -> pj', tmp1_bar_ba[ka], h2mo_ovgv_ba[ka])

                if ki == ka:
                    c1a[kj, nocca:, :nocca] += 2.0*numpy.einsum('iajb, ia -> bj', tmp1_bar_aa[ka], fock_hf_a[ka, :nocca, nocca:])
                    c1b[kj, noccb:, :noccb] += 2.0*numpy.einsum('iajb, ia -> bj', tmp1_bar_bb[ka], fock_hf_b[ka, :noccb, noccb:])
                    c1b[kj, noccb:, :noccb] += 2.0*numpy.einsum('iajb, ia -> bj', tmp1_bar_ab[ka], fock_hf_a[ka, :nocca, nocca:])
                    c1a[kj, nocca:, :nocca] += 2.0*numpy.einsum('iajb, ia -> bj', tmp1_bar_ba[ka], fock_hf_b[ka, :noccb, noccb:])

                c0_1st_a -= 0.5 * numpy.einsum('iajb, iajb -> ', tmp1_bar_aa[ka], h2mo_ovov_aa[ka]).real
                c0_1st_b -= 0.5 * numpy.einsum('iajb, iajb -> ', tmp1_bar_bb[ka], h2mo_ovov_bb[ka]).real
                c0_1st_a -= 0.5 * numpy.einsum('iajb, iajb -> ', tmp1_bar_ba[ka], h2mo_ovov_ba[ka]).real
                c0_1st_b -= 0.5 * numpy.einsum('iajb, iajb -> ', tmp1_bar_ab[ka], h2mo_ovov_ab[ka]).real

                # 2ND ORDER BCH 
                if mp.second_order:
                    # y1 
                    if ki == ka:
                        y1_a[kj] += numpy.einsum('ia, iajb -> bj', fock_hf_a[ka, :nocca, nocca:], tmp1_bar_aa[ka])
                        y1_b[kj] += numpy.einsum('ia, iajb -> bj', fock_hf_b[ka, :noccb, noccb:], tmp1_bar_bb[ka])
                        y1_b[kj] += numpy.einsum('ia, iajb -> bj', fock_hf_a[ka, :nocca, nocca:], tmp1_bar_ab[ka])
                        y1_a[kj] += numpy.einsum('ia, iajb -> bj', fock_hf_b[ka, :noccb, noccb:], tmp1_bar_ba[ka])

                    # y2
                    y2_aa[ka] = numpy.einsum('ca, iclb -> ialb', fock_hf_a[ka, nocca:, nocca:], tmp1_bar_aa[ka].conj())
                    y2_bb[ka] = numpy.einsum('ca, iclb -> ialb', fock_hf_b[ka, noccb:, noccb:], tmp1_bar_bb[ka].conj())
                    y2_ab[ka] = numpy.einsum('ca, iclb -> ialb', fock_hf_a[ka, nocca:, nocca:], tmp1_bar_ab[ka].conj())
                    y2_ba[ka] = numpy.einsum('ca, iclb -> ialb', fock_hf_b[ka, noccb:, noccb:], tmp1_bar_ba[ka].conj())

                    # y3 
                    y3_aa[ka] = numpy.einsum('ik, kalb -> ialb', fock_hf_a[ki, :nocca, :nocca], tmp1_bar_aa[ka].conj())
                    y3_bb[ka] = numpy.einsum('ik, kalb -> ialb', fock_hf_b[ki, :noccb, :noccb], tmp1_bar_bb[ka].conj())
                    y3_ab[ka] = numpy.einsum('ik, kalb -> ialb', fock_hf_a[ki, :nocca, :nocca], tmp1_bar_ab[ka].conj())
                    y3_ba[ka] = numpy.einsum('ik, kalb -> ialb', fock_hf_b[ki, :noccb, :noccb], tmp1_bar_ba[ka].conj())

                    # y4 
                    y4_a[ki] += numpy.einsum('iajb, kajb -> ki', tmp1_aa[ka], tmp1_bar_aa[ka].conj())
                    y4_b[ki] += numpy.einsum('iajb, kajb -> ki', tmp1_bb[ka], tmp1_bar_bb[ka].conj())
                    y4_a[ki] += numpy.einsum('iajb, kajb -> ki', tmp1_ab[ka], tmp1_bar_ab[ka].conj())
                    y4_b[ki] += numpy.einsum('iajb, kajb -> ki', tmp1_ba[ka], tmp1_bar_ba[ka].conj())

                    # y5 
                    y5_a[ka] += numpy.einsum('iajb, icjb -> ac', tmp1_aa[ka], tmp1_bar_aa[ka].conj())
                    y5_b[ka] += numpy.einsum('iajb, icjb -> ac', tmp1_bb[ka], tmp1_bar_bb[ka].conj())
                    y5_a[ka] += numpy.einsum('iajb, icjb -> ac', tmp1_ab[ka], tmp1_bar_ab[ka].conj())
                    y5_b[ka] += numpy.einsum('iajb, icjb -> ac', tmp1_ba[ka], tmp1_bar_ba[ka].conj())

                    c2a[kj, :nocca, :nocca] += numpy.einsum('ialb, iajb -> lj', y2_aa[ka], tmp1_aa[ka]) + numpy.einsum('ialb, iajb -> lj', y2_ba[ka], tmp1_ba[ka])
                    c2b[kj, :noccb, :noccb] += numpy.einsum('ialb, iajb -> lj', y2_bb[ka], tmp1_bb[ka]) + numpy.einsum('ialb, iajb -> lj', y2_ab[ka], tmp1_ab[ka])
                    
                    c2a[ki, :nocca, :nocca] += numpy.einsum('kajb, iajb -> ki', y2_aa[ka], tmp1_aa[ka]) + numpy.einsum('kajb, iajb -> ki', y2_ab[ka], tmp1_ab[ka])
                    c2b[ki, :noccb, :noccb] += numpy.einsum('kajb, iajb -> ki', y2_bb[ka], tmp1_bb[ka]) + numpy.einsum('kajb, iajb -> ki', y2_ba[ka], tmp1_ba[ka])
                    
                    c2a[kb, nocca:, nocca:] -= numpy.einsum('iajd, iajb -> bd', y2_aa[ka], tmp1_aa[ka]) + numpy.einsum('iajd, iajb -> bd', y2_ba[ka], tmp1_ba[ka])
                    c2b[kb, noccb:, noccb:] -= numpy.einsum('iajd, iajb -> bd', y2_bb[ka], tmp1_bb[ka]) + numpy.einsum('iajd, iajb -> bd', y2_ab[ka], tmp1_ab[ka])

                    c2a[kj, :nocca, :nocca] -= numpy.einsum('ialb, iajb -> lj', y3_aa[ka], tmp1_aa[ka]) + numpy.einsum('ialb, iajb -> lj', y3_ba[ka], tmp1_ba[ka])
                    c2b[kj, :noccb, :noccb] -= numpy.einsum('ialb, iajb -> lj', y3_bb[ka], tmp1_bb[ka]) + numpy.einsum('ialb, iajb -> lj', y3_ab[ka], tmp1_ab[ka])
                    
                    c2a[ka, nocca:, nocca:] += numpy.einsum('icjb, iajb -> ac', y3_aa[ka], tmp1_aa[ka]) + numpy.einsum('icjb, iajb -> ac', y3_ab[ka], tmp1_ab[ka])
                    c2b[ka, noccb:, noccb:] += numpy.einsum('icjb, iajb -> ac', y3_bb[ka], tmp1_bb[ka]) + numpy.einsum('icjb, iajb -> ac', y3_ba[ka], tmp1_ba[ka])
                    
                    c2a[kb, nocca:, nocca:] += numpy.einsum('iajd, iajb -> bd', y3_aa[ka], tmp1_aa[ka]) + numpy.einsum('iajd, iajb -> bd', y3_ba[ka], tmp1_ba[ka])
                    c2b[kb, noccb:, noccb:] += numpy.einsum('iajd, iajb -> bd', y3_bb[ka], tmp1_bb[ka]) + numpy.einsum('iajd, iajb -> bd', y3_ab[ka], tmp1_ab[ka])

                    c0_2nd_a -= 2.0 * numpy.einsum('iajb,iajb -> ', y2_aa[ka], tmp1_aa[ka]).real
                    c0_2nd_b -= 2.0 * numpy.einsum('iajb,iajb -> ', y2_bb[ka], tmp1_bb[ka]).real
                    c0_2nd_b -= 2.0 * numpy.einsum('iajb,iajb -> ', y2_ab[ka], tmp1_ab[ka]).real
                    c0_2nd_a -= 2.0 * numpy.einsum('iajb,iajb -> ', y2_ba[ka], tmp1_ba[ka]).real

                    c0_2nd_a += 2.0 * numpy.einsum('iajb,iajb -> ', y3_aa[ka], tmp1_aa[ka]).real
                    c0_2nd_b += 2.0 * numpy.einsum('iajb,iajb -> ', y3_bb[ka], tmp1_bb[ka]).real
                    c0_2nd_b += 2.0 * numpy.einsum('iajb,iajb -> ', y3_ab[ka], tmp1_ab[ka]).real
                    c0_2nd_a += 2.0 * numpy.einsum('iajb,iajb -> ', y3_ba[ka], tmp1_ba[ka]).real

    for ki in range(nkpts):
        if mp.second_order:
            c2a[ki, :nocca, :] -= numpy.einsum('ip, ki -> kp', fock_hf_a[ki, :nocca, :], y4_a[ki])
            c2b[ki, :noccb, :] -= numpy.einsum('ip, ki -> kp', fock_hf_b[ki, :noccb, :], y4_b[ki])
            
            c2a[ki, :, nocca:] -= numpy.einsum('pa, ac -> pc', fock_hf_a[ki, :, nocca:], y5_a[ki])
            c2b[ki, :, noccb:] -= numpy.einsum('pa, ac -> pc', fock_hf_b[ki, :, noccb:], y5_b[ki])

    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                o_i_a = mo_coeff_a[ki][:, :nocca]; o_a_a = mo_coeff_a[ka][:, nocca:]
                o_j_a = mo_coeff_a[kj]; o_b_a = mo_coeff_a[kb]
                
                o_i_b = mo_coeff_b[ki][:, :noccb]; o_a_b = mo_coeff_b[ka][:, noccb:]
                o_j_b = mo_coeff_b[kj]; o_b_b = mo_coeff_b[kb]

                if kb == nk:
                    # aa, bb
                    h2mo_ovoo_aa[ki, ka] = fao2mo((o_i_a, o_a_a, o_j_a[:, :nocca], o_b_a[:, :nocca]),
                                                  (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]), compact=False).reshape(nocca, nvira, nocca, nocca) / nkpts
                    h2mo_ovoo_bb[ki, ka] = fao2mo((o_i_b, o_a_b, o_j_b[:, :noccb], o_b_b[:, :noccb]),
                                                  (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]), compact=False).reshape(noccb, nvirb, noccb, noccb) / nkpts
                    # ab, ba
                    h2mo_ovoo_ab[ki, ka] = fao2mo((o_i_a, o_a_a, o_j_b[:, :noccb], o_b_b[:, :noccb]),
                                                  (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]), compact=False).reshape(nocca, nvira, noccb, noccb) / nkpts
                    h2mo_ovoo_ba[ki, ka] = fao2mo((o_i_b, o_a_b, o_j_a[:, :nocca], o_b_a[:, :nocca]),
                                                  (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]), compact=False).reshape(noccb, nvirb, nocca, nocca) / nkpts

                if kj == nk:
                    # aa, bb
                    h2mo_ovvv_aa[ki, ka] = fao2mo((o_i_a, o_a_a, o_j_a[:, nocca:], o_b_a[:, nocca:]),
                                                  (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]), compact=False).reshape(nocca, nvira, nvira, nvira) / nkpts
                    h2mo_ovvv_bb[ki, ka] = fao2mo((o_i_b, o_a_b, o_j_b[:, noccb:], o_b_b[:, noccb:]),
                                                  (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]), compact=False).reshape(noccb, nvirb, nvirb, nvirb) / nkpts
                    # ab, ba
                    h2mo_ovvv_ab[ki, ka] = fao2mo((o_i_a, o_a_a, o_j_b[:, noccb:], o_b_b[:, noccb:]),
                                                  (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]), compact=False).reshape(nocca, nvira, nvirb, nvirb) / nkpts
                    h2mo_ovvv_ba[ki, ka] = fao2mo((o_i_b, o_a_b, o_j_a[:, nocca:], o_b_a[:, nocca:]),
                                                  (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]), compact=False).reshape(noccb, nvirb, nvira, nvira) / nkpts

                h2mo_ovov_aa[ka] = fao2mo((o_i_a, o_a_a, o_j_a[:, :nocca], o_b_a[:, nocca:]), (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]), compact=False).reshape(nocca, nvira, nocca, nvira) / nkpts
                h2mo_ovov_bb[ka] = fao2mo((o_i_b, o_a_b, o_j_b[:, :noccb], o_b_b[:, noccb:]), (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]), compact=False).reshape(noccb, nvirb, noccb, nvirb) / nkpts
                h2mo_ovov_ab[ka] = fao2mo((o_i_a, o_a_a, o_j_b[:, :noccb], o_b_b[:, noccb:]), (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]), compact=False).reshape(nocca, nvira, noccb, nvirb) / nkpts
                h2mo_ovov_ba[ka] = fao2mo((o_i_b, o_a_b, o_j_a[:, :nocca], o_b_a[:, nocca:]), (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]), compact=False).reshape(noccb, nvirb, nocca, nvira) / nkpts

            # Updated C2, IP, EA
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                (e_iajb_a, e_iajh_a, e_ialb_a,
                 e_iajb_b, e_iajh_b, e_ialb_b,
                 e_iajb_ab, e_iajh_ab, e_ialb_ab,
                 e_iajb_ba, e_iajh_ba, e_ialb_ba) = ene_denom(mp, (mo_ea, mo_eb), ki, ka, kj, kb)

                if kj == kb:
                    # aa, bb 
                    tmp1_bar_aa[ka] = ((h2mo_ovov_aa[ka] - h2mo_ovov_aa[kb].transpose(0, 3, 2, 1)) / e_iajb_a).conj()
                    tmp1_bar_bb[ka] = ((h2mo_ovov_bb[ka] - h2mo_ovov_bb[kb].transpose(0, 3, 2, 1)) / e_iajb_b).conj()
                    
                    # ab, ba 
                    tmp1_bar_ab[ka] = (h2mo_ovov_ab[ka] / e_iajb_ab).conj()
                    tmp1_bar_ba[ka] = (h2mo_ovov_ba[ka] / e_iajb_ba).conj()

                    c2a[kj, :nocca, nocca:] += numpy.einsum('bj, jbkc -> kc', y1_a[ka], tmp1_bar_aa[ka].conj())
                    c2b[kj, :noccb, noccb:] += numpy.einsum('bj, jbkc -> kc', y1_b[ka], tmp1_bar_bb[ka].conj())
                    c2b[kj, :noccb, noccb:] += numpy.einsum('bj, jbkc -> kc', y1_a[ka], tmp1_bar_ab[ka].conj()) # ab -> y1_a, c2b
                    c2a[kj, :nocca, nocca:] += numpy.einsum('bj, jbkc -> kc', y1_b[ka], tmp1_bar_ba[ka].conj()) # ba -> y1_b, c2a
    c0_a = c0_1st_a + c0_2nd_a
    c0_b = c0_1st_b + c0_2nd_b
    c0_a /= nkpts
    c0_b /= nkpts
    
    c1a += c2a
    c1b += c2b
    
    print("c0_a", c0_a)
    print("c0_b", c0_b)
    

    
    return c0_a, c0_b, c1a, c1b

def make_IPEA(mp, mo_energy, mo_coeff):
    mo_ea, mo_eb = mo_energy
    mo_coeff_a, mo_coeff_b = mo_coeff
    nmoa, nmob = mp.get_nmo()
    nocca, noccb = mp.get_nocc()
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    nkpts = len(mp.kpts)
    kd = mp.kpts
    kconserv = mp.khelper.kconserv
    fao2mo = mp._scf.with_df.ao2mo
   
    IPa_v1 = IPa_v2 = IPa_v3 = 0.0
    IPb_v1 = IPb_v2 = IPb_v3 = 0.0
    EAa_c1 = EAa_c2 = EAa_c3 = 0.0
    EAb_c1 = EAb_c2 = EAb_c3 = 0.0
    nk = 0

    # IP Alpha (aa, ba)
    tmp1_bar_iajh_aa = numpy.zeros((nkpts,nkpts,nocca,nvira,nocca,nocca), dtype=complex)
    tmp1_bar_iajh_ba = numpy.zeros((nkpts,nkpts,noccb,nvirb,nocca,nocca), dtype=complex)
    h2mo_ovoo_aa = numpy.zeros((nkpts,nkpts,nocca,nvira,nocca,nocca), dtype=complex)
    h2mo_ovoo_ba = numpy.zeros((nkpts,nkpts,noccb,nvirb,nocca,nocca), dtype=complex)

    # IP Beta (bb, ab)
    tmp1_bar_iajh_bb = numpy.zeros((nkpts,nkpts,noccb,nvirb,noccb,noccb), dtype=complex)
    tmp1_bar_iajh_ab = numpy.zeros((nkpts,nkpts,nocca,nvira,noccb,noccb), dtype=complex)
    h2mo_ovoo_bb = numpy.zeros((nkpts,nkpts,noccb,nvirb,noccb,noccb), dtype=complex)
    h2mo_ovoo_ab = numpy.zeros((nkpts,nkpts,nocca,nvira,noccb,noccb), dtype=complex)

    # EA Alpha (aa, ba)
    h2mo_ovvv_aa = numpy.zeros((nkpts,nkpts,nocca,nvira,nvira,nvira), dtype=complex)
    h2mo_ovvv_ba = numpy.zeros((nkpts,nkpts,noccb,nvirb,nvira,nvira), dtype=complex)

    # EA Beta (bb, ab)
    h2mo_ovvv_bb = numpy.zeros((nkpts,nkpts,noccb,nvirb,nvirb,nvirb), dtype=complex)
    h2mo_ovvv_ab = numpy.zeros((nkpts,nkpts,nocca,nvira,nvirb,nvirb), dtype=complex)

    for ki in range(nkpts):
         for kj in range(nkpts):
             for ka in range(nkpts):
                 kb = kconserv[ki,ka,kj]
                 
                 
                 o_i_a = mo_coeff_a[ki][:,:nocca]
                 o_a_a = mo_coeff_a[ka][:,nocca:]
                 o_j_a = mo_coeff_a[kj]
                 o_b_a = mo_coeff_a[kb]

                 o_i_b = mo_coeff_b[ki][:,:noccb]
                 o_a_b = mo_coeff_b[ka][:,noccb:]
                 o_j_b = mo_coeff_b[kj]
                 o_b_b = mo_coeff_b[kb]

                 if kb==nk:
                     # Alpha IP
                     h2mo_ovoo_aa[ki,ka] = fao2mo((o_i_a, o_a_a, o_j_a[:,:nocca], o_b_a[:,:nocca]),
                                            (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]),
                                            compact=False).reshape(nocca,nvira,nocca,nocca)/nkpts
                     h2mo_ovoo_ba[ki,ka] = fao2mo((o_i_b, o_a_b, o_j_a[:,:nocca], o_b_a[:,:nocca]),
                                            (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]),
                                            compact=False).reshape(noccb,nvirb,nocca,nocca)/nkpts
                     
                     # Beta IP
                     h2mo_ovoo_bb[ki,ka] = fao2mo((o_i_b, o_a_b, o_j_b[:,:noccb], o_b_b[:,:noccb]),
                                            (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]),
                                            compact=False).reshape(noccb,nvirb,noccb,noccb)/nkpts
                     h2mo_ovoo_ab[ki,ka] = fao2mo((o_i_a, o_a_a, o_j_b[:,:noccb], o_b_b[:,:noccb]),
                                            (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]),
                                            compact=False).reshape(nocca,nvira,noccb,noccb)/nkpts
                     
                 if kj==nk:
                     # Alpha EA
                     h2mo_ovvv_aa[ki,ka] = fao2mo((o_i_a, o_a_a, o_j_a[:,nocca:], o_b_a[:,nocca:]),
                                            (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]),
                                            compact=False).reshape(nocca,nvira,nvira,nvira)/nkpts
                     h2mo_ovvv_ba[ki,ka] = fao2mo((o_i_b, o_a_b, o_j_a[:,nocca:], o_b_a[:,nocca:]),
                                            (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]),
                                            compact=False).reshape(noccb,nvirb,nvira,nvira)/nkpts
                     
                     # Beta EA
                     h2mo_ovvv_bb[ki,ka] = fao2mo((o_i_b, o_a_b, o_j_b[:,noccb:], o_b_b[:,noccb:]),
                                            (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]),
                                            compact=False).reshape(noccb,nvirb,nvirb,nvirb)/nkpts
                     h2mo_ovvv_ab[ki,ka] = fao2mo((o_i_a, o_a_a, o_j_b[:,noccb:], o_b_b[:,noccb:]),
                                            (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]),
                                            compact=False).reshape(nocca,nvira,nvirb,nvirb)/nkpts

             for ka in range(nkpts):
                 kb = kconserv[ki,ka,kj]
                 
                 (e_iajb_a, e_iajh_a, e_ialb_a,
                  e_iajb_b, e_iajh_b, e_ialb_b,
                  e_iajb_ab, e_iajh_ab, e_ialb_ab,
                  e_iajb_ba, e_iajh_ba, e_ialb_ba) = ene_denom(mp, (mo_ea, mo_eb), ki, ka, kj, kb)
                 
                 if kb==nk:
                     # --- Tính IP cho Alpha ---
                     w_iajh_aa = h2mo_ovoo_aa[ki,ka] - h2mo_ovoo_aa[kj,ka].transpose(2,1,0,3)
                     tmp1_bar_iajh_aa[ki,ka] = (w_iajh_aa / e_iajh_a).conj()
                     tmp1_bar_iajh_ba[ki,ka] = (h2mo_ovoo_ba[ki,ka] / e_iajh_ba).conj()

                     IPa_v1 += numpy.einsum('iaj, iaj -> ', tmp1_bar_iajh_aa[ki,ka][:,:,:,nocca-1], h2mo_ovoo_aa[ki,ka][:,:,:,nocca-1]).real
                     IPa_v1 += numpy.einsum('iaj, iaj -> ', tmp1_bar_iajh_ba[ki,ka][:,:,:,nocca-1], h2mo_ovoo_ba[ki,ka][:,:,:,nocca-1]).real
                     IPa_v2 += numpy.einsum('iaj, iaj -> ', tmp1_bar_iajh_aa[ki,ka][:,:,:,nocca-2], h2mo_ovoo_aa[ki,ka][:,:,:,nocca-2]).real
                     IPa_v2 += numpy.einsum('iaj, iaj -> ', tmp1_bar_iajh_ba[ki,ka][:,:,:,nocca-2], h2mo_ovoo_ba[ki,ka][:,:,:,nocca-2]).real
                     IPa_v3 += numpy.einsum('iaj, iaj -> ', tmp1_bar_iajh_aa[ki,ka][:,:,:,nocca-3], h2mo_ovoo_aa[ki,ka][:,:,:,nocca-3]).real
                     IPa_v3 += numpy.einsum('iaj, iaj -> ', tmp1_bar_iajh_ba[ki,ka][:,:,:,nocca-3], h2mo_ovoo_ba[ki,ka][:,:,:,nocca-3]).real

                     # --- Tính IP cho Beta ---
                     w_iajh_bb = h2mo_ovoo_bb[ki,ka] - h2mo_ovoo_bb[kj,ka].transpose(2,1,0,3)
                     tmp1_bar_iajh_bb[ki,ka] = (w_iajh_bb / e_iajh_b).conj()
                     tmp1_bar_iajh_ab[ki,ka] = (h2mo_ovoo_ab[ki,ka] / e_iajh_ab).conj()

                     IPb_v1 += numpy.einsum('iaj, iaj -> ', tmp1_bar_iajh_bb[ki,ka][:,:,:,noccb-1], h2mo_ovoo_bb[ki,ka][:,:,:,noccb-1]).real
                     IPb_v1 += numpy.einsum('iaj, iaj -> ', tmp1_bar_iajh_ab[ki,ka][:,:,:,noccb-1], h2mo_ovoo_ab[ki,ka][:,:,:,noccb-1]).real
                     IPb_v2 += numpy.einsum('iaj, iaj -> ', tmp1_bar_iajh_bb[ki,ka][:,:,:,noccb-2], h2mo_ovoo_bb[ki,ka][:,:,:,noccb-2]).real
                     IPb_v2 += numpy.einsum('iaj, iaj -> ', tmp1_bar_iajh_ab[ki,ka][:,:,:,noccb-2], h2mo_ovoo_ab[ki,ka][:,:,:,noccb-2]).real
                     IPb_v3 += numpy.einsum('iaj, iaj -> ', tmp1_bar_iajh_bb[ki,ka][:,:,:,noccb-3], h2mo_ovoo_bb[ki,ka][:,:,:,noccb-3]).real
                     IPb_v3 += numpy.einsum('iaj, iaj -> ', tmp1_bar_iajh_ab[ki,ka][:,:,:,noccb-3], h2mo_ovoo_ab[ki,ka][:,:,:,noccb-3]).real

                 if kj==nk:
                     # --- Tính EA cho Alpha ---
                     w_ialb_aa = h2mo_ovvv_aa[ki,ka] - h2mo_ovvv_aa[ki,kb].transpose(0,3,2,1)
                     tmp1_bar_ialb_aa  = (w_ialb_aa / e_ialb_a).conj()
                     tmp1_bar_ialb_ba  = (h2mo_ovvv_ba[ki,ka] / e_ialb_ba).conj()

                     EAa_c1 -= numpy.einsum('iab, iab -> ', tmp1_bar_ialb_aa[:,:,0,:], h2mo_ovvv_aa[ki,ka][:,:,0,:]).real
                     EAa_c1 -= numpy.einsum('iab, iab -> ', tmp1_bar_ialb_ba[:,:,0,:], h2mo_ovvv_ba[ki,ka][:,:,0,:]).real
                     EAa_c2 -= numpy.einsum('iab, iab -> ', tmp1_bar_ialb_aa[:,:,1,:], h2mo_ovvv_aa[ki,ka][:,:,1,:]).real
                     EAa_c2 -= numpy.einsum('iab, iab -> ', tmp1_bar_ialb_ba[:,:,1,:], h2mo_ovvv_ba[ki,ka][:,:,1,:]).real
                     EAa_c3 -= numpy.einsum('iab, iab -> ', tmp1_bar_ialb_aa[:,:,2,:], h2mo_ovvv_aa[ki,ka][:,:,2,:]).real
                     EAa_c3 -= numpy.einsum('iab, iab -> ', tmp1_bar_ialb_ba[:,:,2,:], h2mo_ovvv_ba[ki,ka][:,:,2,:]).real

                     # --- Tính EA cho Beta ---
                     w_ialb_bb = h2mo_ovvv_bb[ki,ka] - h2mo_ovvv_bb[ki,kb].transpose(0,3,2,1)
                     tmp1_bar_ialb_bb  = (w_ialb_bb / e_ialb_b).conj()
                     tmp1_bar_ialb_ab  = (h2mo_ovvv_ab[ki,ka] / e_ialb_ab).conj()

                     EAb_c1 -= numpy.einsum('iab, iab -> ', tmp1_bar_ialb_bb[:,:,0,:], h2mo_ovvv_bb[ki,ka][:,:,0,:]).real
                     EAb_c1 -= numpy.einsum('iab, iab -> ', tmp1_bar_ialb_ab[:,:,0,:], h2mo_ovvv_ab[ki,ka][:,:,0,:]).real
                     EAb_c2 -= numpy.einsum('iab, iab -> ', tmp1_bar_ialb_bb[:,:,1,:], h2mo_ovvv_bb[ki,ka][:,:,1,:]).real
                     EAb_c2 -= numpy.einsum('iab, iab -> ', tmp1_bar_ialb_ab[:,:,1,:], h2mo_ovvv_ab[ki,ka][:,:,1,:]).real
                     EAb_c3 -= numpy.einsum('iab, iab -> ', tmp1_bar_ialb_bb[:,:,2,:], h2mo_ovvv_bb[ki,ka][:,:,2,:]).real
                     EAb_c3 -= numpy.einsum('iab, iab -> ', tmp1_bar_ialb_ab[:,:,2,:], h2mo_ovvv_ab[ki,ka][:,:,2,:]).real

    IPa = [IPa_v1, IPa_v2, IPa_v3]
    EAa = [EAa_c1, EAa_c2, EAa_c3]
    IPb = [IPb_v1, IPb_v2, IPb_v3]
    EAb = [EAb_c1, EAb_c2, EAb_c3]

    return IPa, EAa, IPb, EAb


"""
def make_rdm1(mp): # , t2=None, eris=None, verbose=logger.NOTE, ao_repr=False):
    '''Spin-traced one-particle density matrix.
    The occupied-virtual orbital response is not included.

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
    nmoa, nmob = mp.get_nmo()
    nocca, noccb = mp.get_nocc()
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    nkpts = len(mo_energy[0])

    # Check if these are padded mo coefficients and energies
    if mo_coeff[0][0].shape[1] != nmoa or mo_coeff[1][0].shape[1] != nmob:
         mo_coeff = padded_mo_coeff(mp, mo_coeff)

    if mo_energy[0][0].shape[0] != nmoa or mo_energy[1][0].shape[0] != nmob:
         mo_energy = padded_mo_energy(mp, mo_energy)
    
    if mo_occ[0][0].shape[0] != nmoa or mo_occ[1][0].shape[0] != nmob:
         mo_occ = padded_mo_occ(mp, mo_occ)

    return mo_coeff, mo_energy, mo_occ

def _padding_k_idx(nmo, nocc, kind="split"):
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
    if kind not in ("split", "joint"):
        raise ValueError("The 'kind' argument must be one of 'split', 'joint'")

    if kind == "split":
        indexes_o = []
        indexes_v = []
    else:
        indexes = []

    #nocca = numpy.array(nocca)
    #noccb = numpy.array(noccb)
    #nmoa = numpy.array(nmoa)
    #nmob = numpy.array(nmob)
    nocc = numpy.array(nocc)
    nmo = numpy.array(nmo)
    nvirt = nmo - nocc
    dense_o = numpy.amax(nocc)
    dense_v = numpy.amax(nvirt)
    dense_nmo = dense_o + dense_v

    for k_o, k_nmo in zip(nocc, nmo):
        k_v = k_nmo - k_o
        if kind == "split":
            indexes_o.append(numpy.arange(k_o))
            indexes_v.append(numpy.arange(dense_v - k_v, dense_v))
        else:
            indexes.append(numpy.concatenate((
                numpy.arange(k_o),
                numpy.arange(dense_nmo - k_v, dense_nmo),
            )))

    if kind == "split":
        return indexes_o, indexes_v

    else:
        return indexes


def padding_k_idx(mp, kind="split"):
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
    nocca, noccb = mp.get_nocc(per_kpoint=True)
    nmoa, nmob = mp.get_nmo(per_kpoint=True)
    padding_a = _padding_k_idx(nmoa, nocca, kind=kind)
    padding_b = _padding_k_idx(nmob, noccb, kind=kind)
    return padding_a, padding_b


def padded_mo_occ(mp, mo_occ):
    """
    Pads occupancy of active MOs.

    Returns:
        Padded molecular occupancy.
    """
    frozen_mask_a_list, frozen_mask_b_list = get_frozen_mask(mp)
    (padding_a_list, _), (padding_b_list, _) = padding_k_idx(mp, kind="joint")
    nkpts = len(padding_a_list)

    mo_occ_a, mo_occ_b = mo_occ

    nmoa, nmob = mp.get_nmo()

    result_a = numpy.zeros((nkpts, nmoa), dtype=mo_occ_a[0].dtype)
    result_b = numpy.zeros((nkpts, nmob), dtype=mo_occ_b[0].dtype)
    for k in range(nkpts):
        
        mask_a_k = frozen_mask_a_list[k]
        padding_a_k = padding_a_list[k]
        
        result_a[k, padding_a_k] = mo_occ_a[k][mask_a_k]
        
        
        mask_b_k = frozen_mask_b_list[k]
        padding_b_k = padding_b_list[k]
        result_b[k, padding_b_k] = mo_occ_b[k][mask_b_k]

    return (result_a, result_b)


def padded_mo_energy(mp, mo_energy):
    """
    Pads energies of active MOs.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        mo_energy (ndarray): original non-padded molecular energies;

    Returns:
        Padded molecular energies.
    """
    frozen_mask_a_list, frozen_mask_b_list = get_frozen_mask(mp)
    (padding_a_list, _), (padding_b_list, _) = padding_k_idx(mp, kind="joint")
    nkpts = len(padding_a_list)
    nmoa, nmob = mp.get_nmo() 
    mo_ea, mo_eb = mo_energy

    result_a = numpy.zeros((nkpts, nmoa), dtype=mo_ea[0].dtype)
    result_b = numpy.zeros((nkpts, nmob), dtype=mo_eb[0].dtype)
    for k in range(nkpts):
        mask_a_k = frozen_mask_a_list[k]
        padding_a_k = padding_a_list[k]
        result_a[k, padding_a_k] = mo_ea[k][mask_a_k]
        
        mask_b_k = frozen_mask_b_list[k]
        padding_b_k = padding_b_list[k]
        result_b[k, padding_b_k] = mo_eb[k][mask_b_k]

    return (result_a, result_b)

def padded_mo_coeff(mp, mo_coeff):
    """
    Pads coefficients of active MOs.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        mo_coeff (ndarray): original non-padded molecular coefficients;

    Returns:
        Padded molecular coefficients.
    """
    frozen_mask_a_list, frozen_mask_b_list = get_frozen_mask(mp)
    (padding_a_list, _), (padding_b_list, _) = padding_k_idx(mp, kind="joint")
    nkpts = len(padding_a_list)
    nmoa, nmob = mp.get_nmo()
    nao_a = mo_coeff[0][0].shape[0]
    nao_b = mo_coeff[1][0].shape[0]
    mo_coeff_a, mo_coeff_b = mo_coeff

    result_a = numpy.zeros((nkpts, nao_a, nmoa), dtype=mo_coeff_a[0].dtype)
    result_b = numpy.zeros((nkpts, nao_b, nmob), dtype=mo_coeff_b[0].dtype)
    for k in range(nkpts):
        mask_a_k = frozen_mask_a_list[k]
        padding_a_k = padding_a_list[k]
        result_a[k, :, padding_a_k] = mo_coeff_a[k][:, mask_a_k]
        
        mask_b_k = frozen_mask_b_list[k]
        padding_b_k = padding_b_list[k]
        result_b[k, :, padding_b_k] = mo_coeff_b[k][:, mask_b_k]

    return (result_a, result_b)


def _frozen_sanity_check(frozen, mo_occ, kpt_idx):
    '''Performs a few sanity checks on the frozen array and mo_occ.

    Specific tests include checking for duplicates within the frozen array.

    Args:
        frozen (array_like of int): The orbital indices that will be frozen.
        mo_occ (:obj:`ndarray` of int): The occupuation number for each orbital
            resulting from a mean-field-like calculation.
        kpt_idx (int): The k-point that `mo_occ` and `frozen` belong to.

    '''
    frozen = numpy.array(frozen)
    nocc = numpy.count_nonzero(mo_occ > 0)
    nvir = len(mo_occ) - nocc
    assert nocc, 'No occupied orbitals?\n\nnocc = %s\nmo_occ = %s' % (nocc, mo_occ)
    all_frozen_unique = (len(frozen) - len(numpy.unique(frozen))) == 0
    if not all_frozen_unique:
        raise RuntimeError('Frozen orbital list contains duplicates!\n\nkpt_idx %s\n'
                           'frozen %s' % (kpt_idx, frozen))
    if len(frozen) > 0 and numpy.max(frozen) > len(mo_occ) - 1:
        raise RuntimeError('Freezing orbital not in MO list!\n\nkpt_idx %s\n'
                           'frozen %s\nmax orbital idx %s' % (kpt_idx, frozen, len(mo_occ) - 1))


def get_nocc(mp, per_kpoint=False):
    '''Trả về số orbital bị chiếm (occupied) cho mỗi k-point (alpha và beta).'''
    if mp._nocc != None:
        return mp._nocc

    mo_occ_a = mp.mo_occ[0]
    mo_occ_b = mp.mo_occ[1]
    nkpts = len(mo_occ_a)
    frozen = mp.frozen

    nocca = []
    noccb = []

    if isinstance(frozen, (int, numpy.integer)):
        for ikpt in range(nkpts):
            nocca.append(numpy.count_nonzero(mo_occ_a[ikpt] > 0) - frozen)
            noccb.append(numpy.count_nonzero(mo_occ_b[ikpt] > 0) - frozen)
    elif isinstance(frozen, (list, numpy.ndarray)) and isinstance(frozen[0], (int, numpy.integer)):
        frozen_list = list(frozen)
        for ikpt in range(nkpts):
            occidxa = mo_occ_a[ikpt] > 0
            occidxa[frozen_list] = False
            nocca.append(numpy.count_nonzero(occidxa))
            
            occidxb = mo_occ_b[ikpt] > 0
            occidxb[frozen_list] = False
            noccb.append(numpy.count_nonzero(occidxb))
            
    elif frozen == None:
        for ikpt in range(nkpts):
            nocca.append(numpy.count_nonzero(mo_occ_a[ikpt] > 0))
            noccb.append(numpy.count_nonzero(mo_occ_b[ikpt] > 0))
    else:
        # 
        raise NotImplementedError

    if per_kpoint:
        return nocca, noccb
    else:
        return numpy.amax(nocca), numpy.amax(noccb)

def get_nmo(mp, per_kpoint=False):
    '''Trả về số orbital (đã pad) (alpha và beta).'''
    if mp._nmo != None:
        return mp._nmo

    mo_occ_a = mp.mo_occ[0]
    mo_occ_b = mp.mo_occ[1]
    nkpts = len(mo_occ_a)
    frozen = mp.frozen

    nmoa = []
    nmob = []

    if isinstance(frozen, (int, numpy.integer)):
        for ikpt in range(nkpts):
            nmoa.append(len(mo_occ_a[ikpt]) - frozen)
            nmob.append(len(mo_occ_b[ikpt]) - frozen)
            
    elif isinstance(frozen, (list, numpy.ndarray)) and isinstance(frozen[0], (int, numpy.integer)):
        frozen_list = list(frozen)
        for ikpt in range(nkpts):
            nmoa.append(len(mo_occ_a[ikpt]) - len(frozen_list))
            nmob.append(len(mo_occ_b[ikpt]) - len(frozen_list))
            
    elif frozen == None:
        for ikpt in range(nkpts):
            nmoa.append(len(mo_occ_a[ikpt]))
            nmob.append(len(mo_occ_b[ikpt]))
    else:
        raise NotImplementedError

    if per_kpoint:
        return nmoa, nmob
    else:
        nocca, noccb = mp.get_nocc(per_kpoint=True)
        
        nvira = numpy.array(nmoa) - numpy.array(nocca)
        nvirb = numpy.array(nmob) - numpy.array(noccb)
        
        nmoa_padded = numpy.amax(nocca) + numpy.amax(nvira)
        nmob_padded = numpy.amax(noccb) + numpy.amax(nvirb)
        
        return nmoa_padded, nmob_padded


def get_frozen_mask(mp):
    '''Get boolean mask for the unrestricted reference orbitals.

    In the returned boolean (mask) array of frozen orbital indices, the
    element is False if it corresonds to the frozen orbital.
    '''
    mo_occ_a = mp.mo_occ[0]
    mo_occ_b = mp.mo_occ[1]
    nkpts = len(mo_occ_a)
    frozen = mp.frozen
    
    moidxa = []
    moidxb = []
    
    if frozen == None:
        for ikpt in range(nkpts):
            moidxa.append(numpy.ones(mo_occ_a[ikpt].size, dtype=bool))
            moidxb.append(numpy.ones(mo_occ_b[ikpt].size, dtype=bool))
            
    elif isinstance(frozen, (int, numpy.integer)):
        for ikpt in range(nkpts):
            mask_a_k = numpy.ones(mo_occ_a[ikpt].size, dtype=bool)
            mask_a_k[:frozen] = False
            moidxa.append(mask_a_k)
            
            mask_b_k = numpy.ones(mo_occ_b[ikpt].size, dtype=bool)
            mask_b_k[:frozen] = False
            moidxb.append(mask_b_k)
            
    elif isinstance(frozen, (list, numpy.ndarray)):
        frozen_list_a = []
        frozen_list_b = []
        
        if isinstance(frozen[0], (int, numpy.integer)):
            frozen_list_a = list(frozen)
            frozen_list_b = list(frozen)
        
        elif len(frozen) == 2 and isinstance(frozen[0], (list, numpy.ndarray)):
            frozen_list_a = list(frozen[0])
            frozen_list_b = list(frozen[1])
        else:
            raise NotImplementedError("'frozen' (list of lists) unknown. ")
                                      
        for ikpt in range(nkpts):
            mask_a_k = numpy.ones(mo_occ_a[ikpt].size, dtype=bool)
            mask_a_k[frozen_list_a] = False
            moidxa.append(mask_a_k)
            
            mask_b_k = numpy.ones(mo_occ_b[ikpt].size, dtype=bool)
            mask_b_k[frozen_list_b] = False
            moidxb.append(mask_b_k)
    
    else:
        raise NotImplementedError(f"Not support: {type(frozen)}")
        
    return moidxa, moidxb


class OBMP2(lib.StreamObject):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):

        if mo_coeff  == None: mo_coeff  = mf.mo_coeff
        if mo_occ    == None: mo_occ    = mf.mo_occ

        self.thresh = 1e-06
        self.shift = 0.0
        self.niter = 100
        self.cell = mf.cell
        self._scf = mf
        self.verbose = self.cell.verbose 
        self.stdout = self.cell.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen
        self.kpts = getattr(mf, 'kpts', None)
        if hasattr(mf, 'khelper'):
            self.khelper = mf.khelper
        self.mom = False
        self.occ_exc = [None, None]
        self.vir_exc = [None, None]

        self.second_order = True
        self.ampf = 0.5
        self._IPa = None 
        self._EAa = None
        self._IPb = None
        self._EAb = None
##################################################
# don't modify the following attributes, they are not input options
        self.kpts = mf.kpts
        self.nkpts = numpy.shape(mf.mo_energy)[0]
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts) 
        self.mo_coeff = mo_coeff   
        self.mo_occ = mo_occ
        self.mo_energy = mf.mo_energy
        self._nocc = None
        self._nmo = None
        self.e_corr = None
        self.t2 = None
        self.mo_ea = None
        self.mo_eb = None
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
    #int_transform = int_transform

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen != 0:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    @property
    def emp2(self):
        return self.e_corr

    @property
    def e_tot(self):
        return self.ene_tot #+ self._scf.e_tot


    def kernel(self, shift=0.0, mo_energy=None, mo_coeff=None, mo_occ=None, with_t2=WITH_T2,
               _kern=kernel):
        '''
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.     
        '''
        if mo_occ == None:
            mo_occ = self.mo_occ
        if mo_energy == None:
            mo_energy = self.mo_energy
        if mo_coeff == None:
            mo_coeff = self.mo_coeff
        if mo_energy == None or mo_coeff == None or mo_occ == None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff, mo_energy are not given.\n'
                     'You may need to call mf.kernel() to generate them.')

        mo_coeff, mo_energy, mo_occ = _add_padding(self, mo_coeff, mo_energy, mo_occ)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        #self.dump_flags()
        #_kern(self, mo_energy, mo_coeff, eris, with_t2, self.verbose)
        self.ene_tot, self.mo_energy_alpha, self.mo_energy_beta, self.IPa, self.EAa, self.IPb, self.EAb = _kern(self, mo_energy, mo_coeff, mo_occ, with_t2, self.verbose)
        self.mo_energy = self.mo_energy_alpha
        self.mo_ea = self.mo_energy_alpha
        self.mo_eb = self.mo_energy_beta
        self._finalize()
        return self.ene_tot, self.mo_energy_alpha, self.mo_energy_beta, self.IPa, self.EAa, self.IPb, self.EAb

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.note(self, 'E(%s) = %.15g',
                    self.__class__.__name__, self.e_tot)
        return self

    make_veff = make_veff
    #make_amp  = make_amp
    first_BCH = first_BCH
    #second_BCH = second_BCH
    #make_rdm1 = make_rdm1
    #make_rdm2 = make_rdm2

    #as_scanner = as_scanner

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.mp import dfmp2
        mymp = dfmp2.DFMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
        if with_df != None:
            mymp.with_df = with_df
        if mymp.with_df.auxbasis != auxbasis:
            mymp.with_df = copy.copy(mymp.with_df)
            mymp.with_df.auxbasis = auxbasis
        return mymp

    def nuc_grad_method(self):
        from pyscf.grad import mp2
        return mp2.Gradients(self)


del(WITH_T2)


