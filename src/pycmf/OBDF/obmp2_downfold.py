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
ROB-MP2
'''

import time
from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf import __config__
from pycmf.OBMP import obmp2_slow as obmp2, obmp2 as obmp2_faster, obmp2_cas as obmp2_active
from pycmf.OBMP import uobmp2_cas as uob_act
from pycmf.OBMP import obmp2_cas as ob_act
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor


WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=logger.NOTE):
    if mp.mo_energy is None or mp.mo_coeff is None:
        #if mo_energy is None or mo_coeff is None:
        #    raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
        #                       'You may need to call mf.kernel() to generate them.')
        #moidx = mp.get_frozen_mask()
        #mo_coeff = None
        #mo_energy = (mp.mo_energy[0][moidx[0]], mp.mo_energy[1][moidx[1]])
        mo_coeff_init  = mp._scf.mo_coeff
        mo_coeff       = mp._scf.mo_coeff
        mo_energy = mp._scf.mo_energy
    else:
        #print("we are here")
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        #assert(mp.frozen is 0 or mp.frozen is None)
        #print("we are here")
        mo_coeff_init  = mp.mo_coeff
        mo_coeff  = mp.mo_coeff
        mo_energy = mp.mo_energy
            
    if mp.mom:
        mo_coeff_init = mp.mom_reorder(mo_coeff_init)
        mo_coeff = mp.mom_reorder(mo_coeff)
        mo_energy = mp.mo_energy

    mo_occ    = mp._scf.mo_occ
            
    nocc = numpy.array(mp.get_nocc())
    nmo = numpy.array(mp.get_nmo())

    nact  = mp.nact     ## nact := array(nact_a, nact_b)
    nocc_act = mp.nocc_act
    
    # Core orbitals = occupied - occupied_active
    ncore = nocc - nocc_act
    # Virtual orbitals trong active space
    nvir_act = nact - nocc_act
    
    mo_e = mo_energy   # ch? m?t m?ng nãng lý?ng cho restricted
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]


    #print("mo_ea, mo_eb: ", mo_ea, mo_eb)
    #print("eia_a, eia_b: ", eia_a, eia_b)
    

    shift = mp.shift #khong dung
    eri_ao = mp.mol.intor('int2e_sph')

    ########### ############ #############
    ######## h1mo & HF eff potential
    # Tao h1 trong AO basis
    h1ao = mp._scf.get_hcore(mp.mol)
    # Transform sang MO basis
    h1mo = mp.mo_coeff.T @ h1ao @ mp.mo_coeff
    # Cut ra active space
    h1mo_act = h1mo[ncore : ncore+nact, ncore : ncore+nact]
    # Tính effective potential cua core (restricted version)
    veff_core = ob_act.make_veff_core(mp)   
    # thêm hieu ung core
    h1mo_act += veff_core
    # Tao h1mo_act_eff (giong h1mo_act trong RHF)
    h1mo_act_eff = h1mo_act.copy()


    #####################
    ### Hartree-Fock            
    # lay matran fock trong vung acive, tu ncore -> ncore+nact
    fock_hf = mp.fock_hf[ncore : ncore+nact , ncore : ncore+nact]
    
    veff, c0  = ob_act.make_veff(mp)
        
    fock = fock_hf.copy()

    ####################
    #### MP1 amplitude
    #if mp.second_order:
    if getattr(mp, "second_order", True):
        mp.ampf = 1.0 #2
    
    #chi can 1 block cho RHF
    #cut tmp1 theo index ncore -> nact
    tmp1_act = mp.tmp1[ncore:ncore+nocc_act,
                       :nvir_act,
                       ncore:ncore+nocc_act,
                       :nvir_act]
    
    tmp1_bar_act = mp.tmp1_bar[ncore:ncore+nocc_act,
                               :nvir_act,
                               ncore:ncore+nocc_act,
                               :nvir_act]
    
    # scale lai bang ampf
    tmp1_bar = mp.tmp1_bar * mp.ampf
    tmp1 = mp.tmp1
    

    #####################
    ### BCH 1st order
    if getattr(mp, "first_order", True):
        c0, c1 = ob_act.first_BCH(mp, fock_hf, tmp1 , tmp1_bar_act, c0) #that ra khong can tmp1!
        # symmetrize
        fock += 0.5 * (c1 + c1.T)
    
    #####################
    ### External BCH 1st order
    if getattr(mp, "first_order", True):
        c1_act = inter_first_BCH(mp, tmp1_bar)
        c1_ext = c1_act - c1
        # symmetrize
        h1mo_act_eff += 0.5 * (c1_ext + c1_ext.T)
        fock += 0.5 * (c1_ext + c1_ext.T)

    #####################
    ### BCH 2nd order
    if getattr(mp, "second_order", True):
        c0, c1 = ob_act.second_BCH(mp, fock_hf, tmp1_act, tmp1_bar_act, c0)
        fock += 0.5 * (c1 + c1.T)
   
    #####################
    ### External BCH 2nd order
    if getattr(mp, "second_order", True):
        c1_act = inter_second_BCH(mp, tmp1, tmp1_bar)
        c1_ext = c1_act - c1
        h1mo_act_eff += 0.5 * (c1_ext + c1_ext.T)
        fock += 0.5 * (c1_ext + c1_ext.T)    
    
    ########### ############ #############
    ######## h1mo & h2mo output
    cg = mo_coeff[:, ncore:ncore+nact]
     
    h2mo_act = ao2mo.general(mp._scf._eri, (cg, cg, cg, cg), compact=False)
    h2mo_act = h2mo_act.reshape(nact, nact, nact, nact)
    

    #sua
    ### Energy core + c0[core, ext]
    ##
    ene_hf_inact = ene_hf_core(mp)
        
    ene_ob_inact = ene_inact_1st(mp, tmp1_bar)
     
    #if mp.second_order:
    if getattr(mp, "second_order", True):
        ene_ob_inact += ene_inact_2nd(mp, tmp1, tmp1_bar)

    ene_act = mp.c0_tot    
    for i in range(nocc_act):
        ene_act += 2.0 * fock[i,i]
    if ncore != 0:
        for i in range(ncore):
            ene_act += 2.0 * mp.c1[i,i]
        
    c0_act = c0_act_1st_BCH(mp, tmp1_bar) #phai test thu cai nay

    #if mp.second_order:
    if getattr(mp, "second_order", True):
        c0_act = c0_act_2nd_BCH(mp, tmp1, tmp1_bar, c0_act)
    
    #test==============    
    ene_act_eff = c0_act    
    for i in range(nocc_act):
        ene_act_eff += 2.0 * fock[i,i]               
    #===================        
        
  
    ### c0[core, vir]
    ## + c0[occ, ext]
    c0_ext = mp.c0_tot -c0_act
    ene_inact =  c0_ext + ene_ob_inact + ene_hf_inact
    ene_inact_hf =  c0_ext + ene_hf_inact
    
    
    print('c0_ext = ',c0_ext)

    print('ene_inact = ',ene_inact)
    
    print('ene_ob_inact = ',ene_ob_inact)

    print('ene_hf_inact = ',ene_hf_inact)

    print(' energy active (+c0_ext)= %8.14f'%ene_act)

    #==========================================
    print(' ene_act_eff= %8.14f'%ene_act_eff)

    print(' energy total (ene_inact +ene_act_eff) = %8.14f'%(ene_inact +ene_act_eff))
    


    
    ss, s = mp._scf.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                 mo_coeff[1][:,mo_occ[1]>0]), mp._scf.get_ovlp())
    print('multiplicity <S^2> = %.8g' %ss, '2S+1 = %.8g' %s)
    
    return ene_act_eff, ene_inact, ene_ob_inact, ene_hf_inact, h1mo_act_eff, h2mo_act, tmp1_bar_act, tmp1, ene_inact_hf

def make_veff(mp):
    
    nmo = numpy.array(mp.get_nmo())
    nocc = numpy.array(mp.get_nocc())
    nvir = [nmo[0] -nocc[0],
            nmo[1] -nocc[1]]

    nact  = mp.nact    
    nocc_act = mp.nocc_act
    nvir_act = [nact[0] - nocc_act[0]\
                ,nact[1]- nocc_act[1]]
    ncore = [nocc[0] - nocc_act[0]\
            ,nocc[1] - nocc_act[1]]
    next = [nmo[0] -nocc[0] -nvir_act[0],
            nmo[1] -nocc[1] -nvir_act[1]]
    mo_coeff = mp.mo_coeff
    
    cO =[mo_coeff[0][:,:nocc[0]]\
        ,mo_coeff[1][:,:nocc[1]]]

    cg = [mo_coeff[0][:,ncore[0]:ncore[0]+nact[0]]\
        ,mo_coeff[1][:,ncore[1]:ncore[1]+nact[1]]]

    h2mo_ggOO =[0,0,0,0]
    ele = 0
    for sp1 in numpy.arange(2):
        for sp2 in numpy.arange(2):
            h2mo_ggOO[ele] = ao2mo.general(mp._scf._eri, 
                        (cg[sp1],cg[sp1],cO[sp2],cO[sp2]), compact=False)
            h2mo_ggOO[ele] = h2mo_ggOO[ele].reshape(
                nact[sp1],nact[sp1],nocc[sp2],nocc[sp2])
            ele += 1 
    h2mo_gOOg =[0,0,0,0]
    ele = 0
    for sp1 in numpy.arange(2):
        for sp2 in numpy.arange(2):
            h2mo_gOOg[ele] = ao2mo.general(mp._scf._eri, 
                        (cg[sp1],cO[sp1],cO[sp2],cg[sp2]), compact=False)
            h2mo_gOOg[ele] = h2mo_gOOg[ele].reshape(
                nact[sp1],nocc[sp1],nocc[sp2],nact[sp2])
            ele += 1 

    veff = [numpy.zeros((nact[0],nact[0]))\
            ,numpy.zeros((nact[1],nact[1]))]
    veff[0] += numpy.einsum('ijkk -> ij',h2mo_ggOO[0]) \
                    - numpy.einsum('ijjk -> ik',h2mo_gOOg[0]) \
                    + numpy.einsum('ijkk -> ij',h2mo_ggOO[1])
        
    veff[1] += numpy.einsum('ijkk -> ij',h2mo_ggOO[3]) \
                - numpy.einsum('ijjk -> ik',h2mo_gOOg[3]) \
                + numpy.einsum('ijkk -> ij',h2mo_ggOO[2])
    return veff


def ene_hf_core(mp):

    nmo = numpy.array(mp.get_nmo())
    nocc = numpy.array(mp.get_nocc())
    nvir = nmo -nocc

    nact  = mp.nact    
    nocc_act = mp.nocc_act
    nvir_act = nact - nocc_act

    ncore = nocc - nocc_act

    next = nmo - nocc - nvir_act

    mo_coeff  = mp.mo_coeff
    
    ene_out =mp._scf.energy_nuc()
    ene_out += 2.0*numpy.trace(mp.fock_hf[:ncore,:ncore])

    return ene_out

'''
def ene_inact_1st(mp, tmp1_bar):

    nmo = numpy.array(mp.get_nmo())
    nocc = numpy.array(mp.get_nocc())
    nvir = [nmo[0] -nocc[0],
            nmo[1] -nocc[1]]

    nact  = mp.nact    
    nocc_act = mp.nocc_act
    nvir_act = [nact[0] - nocc_act[0]\
                ,nact[1]- nocc_act[1]]
    ncore = [nocc[0] - nocc_act[0]\
            ,nocc[1] - nocc_act[1]]
    next = [nmo[0] -nocc[0] -nvir_act[0],
            nmo[1] -nocc[1] -nvir_act[1]]
    ninact = [nmo[0] - nact[0]\
                ,nmo[1]- nact[1]]
    mo_coeff  = mp.mo_coeff
    fock_hf = mp.fock_hf

    hcore = mp._scf.get_hcore()
    energy_core = mp._scf.energy_nuc()

    cc = [mo_coeff[0][:,:ncore[0]]
            ,mo_coeff[1][:,:ncore[1]]]
    cO =[mo_coeff[0][:,:nocc[0]]\
        ,mo_coeff[1][:,:nocc[1]]]

    co = [mo_coeff[0][:,ncore[0]:ncore[0]+nocc_act[0]]\
        ,mo_coeff[1][:,ncore[1]:ncore[1]+nocc_act[1]]]

    cV =[mo_coeff[0][:,nocc[0]:]\
        ,mo_coeff[1][:,nocc[1]:]]

    cv =[mo_coeff[0][:,nocc[0]:nocc[0]+nvir_act[0]]\
        ,mo_coeff[1][:,nocc[1]:nocc[1]+nvir_act[1]]]

    ce =[mo_coeff[0][:,ncore[0]+nact[0]:]\
        ,mo_coeff[1][:,ncore[1]+nact[1]:]]

    cg = [mo_coeff[0][:,ncore[0]:ncore[0]+nact[0]]\
        ,mo_coeff[1][:,ncore[1]:ncore[1]+nact[1]]]

    ci = [numpy.concatenate((cc[0],ce[0]), axis=1), 
            numpy.concatenate((cc[1],ce[1]), axis=1)]

    
    c1_a = numpy.zeros((ninact[0],ninact[0]), dtype=fock_hf[0].dtype)
    c1_b = numpy.zeros((ninact[1],ninact[1]), dtype=fock_hf[1].dtype)

    tmp1_bar_OVce = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_OVce[ele] =\
            tmp1_bar[ele][:,:,:ncore[sb],nvir_act[sb]:]
            ele += 1

    tmp1_bar_OVcV = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_OVcV[ele] =\
            tmp1_bar[ele][:,:,:ncore[sb],:]
            ele += 1

    tmp1_bar_OVOe = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_OVOe[ele] =\
            tmp1_bar[ele][:,:,:,nvir_act[sb]:]
            ele += 1

    h2mo_OViV =[0,0,0,0]
    ele = 0
    for sp1 in numpy.arange(2):
        for sp2 in numpy.arange(2):
            h2mo_OViV[ele] = ao2mo.general(mp._scf._eri,
                        (cO[sp1],cV[sp1],ci[sp2],cV[sp2]))
            h2mo_OViV[ele] = h2mo_OViV[ele].reshape(
                        nocc[sp1],nvir[sp1],ninact[sp2],nvir[sp2])
            ele += 1

    h2mo_OVOi =[0,0,0,0]
    ele = 0
    for sp1 in numpy.arange(2):
        for sp2 in numpy.arange(2):
            h2mo_OVOi[ele] = ao2mo.general(mp._scf._eri, 
                        (cO[sp1],cV[sp1],cO[sp2],ci[sp2]))
            h2mo_OVOi[ele] = h2mo_OVOi[ele].reshape(
                        nocc[sp1],nvir[sp1],nocc[sp2],ninact[sp2])
            ele += 1

    c1_a[:ncore[0],ncore[0]:] += \
        2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_OVce[0],
            fock_hf[0][:nocc[0],nocc[0]:])\
        +2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_OVce[2],
            fock_hf[1][:nocc[1],nocc[1]:])

    c1_b[:ncore[1],ncore[1]:] += \
        2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_OVce[1],
            fock_hf[0][:nocc[0],nocc[0]:])\
        +2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_OVce[3],
            fock_hf[1][:nocc[1],nocc[1]:])

    ####################### c1[p,j] #########################

    c1_a[:,:ncore[0]] += \
        2.*lib.einsum('iajb,iapb -> pj',
            tmp1_bar_OVcV[0],h2mo_OViV[0])\
        +2.*lib.einsum('iajb,iapb -> pj',
            tmp1_bar_OVcV[2],h2mo_OViV[2])

    c1_b[:,:ncore[1]] += \
        2.*lib.einsum('iajb,iapb -> pj',
            tmp1_bar_OVcV[1],h2mo_OViV[1])\
        +2.*lib.einsum('iajb,iapb -> pj',
            tmp1_bar_OVcV[3],h2mo_OViV[3])

    ####################### c1[p,B] #########################

    c1_a[:,ncore[0]:] -= \
        2.*lib.einsum('iajb,iajp -> pb',
            tmp1_bar_OVOe[0],h2mo_OVOi[0])\
        +2.*lib.einsum('iajb,iajp -> pb',
            tmp1_bar_OVOe[2],h2mo_OVOi[2])

    c1_b[:,ncore[1]:] -= \
        2.*lib.einsum('iajb,iajp -> pb',
            tmp1_bar_OVOe[1],h2mo_OVOi[1])\
        +2.*lib.einsum('iajb,iajp -> pb',
            tmp1_bar_OVOe[3],h2mo_OVOi[3])
    
    f_ob = [0.5 * (c1_a + c1_a.T), 0.5 * (c1_b + c1_b.T)]
    ene_inact_1st = numpy.trace(f_ob[0][:ncore[0], :ncore[0]])\
        + numpy.trace(f_ob[1][:ncore[1], :ncore[1]])

    return ene_inact_1st
'''

def ene_inact_1st(mp, tmp1_bar):
    """
    Restricted version of ene_inact_1st.
    Compute inactive-space first-order energy correction
    without spin separation.
    """

    nmo   = mp.get_nmo()
    nocc  = mp.get_nocc()
    nact  = mp.nact
    nocc_act = mp.nocc_act

    nvir  = nmo - nocc
    nvir_act = nact - nocc_act
    ncore = nocc - nocc_act
    ninact = nmo - nact   # number of inactive orbitals (core+external)

    #next = nmo - nocc - nvir_act

    mo_coeff = mp.mo_coeff
    fock_hf  = mp.fock_hf
    hcore    = mp._scf.get_hcore()
    energy_core = mp._scf.energy_nuc()

    # orbital blocks
    cc = mo_coeff[:, :ncore]                          # core
    cO = mo_coeff[:, :nocc]                           # occupied (core+act)
    co = mo_coeff[:, ncore:ncore+nocc_act]            # active occupied
    cV = mo_coeff[:, nocc:]                           # virtual (act+ext)
    cv = mo_coeff[:, nocc:nocc+nvir_act]              # active virtual
    ce = mo_coeff[:, ncore+nact:]                     # external (inactive)
    cg = mo_coeff[:, ncore:ncore+nact]                # active block
    ci = numpy.concatenate((cc, ce), axis=1)          # inactive block (core+ext)

    # allocate Fock-like correction in inactive space
    c1 = numpy.zeros((ninact, ninact), dtype=fock_hf.dtype)

    # slice tmp1_bar into relevant inactive parts
    tmp1_bar_OVce = tmp1_bar[:, :, :ncore, nvir_act:]
    tmp1_bar_OVcV = tmp1_bar[:, :, :ncore, :]
    tmp1_bar_OVOe = tmp1_bar[:, :, :, nvir_act:]

    # two-electron integrals involving inactive orbitals
    h2mo_OViV = ao2mo.general(mp._scf._eri, (cO, cV, ci, cV), compact=False)
    h2mo_OViV = h2mo_OViV.reshape(nocc, nvir, ninact, nvir)

    h2mo_OVOi = ao2mo.general(mp._scf._eri, (cO, cV, cO, ci), compact=False)
    h2mo_OVOi = h2mo_OVOi.reshape(nocc, nvir, nocc, ninact)

    # ============================
    # build c1
    # ============================

    # c1[:ncore, ncore:] (inactive core vs ext) 
    c1[:ncore, ncore:] += 4.0 * lib.einsum('ijkl,ij -> kl',
                                           tmp1_bar_OVce,
                                           fock_hf[:nocc, nocc:])

    # c1[:, :ncore] (inactive coupling with core)  
    c1[:, :ncore] += 4.0 * lib.einsum('iajb,iapb -> pj',
                                      tmp1_bar_OVcV,
                                      h2mo_OViV)

    # c1[:, ncore:] (inactive coupling with ext) 
    c1[:, ncore:] -= 4.0 * lib.einsum('iajb,iajp -> pb',
                                      tmp1_bar_OVOe,
                                      h2mo_OVOi)

    # symmetrize
    f_ob = 0.5 * (c1 + c1.T)

    # energy: trace over inactive-core block
    ene_inact_1st = 2.0*numpy.trace(f_ob[:ncore, :ncore])

    return ene_inact_1st



'''
def ene_inact_2nd(mp, tmp1, tmp1_bar):

    nmo = numpy.array(mp.get_nmo())
    nocc = numpy.array(mp.get_nocc())
    nvir = [nmo[0] -nocc[0],
            nmo[1] -nocc[1]]

    nact  = mp.nact    
    nocc_act = mp.nocc_act
    nvir_act = [nact[0] - nocc_act[0]\
                ,nact[1]- nocc_act[1]]
    ncore = [nocc[0] - nocc_act[0]\
            ,nocc[1] - nocc_act[1]]
    next = [nmo[0] -nocc[0] -nvir_act[0],
            nmo[1] -nocc[1] -nvir_act[1]]
    ninact = [nmo[0] - nact[0]\
                ,nmo[1]- nact[1]]
    mo_coeff  = mp.mo_coeff
    fock_hf = mp.fock_hf

    
    c1_a = numpy.zeros((ninact[0],ninact[0]), dtype=fock_hf[0].dtype)
    c1_b = numpy.zeros((ninact[1],ninact[1]), dtype=fock_hf[1].dtype)

    tmp1_bar_OVce = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_OVce[ele] =\
            tmp1_bar[ele][:,:,:ncore[sb],nvir_act[sb]:]
            ele += 1

    tmp1_bar_OVcV = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_OVcV[ele] =\
            tmp1_bar[ele][:,:,:ncore[sb],:]
            ele += 1

    tmp1_bar_cVOV = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_cVOV[ele] =\
            tmp1_bar[ele][:ncore[sa],:,:,:]
            ele += 1

    tmp1_bar_OVOe = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_OVOe[ele] =\
            tmp1_bar[ele][:,:,:,nvir_act[sb]:]
            ele += 1

    tmp1_bar_OeOV = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_OeOV[ele] =\
            tmp1_bar[ele][:,nvir_act[sa]:,:,:]
            ele += 1
    ##################### ################## ###########

    tmp1_OVcV = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_OVcV[ele] =\
            tmp1[ele][:,:,:ncore[sb],:]
            ele += 1

    tmp1_cVOV = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_cVOV[ele] =\
            tmp1[ele][:ncore[sa],:,:,:]
            ele += 1

    tmp1_OVOe = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_OVOe[ele] =\
            tmp1[ele][:,:,:,nvir_act[sb]:]
            ele += 1

    tmp1_OeOV = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_OeOV[ele] =\
            tmp1[ele][:,nvir_act[sa]:,:,:]
            ele += 1
    

    #[1]
    y1_a = lib.einsum('ij,ijkl -> kl', 
        fock_hf[0][:nocc[0],nocc[0]:], tmp1_bar[0])\
        + lib.einsum('ij,ijkl -> kl', 
        fock_hf[1][:nocc[1],nocc[1]:], tmp1_bar[2])    
    y1_b = lib.einsum('ij,ijkl -> kl', 
        fock_hf[1][:nocc[1],nocc[1]:], tmp1_bar[3])\
        + lib.einsum('ij,ijkl -> kl', 
        fock_hf[0][:nocc[0],nocc[0]:], tmp1_bar[1])

    c1_a[:ncore[0],ncore[0]:] += \
        lib.einsum('kl,klij -> ij', y1_a, tmp1_bar_OVce[0])\
        +lib.einsum('kl,klij -> ij', y1_b, tmp1_bar_OVce[2])
    c1_b[:ncore[1],ncore[1]:] += \
        lib.einsum('kl,klij -> ij', y1_a, tmp1_bar_OVce[1])\
        +lib.einsum('kl,klij -> ij', y1_b, tmp1_bar_OVce[3])
    
    
    #[2]
    
    y1_aa = lib.einsum('ac,kcjb -> kajb',
            fock_hf[0][nocc[0]:,nocc[0]:],tmp1_bar_OVcV[0])
    y1_ab = lib.einsum('ac,kcjb -> kajb',
            fock_hf[0][nocc[0]:,nocc[0]:],tmp1_bar_OVcV[1])
    y1_ba = lib.einsum('ac,kcjb -> kajb',
            fock_hf[1][nocc[1]:,nocc[1]:],tmp1_bar_OVcV[2])
    y1_bb = lib.einsum('ac,kcjb -> kajb',
            fock_hf[1][nocc[1]:,nocc[1]:],tmp1_bar_OVcV[3])
    c1_a[:ncore[0],:ncore[0]] += \
        lib.einsum('iajb,iakb -> jk', tmp1_OVcV[0], y1_aa)\
        + lib.einsum('iajb,iakb -> jk', tmp1_OVcV[2], y1_ba)
    c1_b[:ncore[1],:ncore[1]] += \
        lib.einsum('iajb,iakb -> jk', tmp1_OVcV[3], y1_bb)\
        + lib.einsum('iajb,iakb -> jk', tmp1_OVcV[1], y1_ab)

    
    
    #[3]

    y1_aa = lib.einsum('ac,kcjb -> kajb',
        fock_hf[0][nocc[0]:,nocc[0]:],tmp1_bar_cVOV[0])
    y1_ab = lib.einsum('ac,kcjb -> kajb',
        fock_hf[0][nocc[0]:,nocc[0]:],tmp1_bar_cVOV[1])
    y1_ba = lib.einsum('ac,kcjb -> kajb',
        fock_hf[1][nocc[1]:,nocc[1]:],tmp1_bar_cVOV[2])
    y1_bb = lib.einsum('ac,kcjb -> kajb',
        fock_hf[1][nocc[1]:,nocc[1]:],tmp1_bar_cVOV[3])
    
    c1_a[:ncore[0],:ncore[0]] += \
        lib.einsum('iajb,kajb -> ik', tmp1_cVOV[0], y1_aa)\
        +lib.einsum('iajb,kajb -> ik', tmp1_cVOV[1], y1_ab)
    c1_b[:ncore[1],:ncore[1]] += \
        lib.einsum('iajb,kajb -> ik', tmp1_cVOV[3], y1_bb)\
        +lib.einsum('iajb,kajb -> ik', tmp1_cVOV[2], y1_ba)
    
    #[4]
    
    y1_aa = lib.einsum('ik,kalb -> ialb',
        fock_hf[0][:nocc[0],:nocc[0]],tmp1_bar_OVcV[0])
    y1_ab = lib.einsum('ik,kalb -> ialb',
        fock_hf[0][:nocc[0],:nocc[0]],tmp1_bar_OVcV[1])
    y1_ba = lib.einsum('ik,kalb -> ialb',
        fock_hf[1][:nocc[1],:nocc[1]],tmp1_bar_OVcV[2])
    y1_bb = lib.einsum('ik,kalb -> ialb',
        fock_hf[1][:nocc[1],:nocc[1]],tmp1_bar_OVcV[3])
    
    c1_a[:ncore[0],:ncore[0]] -= \
        lib.einsum('iajb,ialb -> jl', tmp1_OVcV[0], y1_aa)\
        +lib.einsum('iajb,ialb -> jl', tmp1_OVcV[2], y1_ba)
    c1_b[:ncore[1],:ncore[1]] -= \
        lib.einsum('iajb,ialb -> jl', tmp1_OVcV[3], y1_bb)\
        +lib.einsum('iajb,ialb -> jl', tmp1_OVcV[1], y1_ab)
    
    #[5]
    y1_a  = lib.einsum('iajb,kajb -> ik', tmp1[0], tmp1_bar_cVOV[0])
    y1_a += lib.einsum('iajb,kajb -> ik', tmp1[1], tmp1_bar_cVOV[1])
    y1_b  = lib.einsum('iajb,kajb -> ik', tmp1[3], tmp1_bar_cVOV[3])
    y1_b += lib.einsum('iajb,kajb -> ik', tmp1[2], tmp1_bar_cVOV[2])

    fhf = [numpy.concatenate((fock_hf[0][:ncore[0],:],
            fock_hf[0][ncore[0] +nact[0]:,:]), axis=0), 
        numpy.concatenate((fock_hf[1][:ncore[1],:],
            fock_hf[1][ncore[1] +nact[1]:,:]), axis=0)]

    c1_a[:,:ncore[0]] -= lib.einsum('pi,ik -> pk', 
        fhf[0][:,:nocc[0]], y1_a)
    c1_b[:,:ncore[1]] -= lib.einsum('pi,ik -> pk', 
        fhf[1][:,:nocc[1]], y1_b)
    
    #[6]
    y1_aa = lib.einsum('ik,kajd -> iajd',
        fock_hf[0][:nocc[0],:nocc[0]],tmp1_bar_OVOe[0])
    y1_ab = lib.einsum('ik,kajd -> iajd',
        fock_hf[0][:nocc[0],:nocc[0]],tmp1_bar_OVOe[1])
    y1_ba = lib.einsum('ik,kajd -> iajd',
        fock_hf[1][:nocc[1],:nocc[1]],tmp1_bar_OVOe[2])
    y1_bb = lib.einsum('ik,kajd -> iajd',
        fock_hf[1][:nocc[1],:nocc[1]],tmp1_bar_OVOe[3])

    c1_a[ncore[0]:,ncore[0]:] += \
        lib.einsum('iajb,iajd -> bd', tmp1_OVOe[0], y1_aa)\
        + lib.einsum('iajb,iajd -> bd', tmp1_OVOe[2], y1_ba)
    c1_b[ncore[1]:,ncore[1]:] += \
        lib.einsum('iajb,iajd -> bd', tmp1_OVOe[3], y1_bb)\
        +lib.einsum('iajb,iajd -> bd', tmp1_OVOe[1], y1_ab)
    
    #[7]
    y1_aa = lib.einsum('ik,kcjd -> icjd',
        fock_hf[0][:nocc[0],:nocc[0]],tmp1_bar_OeOV[0])
    y1_ab = lib.einsum('ik,kcjd -> icjd',
        fock_hf[0][:nocc[0],:nocc[0]],tmp1_bar_OeOV[1])
    y1_ba = lib.einsum('ik,kcjd -> icjd',
        fock_hf[1][:nocc[1],:nocc[1]],tmp1_bar_OeOV[2])
    y1_bb = lib.einsum('ik,kcjd -> icjd',
        fock_hf[1][:nocc[1],:nocc[1]],tmp1_bar_OeOV[3])

    c1_a[ncore[0]:,ncore[0]:] += \
        lib.einsum('iajb,icjb -> ac', tmp1_OeOV[0], y1_aa)\
        +lib.einsum('iajb,icjb -> ac', tmp1_OeOV[1], y1_ab)
    c1_b[ncore[1]:,ncore[1]:] += \
        lib.einsum('iajb,icjb -> ac', tmp1_OeOV[3], y1_bb)\
        +lib.einsum('iajb,icjb -> ac', tmp1_OeOV[2], y1_ba)

    #[8]
    y1_aa = lib.einsum('ac,icjd -> iajd',
        fock_hf[0][nocc[0]:,nocc[0]:],tmp1_bar_OVOe[0])
    y1_ab = lib.einsum('ac,icjd -> iajd',
        fock_hf[0][nocc[0]:,nocc[0]:],tmp1_bar_OVOe[1])
    y1_ba = lib.einsum('ac,icjd -> iajd',
        fock_hf[1][nocc[1]:,nocc[1]:],tmp1_bar_OVOe[2])
    y1_bb = lib.einsum('ac,icjd -> iajd',
        fock_hf[1][nocc[1]:,nocc[1]:],tmp1_bar_OVOe[3])

    c1_a[ncore[0]:,ncore[0]:] -= \
        lib.einsum('iajb,iajd -> bd', tmp1_OVOe[0], y1_aa)\
        +lib.einsum('iajb,iajd -> bd', tmp1_OVOe[2], y1_ba)
    c1_b[ncore[1]:,ncore[1]:] -= \
        lib.einsum('iajb,iajd -> bd', tmp1_OVOe[3], y1_bb)\
        +lib.einsum('iajb,iajd -> bd', tmp1_OVOe[1], y1_ab)
    
    #[9]
    y1_a  = lib.einsum('iajb,icjb -> ac', tmp1[0], tmp1_bar_OeOV[0])
    y1_a += lib.einsum('iajb,icjb -> ac', tmp1[1], tmp1_bar_OeOV[1])
    y1_b  = lib.einsum('iajb,icjb -> ac', tmp1[3], tmp1_bar_OeOV[3])
    y1_b += lib.einsum('iajb,icjb -> ac', tmp1[2], tmp1_bar_OeOV[2])

    fhf = [numpy.concatenate((fock_hf[0][:ncore[0],:],
            fock_hf[0][ncore[0] +nact[0]:,:]), axis=0), 
        numpy.concatenate((fock_hf[1][:ncore[1],:],
            fock_hf[1][ncore[1] +nact[1]:,:]), axis=0)]
    
    c1_a[:,ncore[0]:] -= \
        lib.einsum('pa,ac -> pc', fhf[0][
                :,nocc[0]:], y1_a)
    c1_b[:,ncore[1]:] -= \
        lib.einsum('pa,ac -> pc', fhf[1][
            :,nocc[1]:], y1_b)

    f_ob = [0.5 * (c1_a + c1_a.T), 0.5 * (c1_b + c1_b.T)]
    ene_inact_2nd = numpy.trace(f_ob[0][:ncore[0], :ncore[0]])\
            + numpy.trace(f_ob[1][:ncore[1], :ncore[1]])

    return ene_inact_2nd
'''    

def ene_inact_2nd(mp, tmp1, tmp1_bar):

    nmo = mp.get_nmo()
    nocc = mp.get_nocc()
    nvir = nmo - nocc

    nact = mp.nact    
    nocc_act = mp.nocc_act
    nvir_act = nact - nocc_act
    ncore = nocc - nocc_act
    next = nmo - nocc - nvir_act
    ninact = nmo - nact

    mo_coeff = mp.mo_coeff
    fock_hf = mp.fock_hf

    c1 = numpy.zeros((ninact, ninact), dtype=fock_hf.dtype)

    tmp1_bar_OVce = tmp1_bar[:,:,:ncore,nvir_act:]
    tmp1_bar_OVcV = tmp1_bar[:,:,:ncore,:]
    tmp1_bar_cVOV = tmp1_bar[:ncore,:,:,:]
    tmp1_bar_OVOe = tmp1_bar[:,:,:,nvir_act:]
    tmp1_bar_OeOV = tmp1_bar[:,nvir_act:,:,:]

    tmp1_OVcV = tmp1[:,:,:ncore,:]
    tmp1_cVOV = tmp1[:ncore,:,:,:]
    tmp1_OVOe = tmp1[:,:,:,nvir_act:]
    tmp1_OeOV = tmp1[:,nvir_act:,:,:]

    #[1]
    y1 = lib.einsum('ij,ijkl -> kl', fock_hf[:nocc,nocc:], tmp1_bar)
    c1[:ncore,ncore:] += 2.0*lib.einsum('kl,klij -> ij', y1, tmp1_bar_OVce)

    #[2]
    y1 = lib.einsum('ac,kcjb -> kajb', fock_hf[nocc:,nocc:], tmp1_bar_OVcV)
    c1[:ncore,:ncore] += 2.0*lib.einsum('iajb,iakb -> jk', tmp1_OVcV, y1)

    #[3]
    y1 = lib.einsum('ac,kcjb -> kajb', fock_hf[nocc:,nocc:], tmp1_bar_cVOV)
    c1[:ncore,:ncore] += 2.0*lib.einsum('iajb,kajb -> ik', tmp1_cVOV, y1)

    #[4]
    y1 = lib.einsum('ik,kalb -> ialb', fock_hf[:nocc,:nocc], tmp1_bar_OVcV)
    c1[:ncore,:ncore] -= 2.0*lib.einsum('iajb,ialb -> jl', tmp1_OVcV, y1)

    #[5]
    y1 = 2.0*lib.einsum('iajb,kajb -> ik', tmp1, tmp1_bar_cVOV)
    fhf = numpy.concatenate((fock_hf[:ncore,:], fock_hf[ncore+nact:,:]), axis=0)
    c1[:,:ncore] -= lib.einsum('pi,ik -> pk', fhf[:,:nocc], y1)

    #[6]
    y1 = lib.einsum('ik,kajd -> iajd', fock_hf[:nocc,:nocc], tmp1_bar_OVOe)
    c1[ncore:,ncore:] += 2.0*lib.einsum('iajb,iajd -> bd', tmp1_OVOe, y1)

    #[7]
    y1 = lib.einsum('ik,kcjd -> icjd', fock_hf[:nocc,:nocc], tmp1_bar_OeOV)
    c1[ncore:,ncore:] += 2.0*lib.einsum('iajb,icjb -> ac', tmp1_OeOV, y1)

    #[8]
    y1 = lib.einsum('ac,icjd -> iajd', fock_hf[nocc:,nocc:], tmp1_bar_OVOe)
    c1[ncore:,ncore:] -= 2.0*lib.einsum('iajb,iajd -> bd', tmp1_OVOe, y1)

    #[9]
    y1 = 2.0*lib.einsum('iajb,icjb -> ac', tmp1, tmp1_bar_OeOV)
    fhf = numpy.concatenate((fock_hf[:ncore,:], fock_hf[ncore+nact:,:]), axis=0)
    c1[:,ncore:] -= lib.einsum('pa,ac -> pc', fhf[:,nocc:], y1)

    f_ob = 0.5 * (c1 + c1.T)
    ene_inact_2nd = 2.0*numpy.trace(f_ob[:ncore,:ncore])
    return ene_inact_2nd


'''
def c0_act_1st_BCH(mp,tmp1_bar):
    nmo = numpy.array(mp.get_nmo())
    nocc = numpy.array(mp.get_nocc())
    nvir = [nmo[0] -nocc[0],
            nmo[1] -nocc[1]]

    nact  = mp.nact    
    nocc_act = mp.nocc_act
    nvir_act = [nact[0] - nocc_act[0]\
                ,nact[1]- nocc_act[1]]
    ncore = [nocc[0] - nocc_act[0]\
            ,nocc[1] - nocc_act[1]]
    mo_coeff  = mp.mo_coeff
    

    co = [mo_coeff[0][:,ncore[0]:ncore[0]+nocc_act[0]]\
        ,mo_coeff[1][:,ncore[1]:ncore[1]+nocc_act[1]]]
    
    cv =[mo_coeff[0][:,nocc[0]:nocc[0]+nvir_act[0]]\
        ,mo_coeff[1][:,nocc[1]:nocc[1]+nvir_act[1]]]


    h2mo_oooo =[0,0,0,0]
    ele = 0
    for sp1 in numpy.arange(2):
        for sp2 in numpy.arange(2):
            h2mo_oooo[ele] = ao2mo.general(mp._scf._eri, 
                        (co[sp1],co[sp1],co[sp2],co[sp2]), compact=False)
            h2mo_oooo[ele] = h2mo_oooo[ele].reshape(
                    nocc_act[sp1],nocc_act[sp1],nocc_act[sp2],nocc_act[sp2])
            ele += 1
    
    h2mo_ovov =[0,0,0,0]
    ele = 0
    for sp1 in numpy.arange(2):
        for sp2 in numpy.arange(2):
            h2mo_ovov[ele] = ao2mo.general(mp._scf._eri, 
                        (co[sp1],cv[sp1],co[sp2],cv[sp2]))
            h2mo_ovov[ele] = h2mo_ovov[ele].reshape(
                nocc_act[sp1],nvir_act[sp1],nocc_act[sp2],nvir_act[sp2])
            ele += 1      

    c0_act = 0.
    for i in range(nocc_act[0]):
        for j in range(nocc_act[0]):
            c0_act -= h2mo_oooo[0][i,i,j,j]-h2mo_oooo[0][i,j,j,i]
        for j in range(nocc_act[1]):
            c0_act -= h2mo_oooo[1][i,i,j,j]
    for i in range(nocc_act[1]):
        for j in range(nocc_act[1]):
            c0_act -= h2mo_oooo[3][i,i,j,j]-h2mo_oooo[3][i,j,j,i]
        for j in range(nocc_act[0]):
            c0_act -= h2mo_oooo[2][i,i,j,j]
    c0_act *= .5

    tmp1_bar_ovov =[0,0,0,0]
    ele =0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):    
            tmp1_bar_ovov[ele] = \
                tmp1_bar[ele][ncore[sa]:ncore[sa]+nocc_act[sa]
                        , :nvir_act[sa]
                        , ncore[sb]:ncore[sb]+nocc_act[sb]
                        , :nvir_act[sb]]
            ele += 1

    for ele in numpy.arange(4):
        c0_act -= 1.*numpy.sum(h2mo_ovov[ele]*tmp1_bar_ovov[ele])
    
    return c0_act
'''   

def c0_act_1st_BCH(mp, tmp1_bar):
    # L?y s? MO, s? occ, s? vir
    nmo = mp.get_nmo()
    nocc = mp.get_nocc()
    nvir = nmo - nocc

    # L?y s? active, occ active, vir active, core
    nact = mp.nact    
    nocc_act = mp.nocc_act
    nvir_act = nact - nocc_act
    ncore = nocc - nocc_act

    mo_coeff  = mp.mo_coeff

    # Orbital trong active: occ và vir
    co = mo_coeff[:, ncore:ncore+nocc_act]
    cv = mo_coeff[:, nocc:nocc+nvir_act]

    # ERI cho oooo (ch? trong active occ)
    h2mo_oooo = ao2mo.general(mp._scf._eri, (co, co, co, co), compact=False)
    h2mo_oooo = h2mo_oooo.reshape(nocc_act, nocc_act, nocc_act, nocc_act)

    # ERI cho ovov (occ-vir- occ-vir trong active)
    h2mo_ovov = ao2mo.general(mp._scf._eri, (co, cv, co, cv))
    h2mo_ovov = h2mo_ovov.reshape(nocc_act, nvir_act, nocc_act, nvir_act)

    # Tính c0_act t? ph?n oooo
    c0_act = 0.
    for i in range(nocc_act):
        for j in range(nocc_act):
            c0_act -= 2.0*h2mo_oooo[i,i,j,j] - h2mo_oooo[i,j,j,i]
    

    c0_act *= 1.0#0.5

    # L?y ph?n ovov c?a tmp1_bar
    tmp1_bar_ovov = tmp1_bar[ncore:ncore+nocc_act, :nvir_act,
                             ncore:ncore+nocc_act, :nvir_act]

    # Tr? ði ph?n ovov
    c0_act -= 4.0*numpy.sum(h2mo_ovov * tmp1_bar_ovov)

    return c0_act
  
  
    
'''
def inter_first_BCH(mp, tmp1_bar):
    nmo = numpy.array(mp.get_nmo())
    nocc = numpy.array(mp.get_nocc())
    nvir = [nmo[0] -nocc[0],
            nmo[1] -nocc[1]]

    nact  = mp.nact    
    nocc_act = mp.nocc_act
    nvir_act = [nact[0] - nocc_act[0]\
                ,nact[1]- nocc_act[1]]
    ncore = [nocc[0] - nocc_act[0]\
            ,nocc[1] - nocc_act[1]]
    next = [nmo[0] -nocc[0] -nvir_act[0],
            nmo[1] -nocc[1] -nvir_act[1]]

    fock_hf = mp.fock_hf
    mo_coeff  = mp.mo_coeff
    tmp1_bar_OVov = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_OVov[ele] =\
            tmp1_bar[ele][:,:,ncore[sb]:,:nvir_act[sb]]
            ele += 1

    tmp1_bar_OVoV = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_OVoV[ele] =\
            tmp1_bar[ele][:,:,ncore[sb]:,:]
            ele += 1
    tmp1_bar_OVOv = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_OVOv[ele] =\
            tmp1_bar[ele][:,:,:,:nvir_act[sb]]
            ele += 1

    cc =[mo_coeff[0][:,:ncore[0]]\
        ,mo_coeff[1][:,:ncore[1]]]

    cO =[mo_coeff[0][:,:nocc[0]]\
        ,mo_coeff[1][:,:nocc[1]]]

    co = [mo_coeff[0][:,ncore[0]:ncore[0]+nocc_act[0]]\
        ,mo_coeff[1][:,ncore[1]:ncore[1]+nocc_act[1]]]

    cV =[mo_coeff[0][:,nocc[0]:]\
        ,mo_coeff[1][:,nocc[1]:]]

    cv =[mo_coeff[0][:,nocc[0]:nocc[0]+nvir_act[0]]\
        ,mo_coeff[1][:,nocc[1]:nocc[1]+nvir_act[1]]]

    ce =[mo_coeff[0][:,nocc[0]+nvir_act[0]:]\
        ,mo_coeff[1][:,nocc[1]+nvir_act[1]:]]

    cg = [mo_coeff[0][:,ncore[0]:ncore[0]+nact[0]]\
        ,mo_coeff[1][:,ncore[1]:ncore[1]+nact[1]]]

    c1_a = numpy.zeros((nact[0],nact[0]), dtype=fock_hf[0].dtype)
    c1_b = numpy.zeros((nact[1],nact[1]), dtype=fock_hf[1].dtype)

    #######################  ################################
    h2mo_OVgV =[0,0,0,0]
    ele = 0
    for sp1 in numpy.arange(2):
        for sp2 in numpy.arange(2):
            h2mo_OVgV[ele] = ao2mo.general(mp._scf._eri,
                        (cO[sp1],cV[sp1],cg[sp2],cV[sp2]))
            h2mo_OVgV[ele] = h2mo_OVgV[ele].reshape(
                        nocc[sp1],nvir[sp1],nact[sp2],nvir[sp2])
            ele += 1

    h2mo_OVOg =[0,0,0,0]
    ele = 0
    for sp1 in numpy.arange(2):
        for sp2 in numpy.arange(2):
            h2mo_OVOg[ele] = ao2mo.general(mp._scf._eri, 
                        (cO[sp1],cV[sp1],cO[sp2],cg[sp2]))
            h2mo_OVOg[ele] = h2mo_OVOg[ele].reshape(
                        nocc[sp1],nvir[sp1],nocc[sp2],nact[sp2])
            ele += 1
    ####################### c1[j,B] #########################
    c1_a[:nocc_act[0],nocc_act[0]:] += \
        2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_OVov[0],
            fock_hf[0][:nocc[0],nocc[0]:])\
        +2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_OVov[2],
            fock_hf[1][:nocc[1],nocc[1]:])

    c1_b[:nocc_act[1],nocc_act[1]:] += \
        2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_OVov[1],
            fock_hf[0][:nocc[0],nocc[0]:])\
        +2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_OVov[3],
            fock_hf[1][:nocc[1],nocc[1]:])

    ####################### c1[p,j] #########################

    c1_a[:,:nocc_act[0]] += \
        2.*lib.einsum('iajb,iapb -> pj',
            tmp1_bar_OVoV[0],h2mo_OVgV[0])\
        +2.*lib.einsum('iajb,iapb -> pj',
            tmp1_bar_OVoV[2],h2mo_OVgV[2])

    c1_b[:,:nocc_act[1]] += \
        2.*lib.einsum('iajb,iapb -> pj',
            tmp1_bar_OVoV[1],h2mo_OVgV[1])\
        +2.*lib.einsum('iajb,iapb -> pj',
            tmp1_bar_OVoV[3],h2mo_OVgV[3])

    ####################### c1[p,B] #########################

    c1_a[:,nocc_act[0]:] -= \
        2.*lib.einsum('iajb,iajp -> pb',
            tmp1_bar_OVOv[0],h2mo_OVOg[0])\
        +2.*lib.einsum('iajb,iajp -> pb',
            tmp1_bar_OVOv[2],h2mo_OVOg[2])

    c1_b[:,nocc_act[1]:] -= \
        2.*lib.einsum('iajb,iajp -> pb',
            tmp1_bar_OVOv[1],h2mo_OVOg[1])\
        +2.*lib.einsum('iajb,iajp -> pb',
            tmp1_bar_OVOv[3],h2mo_OVOg[3])

    return c1_a, c1_b
'''

def inter_first_BCH(mp, tmp1_bar):
    """
    Restricted-version of inter_first_BCH.
    mp: HF object (RHF-like)
    tmp1_bar: 4-index array with shape (nocc, nvir, nocc, nvir) in MO-index ordering
              where 'occ' and 'vir' correspond to the RHF occupied/virtual partitions.
    Returns:
        c1 : numpy array (nact, nact) the first-order correction in the active space (spatial)
    """
    # --- basic numbers (RHF: scalars)
    nmo = int(mp.get_nmo())
    nocc = int(mp.get_nocc())            # number of occupied (spatial)
    nact = int(mp.nact)                  # number of active orbitals (spatial)
    nocc_act = int(mp.nocc_act)          # number of active occupied (spatial)
    nvir_act = nact - nocc_act
    nvir = nmo - nocc
    ncore = nocc - nocc_act

    # fock_hf is a single matrix in restricted case
    fock_hf = mp.fock_hf   # shape (nmo, nmo)
    mo_coeff = mp.mo_coeff  # shape (nao, nmo)

    # --- build AO->MO coefficient blocks (spatial)
    cO = mo_coeff[:, :nocc]                         # all occupied (spatial)
    cV = mo_coeff[:, nocc:]                         # all virtual  (spatial)
    co = mo_coeff[:, ncore : ncore + nocc_act]      # active occupied block
    cv = mo_coeff[:, nocc : nocc + nvir_act]        # active virtual  block
    cg = mo_coeff[:, ncore : ncore + nact]          # all active (occ+vir) block

    # --- slice tmp1_bar into useful sub-blocks (spatial)
    # Expect tmp1_bar shape (nocc, nvir, nocc, nvir)
    # OVov -> (occ, vir, occ_act, vir_act)
    tmp1_bar_OVov = tmp1_bar[:, : , ncore : ncore + nocc_act, : nvir_act]

    # OVoV -> (occ, vir, occ_act, nvir_total)
    tmp1_bar_OVoV = tmp1_bar[:, : , ncore : ncore + nocc_act, : ]

    # OVOv -> (occ, vir, occ_total, vir_act)
    tmp1_bar_OVOv = tmp1_bar[:, : , : , : nvir_act]

    # --- prepare result c1 (nact x nact)
    c1 = numpy.zeros((nact, nact), dtype=fock_hf.dtype)

    # --- build necessary two-electron integral blocks in MO basis
    # h2mo_OVgV: indices (i (occ), a (vir), g (act), b (vir))
    h2mo_OVgV = ao2mo.general(mp._scf._eri, (cO, cV, cg, cV))
    # reshape to (nocc, nvir, nact, nvir)
    h2mo_OVgV = h2mo_OVgV.reshape(nocc, nvir, nact, nvir)

    # h2mo_OVOg: indices (i (occ), a (vir), o (occ), g (act))
    h2mo_OVOg = ao2mo.general(mp._scf._eri, (cO, cV, cO, cg))
    # reshape to (nocc, nvir, nocc, nact)
    h2mo_OVOg = h2mo_OVOg.reshape(nocc, nvir, nocc, nact)

    # --- Now contract to build c1 following the UHF->RHF mapping.
    # In UHF code each spin contributed 2*..., and two spins sum -> overall factor 4
    # 1) c1[occupied_active, virtual_active]  (indexing within active: 0..nocc_act-1 and nocc_act..nact-1)
    fock_block = fock_hf[:nocc, nocc:]   # shape (nocc, nvir)
    c1[:nocc_act, nocc_act:] += 4.0 * lib.einsum('ijkl,ij->kl', tmp1_bar_OVov, fock_block)

    # 2) c1[:, :nocc_act]   (all p -> active occupied j)
    # einsum 'iajb,iapb -> pj' (i occ, a vir, j act-occ, b vir) contracted over i,a
    c1[:, :nocc_act] += 4.0 * lib.einsum('iajb,iapb->pj', tmp1_bar_OVoV, h2mo_OVgV)

    # 3) c1[:, nocc_act:]   (all p -> active virtual b)
    # einsum 'iajb,iajp -> pb' (i,a,j,p) contracted over i,a,j  -> pb where p indexes act, b indexes act-virtual
    c1[:, nocc_act:] -= 4.0 * lib.einsum('iajb,iajp->pb', tmp1_bar_OVOv, h2mo_OVOg)

    return c1


'''
def c0_act_2nd_BCH(mp, tmp1, tmp1_bar, c0_act):
    nocc = numpy.array(mp.get_nocc())
    nact  = mp.nact     ## nact := tuple(nact_a, nact_b)
    nocc_act = mp.nocc_act
    ncore = [nocc[0] - nocc_act[0]\
            ,nocc[1] - nocc_act[1]]
    nvir_act = numpy.array([nact[0] - nocc_act[0]\
                        ,nact[1]- nocc_act[1]])

    fock_hf = mp.fock_hf
    tmp1_bar_ovov = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_ovov[ele] =\
            tmp1_bar[ele][ncore[sa]:,:nvir_act[sa],ncore[sb]:,:nvir_act[sb]]
            ele += 1
    
    tmp1_ovov = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_ovov[ele] =\
            tmp1[ele][ncore[sa]:,:nvir_act[sa],ncore[sb]:,:nvir_act[sb]]
            ele += 1
    
    y1_aa = lib.einsum('ac,kcjb -> kajb',
            fock_hf[0][nocc[0]:nocc[0]+nvir_act[0],
                nocc[0]:nocc[0]+nvir_act[0]],tmp1_bar_ovov[0])
    y1_ab = lib.einsum('ac,kcjb -> kajb',
            fock_hf[0][nocc[0]:nocc[0]+nvir_act[0],
                nocc[0]:nocc[0]+nvir_act[0]],tmp1_bar_ovov[1])
    y1_ba = lib.einsum('ac,kcjb -> kajb',
            fock_hf[1][nocc[1]:nocc[1]+nvir_act[1],
                nocc[1]:nocc[1]+nvir_act[1]],tmp1_bar_ovov[2])
    y1_bb = lib.einsum('ac,kcjb -> kajb',
            fock_hf[1][nocc[1]:nocc[1]+nvir_act[1],
                nocc[1]:nocc[1]+nvir_act[1]],tmp1_bar_ovov[3])
    
    c0_act -= 1.*numpy.sum(tmp1_ovov[0] * y1_aa) + 1.*numpy.sum(tmp1_ovov[3] * y1_bb)
    c0_act -= 1.*numpy.sum(tmp1_ovov[1]* y1_ab) + 1.*numpy.sum(tmp1_ovov[2] * y1_ba)

    y1_aa = lib.einsum('ik,kalb -> ialb',
        fock_hf[0][ncore[0]:ncore[0]+nocc_act[0],
            ncore[0]:ncore[0]+nocc_act[0]],tmp1_bar_ovov[0])
    y1_ab = lib.einsum('ik,kalb -> ialb',
        fock_hf[0][ncore[0]:ncore[0]+nocc_act[0],
            ncore[0]:ncore[0]+nocc_act[0]],tmp1_bar_ovov[1])
    y1_ba = lib.einsum('ik,kalb -> ialb',
        fock_hf[1][ncore[1]:ncore[1]+nocc_act[1],
            ncore[1]:ncore[1]+nocc_act[1]],tmp1_bar_ovov[2])
    y1_bb = lib.einsum('ik,kalb -> ialb',
        fock_hf[1][ncore[1]:ncore[1]+nocc_act[1],
            ncore[1]:ncore[1]+nocc_act[1]],tmp1_bar_ovov[3])
    
    
    c0_act += 1.*numpy.sum(tmp1_ovov[0] * y1_aa) + 1.*numpy.sum(tmp1_ovov[3] * y1_bb)
    c0_act += 1.*numpy.sum(tmp1_ovov[1]* y1_ab) + 1.*numpy.sum(tmp1_ovov[2] * y1_ba)

    return c0_act
'''

def c0_act_2nd_BCH(mp, tmp1, tmp1_bar, c0_act):
    # L?y s? occ, s? active, s? occ_active, core, vir_active
    nocc = mp.get_nocc()
    nact = mp.nact
    nocc_act = mp.nocc_act
    ncore = nocc - nocc_act
    nvir_act = nact - nocc_act

    fock_hf = mp.fock_hf

    # Trích kh?i ovov trong active t? tmp1_bar và tmp1
    tmp1_bar_ovov = tmp1_bar[ncore:ncore+nocc_act, :nvir_act,
                             ncore:ncore+nocc_act, :nvir_act]
    tmp1_ovov     = tmp1[ncore:ncore+nocc_act, :nvir_act,
                         ncore:ncore+nocc_act, :nvir_act]

    # [1] ph?n liên quan ð?n fock trong không gian ?o (vir)
    y1 = lib.einsum('ac,kcjb -> kajb',
        fock_hf[nocc:nocc+nvir_act, nocc:nocc+nvir_act],
        tmp1_bar_ovov)

    c0_act -= 4.0*numpy.sum(tmp1_ovov * y1)

    # [2] ph?n liên quan ð?n fock trong không gian chi?m (occ)
    y1 = lib.einsum('ik,kalb -> ialb',
        fock_hf[ncore:ncore+nocc_act, ncore:ncore+nocc_act],
        tmp1_bar_ovov)

    c0_act += 4.0*numpy.sum(tmp1_ovov * y1)

    return c0_act



'''
def inter_second_BCH(mp, tmp1, tmp1_bar):
    nmo = numpy.array(mp.get_nmo())
    nocc = numpy.array(mp.get_nocc())
    nact  = mp.nact     ## nact := tuple(nact_a, nact_b)
    nocc_act = mp.nocc_act
    ncore = [nocc[0] - nocc_act[0]\
            ,nocc[1] - nocc_act[1]]
    nvir = [nmo[0] -nocc[0],
            nmo[1] -nocc[1]]
    nvir_act = numpy.array([nact[0] - nocc_act[0]\
                        ,nact[1]- nocc_act[1]])
    
    fock_hf = mp.fock_hf
    c1_a = numpy.zeros((nact[0],nact[0]), dtype=fock_hf[0].dtype)
    c1_b = numpy.zeros((nact[1],nact[1]), dtype=fock_hf[1].dtype)

    tmp1_bar_ovov = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_ovov[ele] =\
            tmp1_bar[ele][ncore[sa]:,:nvir_act[sa],ncore[sb]:,:nvir_act[sb]]
            ele += 1

    tmp1_bar_OVov = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_OVov[ele] =\
            tmp1_bar[ele][:,:,ncore[sb]:,:nvir_act[sb]]
            ele += 1
    
    tmp1_bar_OVoV = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_OVoV[ele] =\
            tmp1_bar[ele][:,:,ncore[sb]:,:]
            ele += 1

    tmp1_bar_oVOV = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_oVOV[ele] =\
            tmp1_bar[ele][ncore[sa]:,:,:,:]
            ele += 1
    
    tmp1_bar_OVOv = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_OVOv[ele] =\
            tmp1_bar[ele][:,:,:,:nvir_act[sb]]
            ele += 1

    tmp1_bar_OvOV = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_bar_OvOV[ele] =\
            tmp1_bar[ele][:,:nvir_act[sa],:,:]
            ele += 1
    ##################### ################## ###########

    tmp1_OVoV = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_OVoV[ele] =\
            tmp1[ele][:,:,ncore[sb]:,:]
            ele += 1

    tmp1_oVOV = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_oVOV[ele] =\
            tmp1[ele][ncore[sa]:,:,:,:]
            ele += 1

    tmp1_OVOv = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_OVOv[ele] =\
            tmp1[ele][:,:,:,:nvir_act[sb]]
            ele += 1

    tmp1_OvOV = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):
            tmp1_OvOV[ele] =\
            tmp1[ele][:,:nvir_act[sa],:,:]
            ele += 1
    
    #[1]
    y1_a = lib.einsum('ij,ijkl -> kl', 
        fock_hf[0][:nocc[0],nocc[0]:], tmp1_bar[0])\
        + lib.einsum('ij,ijkl -> kl', 
        fock_hf[1][:nocc[1],nocc[1]:], tmp1_bar[2])    
    y1_b = lib.einsum('ij,ijkl -> kl', 
        fock_hf[1][:nocc[1],nocc[1]:], tmp1_bar[3])\
        + lib.einsum('ij,ijkl -> kl', 
        fock_hf[0][:nocc[0],nocc[0]:], tmp1_bar[1])

    c1_a[:nocc_act[0],nocc_act[0]:] += \
        lib.einsum('kl,klij -> ij', y1_a, tmp1_bar_OVov[0])\
        +lib.einsum('kl,klij -> ij', y1_b, tmp1_bar_OVov[2])
    c1_b[:nocc_act[1],nocc_act[1]:] += \
        lib.einsum('kl,klij -> ij', y1_a, tmp1_bar_OVov[1])\
        +lib.einsum('kl,klij -> ij', y1_b, tmp1_bar_OVov[3])
    
    
    #[2]
    
    y1_aa = lib.einsum('ac,kcjb -> kajb',
            fock_hf[0][nocc[0]:,nocc[0]:],tmp1_bar_OVoV[0])
    y1_ab = lib.einsum('ac,kcjb -> kajb',
            fock_hf[0][nocc[0]:,nocc[0]:],tmp1_bar_OVoV[1])
    y1_ba = lib.einsum('ac,kcjb -> kajb',
            fock_hf[1][nocc[1]:,nocc[1]:],tmp1_bar_OVoV[2])
    y1_bb = lib.einsum('ac,kcjb -> kajb',
            fock_hf[1][nocc[1]:,nocc[1]:],tmp1_bar_OVoV[3])
    c1_a[:nocc_act[0],:nocc_act[0]] += \
        lib.einsum('iajb,iakb -> jk', tmp1_OVoV[0], y1_aa)\
        + lib.einsum('iajb,iakb -> jk', tmp1_OVoV[2], y1_ba)
    c1_b[:nocc_act[1],:nocc_act[1]] += \
        lib.einsum('iajb,iakb -> jk', tmp1_OVoV[3], y1_bb)\
        + lib.einsum('iajb,iakb -> jk', tmp1_OVoV[1], y1_ab)

    
    
    #[3]

    y1_aa = lib.einsum('ac,kcjb -> kajb',
        fock_hf[0][nocc[0]:,nocc[0]:],tmp1_bar_oVOV[0])
    y1_ab = lib.einsum('ac,kcjb -> kajb',
        fock_hf[0][nocc[0]:,nocc[0]:],tmp1_bar_oVOV[1])
    y1_ba = lib.einsum('ac,kcjb -> kajb',
        fock_hf[1][nocc[1]:,nocc[1]:],tmp1_bar_oVOV[2])
    y1_bb = lib.einsum('ac,kcjb -> kajb',
        fock_hf[1][nocc[1]:,nocc[1]:],tmp1_bar_oVOV[3])
    
    c1_a[:nocc_act[0],:nocc_act[0]] += \
        lib.einsum('iajb,kajb -> ik', tmp1_oVOV[0], y1_aa)\
        +lib.einsum('iajb,kajb -> ik', tmp1_oVOV[1], y1_ab)
    c1_b[:nocc_act[1],:nocc_act[1]] += \
        lib.einsum('iajb,kajb -> ik', tmp1_oVOV[3], y1_bb)\
        +lib.einsum('iajb,kajb -> ik', tmp1_oVOV[2], y1_ba)
    
    #[4]
    
    y1_aa = lib.einsum('ik,kalb -> ialb',
        fock_hf[0][:nocc[0],:nocc[0]],tmp1_bar_OVoV[0])
    y1_ab = lib.einsum('ik,kalb -> ialb',
        fock_hf[0][:nocc[0],:nocc[0]],tmp1_bar_OVoV[1])
    y1_ba = lib.einsum('ik,kalb -> ialb',
        fock_hf[1][:nocc[1],:nocc[1]],tmp1_bar_OVoV[2])
    y1_bb = lib.einsum('ik,kalb -> ialb',
        fock_hf[1][:nocc[1],:nocc[1]],tmp1_bar_OVoV[3])
    
    c1_a[:nocc_act[0],:nocc_act[0]] -= \
        lib.einsum('iajb,ialb -> jl', tmp1_OVoV[0], y1_aa)\
        +lib.einsum('iajb,ialb -> jl', tmp1_OVoV[2], y1_ba)
    c1_b[:nocc_act[1],:nocc_act[1]] -= \
        lib.einsum('iajb,ialb -> jl', tmp1_OVoV[3], y1_bb)\
        +lib.einsum('iajb,ialb -> jl', tmp1_OVoV[1], y1_ab)
    
    #[5]
    y1_a  = lib.einsum('iajb,kajb -> ik', tmp1[0], tmp1_bar_oVOV[0])
    y1_a += lib.einsum('iajb,kajb -> ik', tmp1[1], tmp1_bar_oVOV[1])
    y1_b  = lib.einsum('iajb,kajb -> ik', tmp1[3], tmp1_bar_oVOV[3])
    y1_b += lib.einsum('iajb,kajb -> ik', tmp1[2], tmp1_bar_oVOV[2])

    c1_a[:,:nocc_act[0]] -= lib.einsum('pi,ik -> pk', 
        fock_hf[0][ncore[0]:ncore[0]+nact[0],:nocc[0]], y1_a)
    c1_b[:,:nocc_act[1]] -= lib.einsum('pi,ik -> pk', 
        fock_hf[1][ncore[1]:ncore[1]+nact[1],:nocc[1]], y1_b)
    
    #[6]
    y1_aa = lib.einsum('ik,kajd -> iajd',
        fock_hf[0][:nocc[0],:nocc[0]],tmp1_bar_OVOv[0])
    y1_ab = lib.einsum('ik,kajd -> iajd',
        fock_hf[0][:nocc[0],:nocc[0]],tmp1_bar_OVOv[1])
    y1_ba = lib.einsum('ik,kajd -> iajd',
        fock_hf[1][:nocc[1],:nocc[1]],tmp1_bar_OVOv[2])
    y1_bb = lib.einsum('ik,kajd -> iajd',
        fock_hf[1][:nocc[1],:nocc[1]],tmp1_bar_OVOv[3])

    c1_a[nocc_act[0]:,nocc_act[0]:] += \
        lib.einsum('iajb,iajd -> bd', tmp1_OVOv[0], y1_aa)\
        + lib.einsum('iajb,iajd -> bd', tmp1_OVOv[2], y1_ba)
    c1_b[nocc_act[1]:,nocc_act[1]:] += \
        lib.einsum('iajb,iajd -> bd', tmp1_OVOv[3], y1_bb)\
        +lib.einsum('iajb,iajd -> bd', tmp1_OVOv[1], y1_ab)
    
    #[7]
    y1_aa = lib.einsum('ik,kcjd -> icjd',
        fock_hf[0][:nocc[0],:nocc[0]],tmp1_bar_OvOV[0])
    y1_ab = lib.einsum('ik,kcjd -> icjd',
        fock_hf[0][:nocc[0],:nocc[0]],tmp1_bar_OvOV[1])
    y1_ba = lib.einsum('ik,kcjd -> icjd',
        fock_hf[1][:nocc[1],:nocc[1]],tmp1_bar_OvOV[2])
    y1_bb = lib.einsum('ik,kcjd -> icjd',
        fock_hf[1][:nocc[1],:nocc[1]],tmp1_bar_OvOV[3])

    c1_a[nocc_act[0]:,nocc_act[0]:] += \
        lib.einsum('iajb,icjb -> ac', tmp1_OvOV[0], y1_aa)\
        +lib.einsum('iajb,icjb -> ac', tmp1_OvOV[1], y1_ab)
    c1_b[nocc_act[1]:,nocc_act[1]:] += \
        lib.einsum('iajb,icjb -> ac', tmp1_OvOV[3], y1_bb)\
        +lib.einsum('iajb,icjb -> ac', tmp1_OvOV[2], y1_ba)

    #[8]
    y1_aa = lib.einsum('ac,icjd -> iajd',
        fock_hf[0][nocc[0]:,nocc[0]:],tmp1_bar_OVOv[0])
    y1_ab = lib.einsum('ac,icjd -> iajd',
        fock_hf[0][nocc[0]:,nocc[0]:],tmp1_bar_OVOv[1])
    y1_ba = lib.einsum('ac,icjd -> iajd',
        fock_hf[1][nocc[1]:,nocc[1]:],tmp1_bar_OVOv[2])
    y1_bb = lib.einsum('ac,icjd -> iajd',
        fock_hf[1][nocc[1]:,nocc[1]:],tmp1_bar_OVOv[3])

    c1_a[nocc_act[0]:,nocc_act[0]:] -= \
        lib.einsum('iajb,iajd -> bd', tmp1_OVOv[0], y1_aa)\
        +lib.einsum('iajb,iajd -> bd', tmp1_OVOv[2], y1_ba)
    c1_b[nocc_act[1]:,nocc_act[1]:] -= \
        lib.einsum('iajb,iajd -> bd', tmp1_OVOv[3], y1_bb)\
        +lib.einsum('iajb,iajd -> bd', tmp1_OVOv[1], y1_ab)
    
    #[9]
    y1_a  = lib.einsum('iajb,icjb -> ac', tmp1[0], tmp1_bar_OvOV[0])
    y1_a += lib.einsum('iajb,icjb -> ac', tmp1[1], tmp1_bar_OvOV[1])
    y1_b  = lib.einsum('iajb,icjb -> ac', tmp1[3], tmp1_bar_OvOV[3])
    y1_b += lib.einsum('iajb,icjb -> ac', tmp1[2], tmp1_bar_OvOV[2])

    c1_a[:,nocc_act[0]:] -= \
        lib.einsum('pa,ac -> pc', fock_hf[0][
            ncore[0]:ncore[0]+nact[0],nocc[0]:], y1_a)
    c1_b[:,nocc_act[1]:] -= \
        lib.einsum('pa,ac -> pc', fock_hf[1][
            ncore[1]:ncore[1]+nact[1],nocc[1]:], y1_b)
    
    return c1_a, c1_b
'''



def inter_second_BCH(mp, tmp1, tmp1_bar):
    """
    Restricted version of inter_second_BCH.
    Compute the second-order BCH correction matrix in active space.

    Parameters
    ----------
    mp : object
        RHF-MP2 object containing Fock matrix, MO coefficients, etc.
    tmp1 : ndarray
        Intermediate tensor, shape (nocc, nvir, nocc, nvir).
    tmp1_bar : ndarray
        Symmetrized intermediate tensor, shape (nocc, nvir, nocc, nvir).

    Returns
    -------
    c1 : ndarray
        BCH correction matrix (nact, nact).
    """
    # --- orbital counts
    nmo   = mp.get_nmo()
    nocc  = mp.get_nocc()
    nact  = mp.nact
    nocc_act = mp.nocc_act
    ncore = nocc - nocc_act
    nvir  = nmo - nocc
    nvir_act = nact - nocc_act

    # --- Fock matrix (spatial, restricted)
    fock_hf = mp.fock_hf   # shape (nmo, nmo)

    # --- initialize result
    c1 = numpy.zeros((nact, nact), dtype=fock_hf.dtype)

    # --- slice useful tmp1_bar blocks
    tmp1_bar_OVov = tmp1_bar[:, :, ncore:ncore+nocc_act, :nvir_act]
    tmp1_bar_OVoV = tmp1_bar[:, :, ncore:ncore+nocc_act, :]
    tmp1_bar_oVOV = tmp1_bar[ncore:, :, :, :]
    tmp1_bar_OVOv = tmp1_bar[:, :, :, :nvir_act]
    tmp1_bar_OvOV = tmp1_bar[:, :nvir_act, :, :]

    # --- slice tmp1 blocks similarly
    tmp1_OVoV = tmp1[:, :, ncore:ncore+nocc_act, :]
    tmp1_oVOV = tmp1[ncore:, :, :, :]
    tmp1_OVOv = tmp1[:, :, :, :nvir_act]
    tmp1_OvOV = tmp1[:, :nvir_act, :, :]
        
    # =====================================================
    # [1] contractions with fock(i,a) and tmp1_bar_OVov
    # =====================================================
    y1 = 2.0 * lib.einsum('ij,ijkl -> kl', fock_hf[:nocc, nocc:], tmp1_bar)
    # factor 2 for spin, another 2 for symmetry ? 4 total
    c1[:nocc_act, nocc_act:] += 2.0 * lib.einsum('kl,klij -> ij', y1, tmp1_bar_OVov)

    # =====================================================
    # [2] contractions with fock(a,c) and OV-oV blocks
    # =====================================================
    y1 = lib.einsum('ac,kcjb -> kajb', fock_hf[nocc:, nocc:], tmp1_bar_OVoV)
    c1[:nocc_act, :nocc_act] += 2.0 * lib.einsum('iajb,iakb -> jk', tmp1_OVoV, y1)

    # =====================================================
    # [3] contractions with fock(a,c) and oVOV blocks
    # =====================================================
    y1 = lib.einsum('ac,kcjb -> kajb', fock_hf[nocc:, nocc:], tmp1_bar_oVOV)
    c1[:nocc_act, :nocc_act] += 2.0 * lib.einsum('iajb,kajb -> ik', tmp1_oVOV, y1)

    # =====================================================
    # [4] contractions with fock(i,k) and OV-oV
    # =====================================================
    y1 = lib.einsum('ik,kalb -> ialb', fock_hf[:nocc, :nocc], tmp1_bar_OVoV)
    c1[:nocc_act, :nocc_act] -= 2.0 * lib.einsum('iajb,ialb -> jl', tmp1_OVoV, y1)

    # =====================================================
    # [5] contractions mixing tmp1 and tmp1_bar
    # =====================================================
    y1 = lib.einsum('iajb,kajb->ik', tmp1, tmp1_bar_oVOV)
    c1[:, :nocc_act] -= 2.0 * lib.einsum('pi,ik->pk', fock_hf[ncore:ncore+nact, :nocc], y1)

    # =====================================================
    # [6] contractions with OV-Ov block
    # =====================================================
    y1 = lib.einsum('ik,kajd -> iajd', fock_hf[:nocc, :nocc], tmp1_bar_OVOv)
    c1[nocc_act:, nocc_act:] += 2.0 * lib.einsum('iajb,iajd -> bd', tmp1_OVOv, y1)

    # =====================================================
    # [7] contractions with OvOV block
    # =====================================================
    y1 = lib.einsum('ik,kcjd -> icjd', fock_hf[:nocc, :nocc], tmp1_bar_OvOV)
    c1[nocc_act:, nocc_act:] += 2.0 * lib.einsum('iajb,icjb -> ac', tmp1_OvOV, y1)

    # =====================================================
    # [8] contractions with fock(a,c) and OV-Ov
    # =====================================================
    y1 = lib.einsum('ac,icjd -> iajd', fock_hf[nocc:, nocc:], tmp1_bar_OVOv)
    c1[nocc_act:, nocc_act:] -= 2.0 * lib.einsum('iajb,iajd -> bd', tmp1_OVOv, y1)

    # =====================================================
    # [9] contractions mixing tmp1 and OvOV block
    # =====================================================
    y1 = lib.einsum('iajb,icjb->ac', tmp1, tmp1_bar_OvOV)
    c1[:, nocc_act:] -= 2.0 * lib.einsum('pa,ac->pc', fock_hf[ncore:ncore+nact, nocc:], y1)

    return c1




'''    
class UOBMP2(uob_act.UOBMP2):
    get_nocc = uob_act.get_nocc
    get_nmo = uob_act.get_nmo
    get_frozen_mask = uob_act.get_frozen_mask
    int_transform_ss = uob_act.int_transform_ss
    int_transform_os = uob_act.int_transform_os
    mom_select = uob_act.mom_select
    mom_reorder = uob_act.mom_reorder
    break_sym = False
    #use_t2 = False
    
    #sua
    @lib.with_doc(obmp2_active.OBMP2.kernel.__doc__)
    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, _kern=kernel):
        self.ene_act_eff, self.ene_inact, self.ene_ob_inact, self.ene_hf_inact, self.h1mo_act_eff\
            , self.h2mo_act, self.tmp1_bar_act, self.tmp1, self.ene_inact_hf\
             =_kern(self, mo_energy, mo_coeff, eris, with_t2)
                                     
        return self.ene_act_eff, self.ene_inact, self.ene_ob_inact, self.ene_hf_inact, self.h1mo_act_eff, self.h2mo_act\
            , self.tmp1_bar_act, self.tmp1, self.ene_inact_hf
        #kernel(self, mo_energy, mo_coeff, eris, with_t2, kernel)

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return _make_eris(self, mo_coeff, verbose=self.verbose)

    make_rdm1 = uob_act.make_rdm1
    sort_tmp1 = uob_act.sort_tmp1
    #make_rdm2 = make_rdm2
    make_fc = uob_act.make_fc
    eval_fc = False

    def nuc_grad_method(self):
        from pyscf.grad import ump2
        return ump2.Gradients(self)

'''

#sua
class OBMP2(ob_act.OBMP2):
    get_nocc = ob_act.get_nocc
    get_nmo = ob_act.get_nmo
    get_frozen_mask = ob_act.get_frozen_mask
    #int_transform_ss = ob_act.int_transform_ss
    #int_transform_os = ob_act.int_transform_os
    #mom_select = ob_act.mom_select
    #mom_reorder = ob_act.mom_reorder
    break_sym = False
    #use_t2 = False
    
    #sua
    @lib.with_doc(obmp2_active.OBMP2.kernel.__doc__)
    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, _kern=kernel):
        self.ene_act_eff, self.ene_inact, self.ene_ob_inact, self.ene_hf_inact, self.h1mo_act_eff\
            , self.h2mo_act, self.tmp1_bar_act, self.tmp1, self.ene_inact_hf\
             =_kern(self, mo_energy, mo_coeff, eris, with_t2)
                                     
        return self.ene_act_eff, self.ene_inact, self.ene_ob_inact, self.ene_hf_inact, self.h1mo_act_eff, self.h2mo_act\
            , self.tmp1_bar_act, self.tmp1, self.ene_inact_hf
        #kernel(self, mo_energy, mo_coeff, eris, with_t2, kernel)

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return _make_eris(self, mo_coeff, verbose=self.verbose)

    make_rdm1 = ob_act.make_rdm1
    #sort_tmp1 = ob_act.sort_tmp1
    #make_rdm2 = make_rdm2
    #make_fc = ob_act.make_fc
    eval_fc = False

    def nuc_grad_method(self):
        from pyscf.grad import ump2
        return ump2.Gradients(self)


OBMP2 = OBMP2

#from pyscf import scf
#scf.uhf.UHF.MP2 = lib.class_as_method(MP2)


class _ChemistsERIs(obmp2._ChemistsERIs):
    def __init__(self, mp, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        moidx = mp.get_frozen_mask()
        self.mo_coeff = mo_coeff = \
                (mo_coeff[0][:,moidx[0]], mo_coeff[1][:,moidx[1]])

def _make_eris(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    log = logger.new_logger(mp, verbose)
    time0 = (time.clock(), time.time())
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
    time0 = (time.clock(), time.time())
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
del(WITH_T2)


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
    mf = scf.RHF(mol).run()
    mp = OBMP2(mf, 2, 1)
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
