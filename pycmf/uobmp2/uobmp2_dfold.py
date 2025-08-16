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
UOB-MP2
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
from pyscf.mp import obmp2, obmp2_faster, obmp2_active
from pyscf.mp import uobmp2_active as uob_act
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
    #print("before")
    #print(mo_coeff[0])

    if mp.mom:
        mo_coeff_init = mp.mom_reorder(mo_coeff_init)
        mo_coeff = mp.mom_reorder(mo_coeff)
        mo_energy = mp.mo_energy

    #mo_coeff = mo_coeff_init

    #if eris is None: eris = mp.ao2mo(mo_coeff)

    #initializing w/ HF
    #mo_coeff  = mp._scf.mo_coeff
    #mo_energy = mp._scf.mo_energy
    mo_occ    = mp._scf.mo_occ

    nocca, noccb = mp.get_nocc()
    nocc = numpy.array(mp.get_nocc())
    nmo = numpy.array(mp.get_nmo())

    nact  = mp.nact     ## nact := array(nact_a, nact_b)
    nocc_act = mp.nocc_act
    ncore = [nocc[0] - nocc_act[0]\
            ,nocc[1] - nocc_act[1]]
    nvir_act = [nact[0] - nocc_act[0]\
                ,nact[1]- nocc_act[1]]

    mo_ea, mo_eb = mo_energy
    eia_a = mo_ea[:nocca,None] - mo_ea[None,nocca:]
    eia_b = mo_eb[:noccb,None] - mo_eb[None,noccb:]

    shift = mp.shift
    eri_ao = mp.mol.intor('int2e_sph')

    ########### ############ #############
    ######## h1mo & HF eff potential

    h1ao = mp._scf.get_hcore(mp.mol)
    h1mo_act = [0,0]
    for sp in numpy.arange(2):
        h1mo = numpy.matmul(mp.mo_coeff[sp].T,numpy.matmul(h1ao,mp.mo_coeff[sp]))
        h1mo_act[sp] = h1mo[ncore[sp] : ncore[sp]+nact[sp]\
                    , ncore[sp] : ncore[sp]+nact[sp]]
    veff_core = uob_act.make_veff_core(mp)
    h1mo_act[0] += veff_core[0]
    h1mo_act[1] += veff_core[1]

    h1mo_vqe = [0, 0]
    h1mo_vqe[0] += h1mo_act[0]
    h1mo_vqe[1] += h1mo_act[1]

    #####################
    ### Hartree-Fock
    #fock_hf = [0 ,0]
    #fock_hf[0] +=h1mo_act[0]
    #fock_hf[1] +=h1mo_act[1]

    fock_hf =[0,0]
    for sp in numpy.arange(2): 
        fock_hf[sp] = mp.fock_hf[sp][ncore[sp] : ncore[sp]+nact[sp]\
                    , ncore[sp] : ncore[sp]+nact[sp]]

    veff, c0  = uob_act.make_veff(mp)

    fock = [0, 0]
    fock[0] += fock_hf[0]
    fock[1] += fock_hf[1]
    c0 *= 0.5
    
    ####################
    #### MP1 amplitude
    """
    tmp1, tmp1_bar = make_amp(mp) 

    tmp1_bar_act = [0, 0, 0, 0]
    for ele in numpy.arange(4):
        tmp1_bar_act[ele] += tmp1_bar[ele]
    """
    if mp.second_order:
        mp.ampf = 1.0

    tmp1_act = [0,0,0,0]
    tmp1_bar_act = [0,0,0,0]
    ele = 0
    for sa in numpy.arange(2):
        for sb in numpy.arange(2):    
            tmp1_act[ele] = \
                mp.tmp1[ele][ncore[sa]:ncore[sa]+nocc_act[sa]
                        , :nvir_act[sa]
                        , ncore[sb]:ncore[sb]+nocc_act[sb]
                        , :nvir_act[sb]]
            tmp1_bar_act[ele] = \
                mp.tmp1_bar[ele][ncore[sa]:ncore[sa]+nocc_act[sa]
                        , :nvir_act[sa]
                        , ncore[sb]:ncore[sb]+nocc_act[sb]
                        , :nvir_act[sb]]
            ele += 1

    tmp1_bar = mp.tmp1_bar
    tmp1 = mp.tmp1
    for ele in numpy.arange(4):
        tmp1_bar[ele] *= mp.ampf

    #####################
    ### BCH 1st order  
    c0, c1_a, c1_b = uob_act.first_BCH(mp, fock_hf, tmp1_bar_act, c0)
    # symmetrize c1
    fock[0] += 0.5 * (c1_a + c1_a.T)
    fock[1] += 0.5 * (c1_b + c1_b.T)   

    #####################
    ### External BCH 1st order  
    #c1_a_ext, c1_b_ext = inter_first_BCH(mp, tmp1_bar)
    c1_a_act, c1_b_act = inter_first_BCH(mp, tmp1_bar)
    c1_a_ext = c1_a_act - c1_a
    c1_b_ext = c1_b_act - c1_b
    # symmetrize c1
    h1mo_vqe[0] += 0.5 * (c1_a_ext + c1_a_ext.T)
    h1mo_vqe[1] += 0.5 * (c1_b_ext + c1_b_ext.T)  

    fock[0] += 0.5 * (c1_a_ext + c1_a_ext.T)
    fock[1] += 0.5 * (c1_b_ext + c1_b_ext.T) 
    
    #####################
    ### BCH 2nd order 
    if mp.second_order:
        c0, c1_a, c1_b = uob_act.second_BCH(mp, fock, fock_hf, tmp1_act, tmp1_bar_act, c0)
    # symmetrize c1
        fock[0] += 0.5 * (c1_a + c1_a.T)
        fock[1] += 0.5 * (c1_b + c1_b.T) 
    
    
    #####################
    ### External BCH 2nd order  
    if mp.second_order:
        c1_a_act, c1_b_act = inter_second_BCH(mp, tmp1, tmp1_bar)
        c1_a_ext = c1_a_act -c1_a
        c1_b_ext = c1_b_act - c1_b 
    # symmetrize c1
        h1mo_vqe[0] += 0.5 * (c1_a_ext + c1_a_ext.T)
        h1mo_vqe[1] += 0.5 * (c1_b_ext + c1_b_ext.T) 

        fock[0] += 0.5 * (c1_a_ext + c1_a_ext.T)
        fock[1] += 0.5 * (c1_b_ext + c1_b_ext.T)  
    
    ########### ############ #############
    ######## h1mo & h2mo output

    cg = [mo_coeff[0][:,ncore[0]:ncore[0]+nact[0]]\
        ,mo_coeff[1][:,ncore[1]:ncore[1]+nact[1]]]
    
    ## h2mo_act := (aa, ab, ba, bb)
    h2mo_act = numpy.array(
                [numpy.zeros([nact[0], nact[0], nact[0], nact[0]])\
                ,numpy.zeros([nact[0], nact[0], nact[1], nact[1]])\
                ,numpy.zeros([nact[1], nact[1], nact[0], nact[0]])\
                ,numpy.zeros([nact[1], nact[1], nact[1], nact[1]])])
    ele = 0
    for s1 in numpy.arange(2):
        for s2 in numpy.arange(2):
            h2mo = ao2mo.general(mp._scf._eri, (cg[s1],cg[s1],cg[s2],cg[s2]), compact=False)
            h2mo_act[ele] = h2mo.reshape(nact[s1],nact[s1],nact[s2],nact[s2])
            ele +=1

    
    ### Energy core + c0[core, ext]
    ##
    ene_hf_inact = ene_hf_core(mp)
    ene_ob_inact = ene_inact_1st(mp, tmp1_bar)
    if mp.second_order:
        ene_ob_inact += ene_inact_2nd(mp, tmp1, tmp1_bar)

    ene_act = mp.c0_tot
    for i in range(nocc_act[0]):
        ene_act += 1. * fock[0][i,i]
    for i in range(nocc_act[1]):
        ene_act += 1. * fock[1][i,i]

    if ncore[0] != 0:
        for i in range(ncore[0]):
            ene_act += 1. * mp.c1[0][i,i]
        for i in range(ncore[1]):
            ene_act += 1. * mp.c1[1][i,i]
    
    c0_act = c0_act_1st_BCH(mp, tmp1_bar)
    if mp.second_order:
        c0_act = c0_act_2nd_BCH(mp, tmp1, tmp1_bar, c0_act)
        
    ### c0[core, vir]
    ## + c0[occ, ext]
    c0_ext = mp.c0_tot -c0_act
    ene_inact =  c0_ext + ene_ob_inact + ene_hf_inact
    print('c0_ext = ',c0_ext)
    print('ene_inact = ',ene_inact)

    print()
    print("========================")
    print(' energy active (eff)= %8.8f'%ene_act)
    print()

    print("========================")
    print(' energy active + outer = %8.8f'%(ene_act +ene_hf_inact))
    print()

    
    ss, s = mp._scf.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                 mo_coeff[1][:,mo_occ[1]>0]), mp._scf.get_ovlp())
    print('multiplicity <S^2> = %.8g' %ss, '2S+1 = %.8g' %s)
    
    return ene_act, ene_inact, ene_ob_inact, ene_hf_inact, h1mo_vqe, h2mo_act, tmp1_bar_act, tmp1

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
    mo_coeff  = mp.mo_coeff
    
    ene_out =mp._scf.energy_nuc()
    ene_out += numpy.trace(mp.fock_hf[0][:ncore[0],:ncore[0]])\
        +numpy.trace(mp.fock_hf[1][:ncore[1],:ncore[1]])

    return ene_out

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
    
    @lib.with_doc(obmp2_active.OBMP2.kernel.__doc__)
    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, _kern=kernel):
        self.ene_act, self.ene_inact, self.ene_ob_inact, self.ene_hf_inact, self.h1mo_vqe\
            , self.h2mo_act, self.tmp1_bar_act, self.tmp1\
             =_kern(self, mo_energy, mo_coeff, eris, with_t2)
                                     
        return self.ene_act, self.ene_inact, self.ene_ob_inact, self.ene_hf_inact, self.h1mo_vqe, self.h2mo_act\
            , self.tmp1_bar_act, self.tmp1
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


OBMP2 = UOBMP2

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
