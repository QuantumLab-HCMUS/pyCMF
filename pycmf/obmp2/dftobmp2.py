# !/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''
# OB-MP2
'''

import time
#from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.mp import mp2
import obmp2_faster
import dfobmp2_faster_ram
from pyscf import df
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf import __config__
from pyscf import scf, cc, dft
import json

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)

def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=logger.NOTE, alpha=(0.5, 0.7)):
    mol = mp.mol
    if mo_energy is None or mo_coeff is None:
        if mp.mo_energy is None or mp.mo_coeff is None:
            raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
                               'You may need to call mf.kernel() to generate them.')
        mo_coeff = None
        mo_energy = _mo_energy_without_core(mp, mp.mo_energy)
    else:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert(mp.frozen is 0 or mp.frozen is None)
    
    nuc = mp._scf.energy_nuc()
    ene_hf = mp._scf.energy_tot()

    nmo = mp.nmo
    nocc = mp.nocc
    nvir = mp.nmo - mp.nocc
    niter = mp.niter
    ene_old = 0.
    #eri_ao = mp.mol.intor('int2e_sph')

    S = mp._scf.get_ovlp()
    A = scipy.linalg.fractional_matrix_power(S, -0.5)
    #print('S shape =', S.shape) #(nao,nao)
    #print('A shape =', A.shape) #(nao,nao)

    F_list_a = []
    DIIS_RESID_a = []

    nmoa = mp.get_nmo()    
    nocca= mp.get_nocc()
    nvira = nmoa - nocca
    

    D_a = numpy.zeros((nmoa, nmoa))                              # Density in this iteration
    D_old_a = numpy.zeros((nmoa, nmoa)) + 1e-4   
   
    '''print("Number mo", nmo)
    print("Number occ", nocc)
    print("Number Vir", nvir)'''



    print()
    print("shift = ", mp.shift)
    print ("thresh = ", mp.thresh)
    print()
    
    
    # dft
    ks = dft.RKS(mol,f"{(alpha[0])}*HF+{(1-alpha[0])}*B88, {(1-alpha[1])}* LYP").density_fit()
    
    # density matrix
    dm = mp._scf.make_rdm1(mp.mo_coeff, mp.mo_occ)
    
    s1e = mp._scf.get_ovlp(mol)
    h1e = mp._scf.get_hcore(mol)
    vhf = mp._scf.get_veff(mol, dm)


    #s1e = mp._scf.get_ovlp(mol)
    
    
    dm_last = None
    fock_last = None
    vxc = ks.get_veff(mp._scf.mol, dm)
    
    print("\n \n \n Run double hybrid functional \n ")
  
    adiis = lib.diis.DIIS()

    for it in range(niter):
        print('alpha=',mp.alpha)

        #h2mo = [] #numpy.zeros((nmo,nmo,nmo,nmo)) #int_transform(eri_ao, mp.mo_coeff)
        #print(h1ao.shape)
        
        mp.mo_coeff, mp.mo_energy = coeff_active(mp)

        h1ao = mp._scf.get_hcore(mp.mol)
        h1mo = numpy.matmul(mp.mo_coeff.T,numpy.matmul(h1ao, mp.mo_coeff))
        #print('h1mo shape in iter = ', h1mo.shape) #(n_keep_vir, n_keep_vir)

        h1mo_vqe = 0
        h1mo_vqe += h1mo

        #for istep, qov in enumerate(mp.loop_ao2mo(mp.mo_coeff, mp.nocc)):
        #    buf = numpy.dot(qov.T,qov)

        #h2mo = buf  #ao2mo.general(mp._scf._eri, (co,cv,co,cv))
        #h2mo = buf.reshape(nocc,nvir,nocc,nvir)
        #####################
        ### Hartree-Fock

        fock_hf = h1mo
        veff, c0_hf = make_veff(mp)
        fock_hf += veff # (nmo, nmo)
        
        

        #initializing w/ HF
        fock = 0
        fock_obmp2 = 0
        
        fock_obmp2 += fock_hf
        c0 = c0_hf
        
        #dft
        hermi = 1
        vhfopt = None
        # vj, vk = mf.get_jk(mol, numpy.asarray(dm), hermi, vhfopt)
        vxc = ks.get_veff(mp._scf.mol, dm)#, dm_last, vxc)
        
        #print("vxc: ", numpy.matmul(mp.mo_coeff.T, numpy.matmul(vxc, mp.mo_coeff)))
        #print("fock_hf: ", fock_hf)
        
        #v_c_dft
        fock_dft = ks.get_fock(h1ao, s1e, vxc, dm, it)#, mf_diis, fock_last=fock_last)
        
        vhf = mp._scf.get_veff(mol, dm)
        fock_hf_pyscf = mp._scf.get_fock(h1ao, S, vhf, dm)
        #fock_last = fock_dft
        fock_dft =  numpy.matmul(mp.mo_coeff.T, numpy.matmul(fock_dft, mp.mo_coeff))
        fock_hf_pyscf =  numpy.matmul(mp.mo_coeff.T, numpy.matmul(fock_hf_pyscf, mp.mo_coeff))  
        
        print("Norm: ", numpy.linalg.norm(fock_hf - fock_hf_pyscf))
        
        
        
        # OBMP2
        if  mp.second_order:
            mp.ampf = 1.0
            
        #####################
        ### MP1 amplitude
        tmp1, tmp1_bar = make_amp(mp)
        
        #####################
        ### BCH 1st order  
        c0, c1 = first_BCH(mp, fock_hf, tmp1, tmp1_bar, c0)

        # symmetrize c1
        fock_obmp2 += 0.5 * (c1 + c1.T)

        
        #####################
        ### BCH 2nd order  
        if mp.second_order:

            c0, c1 = second_BCH(mp, fock_hf, tmp1, tmp1_bar, c0)
            # symmetrize c1
            fock_obmp2 += 0.5 * (c1 + c1.T)
        
        
        
        # Energy
        ene_hatree_fock = c0_hf
        for i in range(nocc):    
            ene_hatree_fock += 2. * fock_hf[i,i] 
        
       # print("Energy Difference", ene_hatree_fock - numpy.einsum('ij,ji->', h1ao, dm).real - numpy.einsum('ij,ji->', vhf, dm).real * .5)
        
        print("Norm: ", numpy.linalg.norm(fock_hf - fock_hf_pyscf))
        
        ene_hf_core = 0
        for i in range(nocc):    
            ene_hf_core += 2. * h1ao[i,i]
        

        # print('dm = ', dm)
        # print('h1ao =', h1ao)
        # print('vhf =', vhf)
        ene_dft =  ks.energy_elec(dm, h1ao, vxc)[0]
        # print('E_DFT = ', ene_dft )
        
        ene_obmp2 = c0
        for i in range(nocc):
            ene_obmp2 += 2. * fock_obmp2[i,i]
        
        
        # Total energy
        ene =  (ene_dft) + (ene_obmp2 - ene_hatree_fock) * alpha[1] 
        
        # Fock
        fock = (fock_dft) + (fock_obmp2 - fock_hf_pyscf) * alpha[1] 
        #print('fock shape =', fock.shape) #(nmo_active, nmo_active)now(69,69)
        
        ene_tot = ene + nuc
        
        de = abs(ene_tot - ene_old)
        ene_old = ene_tot
        print('iter = %d'%it, ' energy = %8.6f'%ene_tot, ' energy diff = %14.8f'%de, flush=True)

        # fock mo to Fock ao
        #print('S shape DIIS =', S.shape) #(110,110)
        #print('mo_coeff shape DIIS =', mp.mo_coeff.shape) #(110,69)
        #print('fock shape DIIS =', fock.shape) #(69,69)
        F_a = S@ mp.mo_coeff@ fock@ mp.mo_coeff.T@ S

        F_a = numpy.array(F_a, dtype=numpy.float64)
        S   = numpy.array(S, dtype=numpy.float64)
        #print('F_a shape =', F_a.shape) #(110,110)
        C_occa = mp.mo_coeff[:, :nocca]
        D_a = numpy.einsum('pi,qi->pq', C_occa, C_occa, optimize=True)
        
        err_a_ao = F_a @ D_a @ S - S @ D_a @ F_a   # AO residual (use this)
        diis_r_a = 0.5*(err_a_ao + err_a_ao.T)     # symmetrize residual for safety


        #err_a_ao = F_a.dot(D_a).dot(S) - S.dot(D_a).dot(F_a)
        err_a_mo = numpy.matmul(mp.mo_coeff.T,numpy.matmul(err_a_ao,mp.mo_coeff))
        #print('err_a_mo shape =', err_a_mo.shape) #(69,69)
        #Change A_ao to A_mo
        #print('A shape =', A.shape) #(110,110)
        A_mo = mp.mo_coeff.T @ A @ mp.mo_coeff

        
        # Build DIIS Residual
        #diis_r_a = A.dot(err_a_ao*100).dot(A)
        #diis_r_a = A_mo.dot(err_a_mo*100).dot(A_mo)
        diis_r_a = diis_r_a.real
        # Append trial & residual vectors to lists
        F_list_a.append(F_a)
        DIIS_RESID_a.append(diis_r_a) 
        
        max_diis = 40
        n_keep = 30
        if len(F_list_a) > max_diis:
            F_list_a = F_list_a[-n_keep:]
            DIIS_RESID_a = DIIS_RESID_a[-n_keep:]

        dRMS = numpy.mean(diis_r_a**2)**0.5
        
        if it >= 2:
        # Build B matrix
            B_dim_a = len(F_list_a) + 1
            B_a = numpy.empty((B_dim_a, B_dim_a))
            B_a[-1, :] = -1
            B_a[:, -1] = -1
            B_a[-1, -1] = 0
            for i in range(len(F_list_a)):
                for j in range(len(F_list_a)):
                    B_a[i, j] = numpy.einsum('ij,ij->', DIIS_RESID_a[i], DIIS_RESID_a[j], optimize=True)


            # Build RHS of Pulay equation 
            rhs_a = numpy.zeros((B_dim_a))
            rhs_a[-1] = -1
            
            # Solve Pulay equation for c_i's with NumPy
            coeff_a = numpy.linalg.solve(B_a, rhs_a)
            
            # Build DIIS Fock matrix
            F_a = numpy.zeros_like(F_a)
            for x in range(coeff_a.shape[0] - 1):
                F_a += coeff_a[x] * F_list_a[x]
            
        
        #print('mo_coeff shape before DIIS =', mp.mo_coeff) #(110,69)
        
        # Compute new orbital guess with DIIS Fock matrix
        mp.mo_energy, mp.mo_coeff = scipy.linalg.eigh(F_a, S)
        
        #print('mo_coeff shape after DIIS =', mp.mo_coeff) #(110,69)
        
        if de <= mp.thresh:
            break
            
        ks.mo_energy = mp. mo_energy
        ks.mo_coeff  = mp.mo_coeff
        
        #dm_last = dm
        
        dm = ks.make_rdm1(mp.mo_coeff, mp.mo_occ)
        # attach mo_coeff and mo_occ to dm to improve DFT get_veff efficiency
        dm = lib.tag_array(dm, mo_coeff=mp.mo_coeff, mo_occ=mp.mo_occ)


    return ene_tot - ene_hf, tmp1, h1mo_vqe, fock_hf

#################################################################################################################
def coeff_active(mp):
    mo_coeff_active = mp.mo_coeff[:, :mp.nocc + mp.n_keep_vir]
    mo_energy_active = mp.mo_energy[:mp.nocc + mp.n_keep_vir]
    return mo_coeff_active, mo_energy_active

def int_transform(eri_ao, mo_coeff):
    nao = mo_coeff.shape[0]
    nmo = mo_coeff.shape[1]
    eri_mo = numpy.dot(mo_coeff.T, eri_ao.reshape(nao,-1))
    eri_mo = numpy.dot(eri_mo.reshape(-1,nao), mo_coeff)
    eri_mo = eri_mo.reshape(nmo,nao,nao,nmo).transpose(1,0,3,2)
    eri_mo = numpy.dot(mo_coeff.T, eri_mo.reshape(nao,-1))
    eri_mo = numpy.dot(eri_mo.reshape(-1,nao), mo_coeff)
    eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
    return eri_mo

def make_veff(mp):
    nmo  = mp.nocc + mp.n_keep_vir
    nocc = mp.nocc
    #mo_coeff  = mp.mo_coeff
    naux = mp.with_df.get_naoaux()
    #print('nmo make_veff =', nmo)
    #print('nocc make_veff =', nocc)
    #print('nvir make_veff =', nmo - nocc)
    #print('naux make_veff =', naux)
    from pyscf.lib import current_memory
    import tracemalloc
    tracemalloc.start()

    for istep, qgg in enumerate(mp.loop_ao2mo_ggoo_cgcg(mp.mo_coeff, mp.nocc)):
        qgg = qgg.reshape(naux, nmo, nmo)
    
    print("qgg memory: %.1f MiB" % current_memory()[0])
    veff = numpy.zeros((nmo,nmo))
    veff = 2.*numpy.einsum("Lpq, Lii  -> pq", qgg,qgg[:,0:nocc, 0:nocc] )- numpy.einsum("Lpi, Liq -> pq", qgg[:, :, 0:nocc], qgg[:, 0:nocc,:])

    c0_hf = 0.
    c0_hf = -2.*numpy.einsum("Lpp, Lii -> ",qgg[:,0:nocc, 0:nocc], qgg[:,0:nocc, 0:nocc]) + numpy.einsum("Lij, Lji",qgg[:,0:nocc, 0:nocc], qgg[:,0:nocc, 0:nocc])
 
    print("veff memory: %.1f MiB" % current_memory()[0])
    return veff, c0_hf


def make_amp(mp):
    #nmo  = mp.nmo
    nmo = mp.nocc + mp.n_keep_vir
    nocc = mp.nocc
    #nvir = nmo - nocc
    nvir = mp.n_keep_vir
    mo_energy = mp.mo_energy
    mo_coeff  = mp.mo_coeff
    print('mo_coeff make_amp shape =', mo_coeff.shape)
    #print('nmo make_amp =', nmo)
    #print('nocc make_amp =', nocc)
    #print('nvir make_amp =', nvir)
    #co = numpy.asarray(mo_coeff[:,:nocc], order='F')
    #cv = numpy.asarray(mo_coeff[:,nocc:], order='F')
    from pyscf.lib import current_memory
    import tracemalloc
    tracemalloc.start()

    for istep, qov in enumerate(mp.loop_ao2mo(mp.mo_coeff, mp.nocc)):
        qov = qov
  
    h2mo = numpy.dot(qov.T,qov)  #ao2mo.general(mp._scf._eri, (co,cv,co,cv))
    h2mo = h2mo.reshape(nocc,nvir,nocc,nvir)

    print("ovov: %.1f MiB" % current_memory()[0])

    tmp1 = numpy.zeros((nocc,nvir,nocc,nvir))
    x = numpy.tile(mo_energy[:nocc,None] - mo_energy[None,nocc:],(nocc,nvir,1,1))
    x += numpy.einsum('ijkl -> klij', x) - mp.shift
    tmp1 = mp.ampf * h2mo/x
    
    tmp1_bar = numpy.zeros((nocc,nvir,nocc,nvir))
    tmp1_bar = tmp1 - 0.5*numpy.einsum('ijkl -> ilkj', tmp1)   

    print("tmp1: %.1f MiB" % current_memory()[0])
      
    return tmp1, tmp1_bar

def first_BCH(mp, fock_hf, tmp1, tmp1_bar, c0):
    #mo_coeff  = mp.mo_coeff
    #nmo  = mp.nmo
    nmo = mp.nocc + mp.n_keep_vir
    nocc = mp.nocc
    #nvir = mp.nmo - nocc
    nvir = mp.n_keep_vir
    naux = mp.with_df.get_naoaux()

    c1 = numpy.zeros((nmo,nmo), dtype=fock_hf.dtype)

    from pyscf.lib import current_memory
    import tracemalloc
    tracemalloc.start()

    for istep, qov in enumerate(mp.loop_ao2mo(mp.mo_coeff, mp.nocc)):
        qov = qov
        #print('qov shape in first_BCh =', qov.shape) #(212,909)
    
    for i in range(nocc):
        c0 -= 4.*numpy.einsum("ajb, ajb -> ", numpy.dot(qov[:,i*nvir:(i+1)*nvir].T, qov).reshape(nvir, nocc, nvir), tmp1_bar[i,:,:,:])
    print("c0 memory: %.1f MiB" % current_memory()[0])
##################################################################################
    for istep, qgv in enumerate(mp.loop_ao2mo_cgcv(mp.mo_coeff, mp.nocc)):
        qgv = qgv
    for i in range(nocc):
        c1[:,0:nocc] += 4. * numpy.einsum("apb, ajb -> pj",numpy.dot(qov[:,i*nvir:(i+1)*nvir].T,qgv).reshape(nvir,nmo,nvir),tmp1_bar[i,:,:,:])

    print("ovgv memory: %.1f MiB" % current_memory()[0])
    del(qgv)
####################################################################################
    for istep, qog in enumerate(mp.loop_ao2mo_goog_cocg(mp.mo_coeff, mp.nocc)):
        qog = qog

    for i in range(nocc):
        c1[:,nocc:nmo] -= 4.*numpy.einsum("ajp, ajb -> pb", numpy.dot(qov[:,i*nvir:(i+1)*nvir].T,qog).reshape(nvir,nocc, nmo),tmp1_bar[i,:,:,:])
    print("ovog memory: %.1f MiB" % current_memory()[0])
    #del(h2mo_ovog)
##################################################################################    
    c1_jb = 4.*numpy.einsum('ijkl -> ij',numpy.einsum('ijkl -> klij',tmp1_bar)\
            *numpy.tile(fock_hf[:nocc,nocc:],(nocc,nvir,1,1)))
    c1_jb = numpy.pad(c1_jb, [(0, nvir), (nocc, 0)], mode='constant')

    c1 += c1_jb
                    
    print("1st memory: %.1f MiB" % current_memory()[0])

    return c0, c1

def second_BCH(mp, fock_hf, tmp1, tmp1_bar, c0):
    #nmo  = mp.nmo
    nmo = mp.nocc + mp.n_keep_vir
    nocc = mp.nocc
    #nvir = mp.nmo - nocc
    nvir = mp.n_keep_vir

    c1 = numpy.zeros((nmo,nmo), dtype=fock_hf.dtype)
    #[1]
    y1 = numpy.zeros((nocc,nvir), dtype=fock_hf.dtype)
    y1 = numpy.einsum('ijkl -> kl', numpy.einsum('ijkl -> klij',\
        numpy.tile(fock_hf[:nocc,nocc:],(nocc,nvir,1,1))) * tmp1_bar)

    c1[:nocc,nocc:] += 4.*numpy.einsum('ijkl -> ij',\
        numpy.tile(y1,(nocc,nvir,1,1)) * tmp1_bar)

    #[2] [3] [8] [11]
    y1 = numpy.zeros((nocc,nvir,nocc,nvir), dtype=fock_hf.dtype)

    for c in range(nvir):
        y1 += numpy.einsum('ijkl -> klij',numpy.tile(fock_hf[nocc:,c-nvir].T,(nocc,nvir,nocc,1))) \
        *numpy.einsum('ijkl ->jikl',numpy.tile(tmp1_bar[:,c,:,:],(nvir,1,1,1)))

    for k in range(nocc):
        c1[:nocc,k] += 2.*(numpy.einsum('ijkl -> k',tmp1 \
                            * numpy.einsum('ijkl -> jkil',numpy.tile(y1[:,:,k,:],(nocc,1,1,1)))) \
                            + numpy.einsum('ijkl -> i',tmp1 * numpy.tile(y1[k,:,:,:],(nocc,1,1,1))))

    for b in range(nvir):    
        c1[b+nocc,nocc:] -= 2. * numpy.einsum('ijkl -> l',y1 * \
            numpy.einsum('ijkl -> jkli',numpy.tile(tmp1[:,:,:,b],(nvir,1,1,1))))
                
    c0 -= 4.*numpy.sum(tmp1 * y1)

    # [6] [7] [4] [10]
    y1 = numpy.zeros((nocc,nvir,nocc,nvir), dtype=fock_hf.dtype)

    for k in range(nocc):
        y1 += numpy.einsum('ijkl -> ljik',numpy.tile(fock_hf[:nocc,k],(nocc,nvir,nvir,1))) \
        * numpy.tile(tmp1_bar[k,:,:,:],(nocc,1,1,1))

    for c in range (nvir):
        c1[nocc:,c+nocc] += 2. * (numpy.einsum('ijkl -> l',tmp1 * \
            numpy.einsum('ijkl -> jkli',numpy.tile(y1[:,:,:,c],(nvir,1,1,1)))) \
             +  numpy.einsum('ijkl -> j',tmp1 * \
            numpy.einsum('ijkl -> jikl',numpy.tile(y1[:,c,:,:],(nvir,1,1,1)))))          

    for k in range(nocc):
        c1[:nocc,k] -= 2.*(numpy.einsum('ijkl -> k',tmp1 \
                        * numpy.einsum('ijkl -> jkil',numpy.tile(y1[:,:,k,:],(nocc,1,1,1)))))    

    c0 += 4.*numpy.sum(tmp1 * y1)    
    #[5]
    y1 = numpy.zeros((nocc,nocc), dtype=fock_hf.dtype)

    for k in range(nocc):
        y1[:,k] += numpy.einsum('ijkl -> i',tmp1 * numpy.tile(tmp1_bar[k,:,:,:],(nocc,1,1,1)))

    for k in range(nocc):
        c1[:,k] -= 2. * numpy.einsum('ij -> i', \
                    fock_hf[:nocc,:].T * numpy.tile(y1[:,k],(nmo,1)))

    #[9]
    y1 = numpy.zeros((nvir,nvir), dtype=fock_hf.dtype)

    for c in range(nvir):                
        y1[:,c] += numpy.einsum('ijkl -> j',tmp1 * \
            numpy.einsum('ijkl -> jikl',numpy.tile(tmp1_bar[:,c,:,:],(nvir,1,1,1))))     
                 
    for c in range(nvir):
        c1[:,c+nocc] -= 2. * numpy.einsum('ij -> i', \
                    fock_hf[nocc:,:].T * numpy.tile(y1[:,c],(nmo,1)))
    return c0, c1


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
    #nvir = mp.nmo - nocc
    nvir = mp.n_keep_vir
    eia = mp.mo_energy[:nocc,None] - mp.mo_energy[None,nocc:] 
    eris = mp.ao2mo(mp.mo_coeff)

    t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
    for i in range(nocc):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals wiht the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            gi = eris.ovov[i]
        else:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])

        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
        t2[i] = t2i

    doo, dvv = _gamma1_intermediates(mp, t2, eris)
    nocc = doo.shape[0]
    nvir = dvv.shape[0]
    dov = numpy.zeros((nocc,nvir), dtype=doo.dtype)
    dvo = dov.T
    return ccsd_rdm._make_rdm1(mp, (doo, dov, dvo, dvv), with_frozen=True,
                               ao_repr=False)

def _gamma1_intermediates(mp, t2=None, eris=None):
    if t2 is None: t2 = mp.t2
    #nmo = mp.nmo
    nmo = mp.nocc + mp.n_keep_vir
    nocc = mp.nocc
    #nvir = nmo - nocc
    nvir = mp.n_keep_vir
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



def get_nocc(mp):
    if mp._nocc is not None:
        return mp._nocc
    elif mp.frozen is None:
        nocc = numpy.count_nonzero(mp.mo_occ > 0)
        assert(nocc > 0)
        return nocc
    elif isinstance(mp.frozen, (int, numpy.integer)):
        nocc = numpy.count_nonzero(mp.mo_occ > 0) - mp.frozen
        assert(nocc > 0)
        return nocc
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        occ_idx = mp.mo_occ > 0
        occ_idx[list(mp.frozen)] = False
        nocc = numpy.count_nonzero(occ_idx)
        assert(nocc > 0)
        return nocc
    else:
        raise NotImplementedError

def get_nmo(mp):
    if mp._nmo is not None:
        return mp._nmo
    elif mp.frozen is None:
        return len(mp.mo_occ)
    elif isinstance(mp.frozen, (int, numpy.integer)):
        return len(mp.mo_occ) - mp.frozen
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        return len(mp.mo_occ) - len(set(mp.frozen))
    else:
        raise NotImplementedError

def get_frozen_mask(mp):
    '''Get boolean mask for the restricted reference orbitals.

    In the returned boolean (mask) array of frozen orbital indices, the
    element is False if it corresonds to the frozen orbital.
    '''
    moidx = numpy.ones(mp.mo_occ.size, dtype=numpy.bool_)
    if mp._nmo is not None:
        moidx[mp._nmo:] = False
    elif mp.frozen is None:
        pass
    elif isinstance(mp.frozen, (int, numpy.integer)):
        moidx[:mp.frozen] = False
    elif len(mp.frozen) > 0:
        moidx[list(mp.frozen)] = False
    else:
        raise NotImplementedError
    return moidx


class B2PLYPDFOBMP2(obmp2_faster.OBMP2):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):

        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.thresh = 1e-07
        self.shift = 0
        self.niter = 1000
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
        self.ampf = 0.5

        self.alpha = None
        self.n_keep_vir = None
##################################################
# don't modify the following attributes, they are not input options
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.e_corr = None
        self.t2 = None
        self._keys = set(self.__dict__.keys())

        mp2.MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
        self._keys.update(['with_df'])
    
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
    int_transform = int_transform

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen is not 0:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    @property
    def emp2(self):
        return self.e_corr

    @property
    def e_tot(self):
        return self.e_corr + self._scf.e_tot


    def kernel(self, shift=0.0, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
               _kern=kernel):
        '''
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.
        '''
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()
        
        
        if hasattr(self, 'alpha'):
            self.e_corr,self.tmp1, self.h1mo_vqe, self.fock_hf = _kern(self, mo_energy, mo_coeff,
                                     eris, with_t2, self.verbose, self.alpha)
        else:
            self.e_corr,self.tmp1, self.h1mo_vqe, self.fock_hf = _kern(self, mo_energy, mo_coeff,
                                     eris, with_t2, self.verbose)
        self._finalize()
        return self.e_corr,self.tmp1, self.h1mo_vqe, self.fock_hf

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.note(self, 'E(%s) = %.15g',
                    self.__class__.__name__, self.e_tot)
        return self

    def ao2mo(self, mo_coeff=None):
        return _make_eris(self, mo_coeff, verbose=self.verbose)

    make_veff = make_veff
    make_amp  = make_amp
    first_BCH = first_BCH
    second_BCH = second_BCH
    make_rdm1 = make_rdm1
    #make_rdm2 = make_rdm2

    #as_scanner = as_scanner

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
    
    def loop_ao2mo(self, mo_coeff, nocc):
        mo = numpy.asarray(mo_coeff, order='F')
        #nmo = mo.shape[0]
        nmo = self.nocc + self.n_keep_vir
        ijslice = (0, nocc, nocc, nmo)
        Lov = None
        with_df = self.with_df

        nvir = nmo - nocc
        naux = with_df.get_naoaux()
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        blksize = int(min(naux, max(with_df.blockdim,
                                    (max_memory*1e6/8-nocc*nvir**2*2)/(nocc*nvir))))
        for eri1 in with_df.loop(blksize=blksize):
            Lov = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', out=Lov)
            yield Lov

    def loop_ao2mo_goog_cocg(self, mo_coeff, nocc):
        mo = numpy.asarray(mo_coeff, order='F')
        #nmo = mo.shape[0]
        nmo = self.nocc + self.n_keep_vir
        ijslice = (0, nocc , 0, nmo)
        Lov = None
        with_df = self.with_df

        nvir = nmo - nocc
        naux = with_df.get_naoaux()
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        blksize = int(min(naux, max(with_df.blockdim,
                                    (max_memory*1e6/8-nocc*nvir**2*2)/(nocc*nvir))))
        for eri1 in with_df.loop(blksize=blksize):
            Lov = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', out=Lov)
            yield Lov


    def loop_ao2mo_goog_cgco(self, mo_coeff, nocc):
        mo = numpy.asarray(mo_coeff, order='F')
        nmo = mo.shape[0]
        ijslice = (0, nmo , 0, nocc)
        Lov = None
        with_df = self.with_df

        nvir = nmo - nocc
        naux = with_df.get_naoaux()
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        blksize = int(min(naux, max(with_df.blockdim,
                                    (max_memory*1e6/8-nocc*nvir**2*2)/(nocc*nvir))))
        for eri1 in with_df.loop(blksize=blksize):
            Lov = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', out=Lov)
            yield Lov

    def loop_ao2mo_ggoo_coco(self, mo_coeff, nocc):
        mo = numpy.asarray(mo_coeff, order='F')
        nmo = mo.shape[0]
        ijslice = (0, nocc , 0, nocc)
        Lov = None
        with_df = self.with_df

        nvir = nmo - nocc
        naux = with_df.get_naoaux()
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        blksize = int(min(naux, max(with_df.blockdim,
                                    (max_memory*1e6/8-nocc*nvir**2*2)/(nocc*nvir))))
        for eri1 in with_df.loop(blksize=blksize):
            Lov = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', out=Lov)
            yield Lov

    def loop_ao2mo_ggoo_cgcg(self, mo_coeff, nocc):
        mo = numpy.asarray(mo_coeff, order='F')
        #nmo = mo.shape[0]
        nmo = self.nocc + self.n_keep_vir
        ijslice = (0, nmo , 0, nmo)
        Lov = None
        with_df = self.with_df

        nvir = nmo - nocc
        naux = with_df.get_naoaux()
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        blksize = int(min(naux, max(with_df.blockdim,
                                    (max_memory*1e6/8-nocc*nvir**2*2)/(nocc*nvir))))
        for eri1 in with_df.loop(blksize=blksize):
            Lov = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', out=Lov)
            yield Lov

    def loop_ao2mo_cgcv(self, mo_coeff, nocc):
        mo = numpy.asarray(mo_coeff, order='F')
        #nmo = mo.shape[0]
        nmo = self.nocc + self.n_keep_vir
        Lov = None
        with_df = self.with_df

        #nvir = nmo - nocc
        nvir  = self.n_keep_vir
        ijslice = (0, nmo , nocc , nmo)
        naux = with_df.get_naoaux()
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        blksize = int(min(naux, max(with_df.blockdim,
                                    (max_memory*1e6/8-nocc*nvir**2*2)/(nocc*nvir))))
        for eri1 in with_df.loop(blksize=blksize):
            Lov = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', out=Lov)
            yield Lov

    

    

#RMP2 = MP2

#from pyscf import scf
#scf.hf.RHF.MP2 = lib.class_as_method(MP2)
#scf.rohf.ROHF.MP2 = None


def _mo_energy_without_core(mp, mo_energy):
    return mo_energy[get_frozen_mask(mp)]

def _mo_without_core(mp, mo):
    return mo[:,get_frozen_mask(mp)]

def _mem_usage(nocc, nvir):
    nmo = nocc + nvir
    basic = ((nocc*nvir)**2 + nocc*nvir**2*2)*8 / 1e6
    incore = nocc*nvir*nmo**2/2*8 / 1e6 + basic
    outcore = basic
    return incore, outcore, basic

class _ChemistsERIs:
    def __init__(self, mp, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        self.mo_coeff = _mo_without_core(mp, mo_coeff)

def _make_eris(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    log = logger.new_logger(mp, verbose)
    time0 = (time.clock(), time.time())
    eris = _ChemistsERIs(mp, mo_coeff)
    mo_coeff = eris.mo_coeff
    print('mo_coeff _make_eris shape =', mo_coeff.shape)
    nocc = mp.nocc
    nmo = mp.nmo
    #nvir = nmo - nocc
    nvir = mp.n_keep_vir
    mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mp.max_memory - mem_now)
    if max_memory < mem_basic:
        log.warn('Not enough memory for integral transformation. '
                 'Available mem %s MB, required mem %s MB',
                 max_memory, mem_basic)

    co = numpy.asarray(mo_coeff[:,:nocc], order='F')
    cv = numpy.asarray(mo_coeff[:,nocc:], order='F')
    if (mp.mol.incore_anyway or
        (mp._scf._eri is not None and mem_incore < max_memory)):
        log.debug('transform (ia|jb) incore')
        if callable(ao2mofn):
            eris.ovov = ao2mofn((co,cv,co,cv)).reshape(nocc*nvir,nocc*nvir)
        else:
            eris.ovov = ao2mo.general(mp._scf._eri, (co,cv,co,cv))

    elif getattr(mp._scf, 'with_df', None):
        # To handle the PBC or custom 2-electron with 3-index tensor.
        # Call dfmp2.MP2 for efficient DF-MP2 implementation.
        log.warn('DF-HF is found. (ia|jb) is computed based on the DF '
                 '3-tensor integrals.\n'
                 'You can switch to dfmp2.MP2 for better performance')
        log.debug('transform (ia|jb) with_df')
        eris.ovov = mp._scf.with_df.ao2mo((co,cv,co,cv))

    else:
        log.debug('transform (ia|jb) outcore')
        eris.feri = lib.H5TmpFile()
        #ao2mo.outcore.general(mp.mol, (co,cv,co,cv), eris.feri,
        #                      max_memory=max_memory, verbose=log)
        #eris.ovov = eris.feri['eri_mo']
        eris.ovov = _ao2mo_ovov(mp, co, cv, eris.feri, max(2000, max_memory), log)

    time1 = log.timer('Integral transformation', *time0)
    return eris

#
# the MO integral for MP2 is (ov|ov). This is the efficient integral
# (ij|kl) => (ij|ol) => (ol|ij) => (ol|oj) => (ol|ov) => (ov|ov)
#   or    => (ij|ol) => (oj|ol) => (oj|ov) => (ov|ov)
#
def _ao2mo_ovov(mp, orbo, orbv, feri, max_memory=2000, verbose=None):
    time0 = (time.clock(), time.time())
    log = logger.new_logger(mp, verbose)

    mol = mp.mol
    int2e = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, int2e, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    nao, nocc = orbo.shape
    nvir = orbv.shape[1]
    nbas = mol.nbas
    assert(nvir <= nao)

    ao_loc = mol.ao_loc_nr()
    dmax = max(4, min(nao/3, numpy.sqrt(max_memory*.95e6/8/(nao+nocc)**2)))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, dmax)
    dmax = max(x[2] for x in sh_ranges)
    eribuf = numpy.empty((nao,dmax,dmax,nao))
    ftmp = lib.H5TmpFile()
    log.debug('max_memory %s MB (dmax = %s) required disk space %g MB',
              max_memory, dmax, nocc**2*(nao*(nao+dmax)/2+nvir**2)*8/1e6)

    buf_i = numpy.empty((nocc*dmax**2*nao))
    buf_li = numpy.empty((nocc**2*dmax**2))
    buf1 = numpy.empty_like(buf_li)

    fint = gto.moleintor.getints4c
    jk_blk_slices = []
    count = 0
    time1 = time0
    with lib.call_in_background(ftmp.__setitem__) as save:
        for ip, (ish0, ish1, ni) in enumerate(sh_ranges):
            for jsh0, jsh1, nj in sh_ranges[:ip+1]:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                jk_blk_slices.append((i0,i1,j0,j1))

                eri = fint(int2e, mol._atm, mol._bas, mol._env,
                           shls_slice=(0,nbas,ish0,ish1, jsh0,jsh1,0,nbas),
                           aosym='s1', ao_loc=ao_loc, cintopt=ao2mopt._cintopt,
                           out=eribuf)
                tmp_i = numpy.ndarray((nocc,(i1-i0)*(j1-j0)*nao), buffer=buf_i)
                tmp_li = numpy.ndarray((nocc,nocc*(i1-i0)*(j1-j0)), buffer=buf_li)
                lib.ddot(orbo.T, eri.reshape(nao,(i1-i0)*(j1-j0)*nao), c=tmp_i)
                lib.ddot(orbo.T, tmp_i.reshape(nocc*(i1-i0)*(j1-j0),nao).T, c=tmp_li)
                tmp_li = tmp_li.reshape(nocc,nocc,(i1-i0),(j1-j0))
                save(str(count), tmp_li.transpose(1,0,2,3))
                buf_li, buf1 = buf1, buf_li
                count += 1
                time1 = log.timer_debug1('partial ao2mo [%d:%d,%d:%d]' %
                                         (ish0,ish1,jsh0,jsh1), *time1)
    time1 = time0 = log.timer('mp2 ao2mo_ovov pass1', *time0)
    eri = eribuf = tmp_i = tmp_li = buf_i = buf_li = buf1 = None

    h5dat = feri.create_dataset('ovov', (nocc*nvir,nocc*nvir), 'f8',
                                chunks=(nvir,nvir))
    occblk = int(min(nocc, max(4, 250/nocc, max_memory*.9e6/8/(nao**2*nocc)/5)))
    def load(i0, eri):
        if i0 < nocc:
            i1 = min(i0+occblk, nocc)
            for k, (p0,p1,q0,q1) in enumerate(jk_blk_slices):
                eri[:i1-i0,:,p0:p1,q0:q1] = ftmp[str(k)][i0:i1]
                if p0 != q0:
                    dat = numpy.asarray(ftmp[str(k)][:,i0:i1])
                    eri[:i1-i0,:,q0:q1,p0:p1] = dat.transpose(1,0,3,2)

    def save(i0, i1, dat):
        for i in range(i0, i1):
            h5dat[i*nvir:(i+1)*nvir] = dat[i-i0].reshape(nvir,nocc*nvir)

    orbv = numpy.asarray(orbv, order='F')
    buf_prefecth = numpy.empty((occblk,nocc,nao,nao))
    buf = numpy.empty_like(buf_prefecth)
    bufw = numpy.empty((occblk*nocc,nvir**2))
    bufw1 = numpy.empty_like(bufw)
    with lib.call_in_background(load) as prefetch:
        with lib.call_in_background(save) as bsave:
            load(0, buf_prefecth)
            for i0, i1 in lib.prange(0, nocc, occblk):
                buf, buf_prefecth = buf_prefecth, buf
                prefetch(i1, buf_prefecth)
                eri = buf[:i1-i0].reshape((i1-i0)*nocc,nao,nao)

                dat = _ao2mo.nr_e2(eri, orbv, (0,nvir,0,nvir), 's1', 's1', out=bufw)
                bsave(i0, i1, dat.reshape(i1-i0,nocc,nvir,nvir).transpose(0,2,1,3))
                bufw, bufw1 = bufw1, bufw
                time1 = log.timer_debug1('pass2 ao2mo [%d:%d]' % (i0,i1), *time1)

    time0 = log.timer('mp2 ao2mo_ovov pass2', *time0)
    return h5dat


del(WITH_T2)

def run_parallel(params):   
    object, alpha = params
    if alpha != None:
        object.alpha = alpha
    energy = object.run().e_tot
    
    return energy
    


def scan_n_keep_vir(max_n=100, outdir="Result"):
    import pandas as pd
    import time
    import os
    from datetime import datetime
    results = []
    mf = scf.RHF(mol).density_fit().run()
    mppp = B2PLYPDFOBMP2(mf)
    mppp.alpha= (0.53,0.1)
    # Thêm thời gian kết thúc (YYYYmmdd_HHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{outdir}/result_{mppp.nmo}_{mppp.thresh}_{timestamp}.xlsx"

    os.makedirs(outdir, exist_ok=True)

    for i in range(75,max_n):
        mf = scf.RHF(mol).density_fit().run()
        mppp = B2PLYPDFOBMP2(mf)
        mppp.alpha= (0.53,0.1)

        mppp.n_keep_vir = i + 1

        start = time.time()
        dhf = mppp.run()
        end = time.time()
        elapsed = end - start

        results.append({
            "n_keep_vir": i + 1,
            "e_tot": dhf.e_tot,
            "time_sec": elapsed
        })
        print(f"n_keep_vir={i+1}, e_tot={dhf.e_tot}, time={elapsed:.3f}s")

    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    return df, filename



if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    #from pyscf.mp import dfobmp2_faster_ram , dfmp2_native, mp2
    from pyscf.mp import mp2
    import dfobmp2_faster_ram
    mol = gto.Mole()
    mol.atom = [
            [9 , (0. , 0 , 0.6)],
            [9 , (0. , 0  , 0)]]
    mol.spin = 0
    mol.verbose= 3
    mol.basis = 'ccpvqz'
    mol.build()
    # mf = scf.UHF(mol).run()
    #df, fname = scan_n_keep_vir(max_n=101, outdir="Result")
    #print("File đã lưu:", fname)
    mf = scf.RHF(mol).density_fit().run()
    mppp = B2PLYPDFOBMP2(mf)
    mppp.alpha= (0.53,0.1)
    mppp.n_keep_vir = 99
    dhf = mppp.run()
    # print('dftobmp2=',dhf.e_tot)
    # ks = dft.RKS(mol, f"0.53*HF+ 0.47*B88,LYP").density_fit().run()
    # mpp=dfobmp2_faster_ram.DFOBMP2(mf).run()
    # mf = scf.RHF(mol).density_fit().run()
    # mp22=dfmp2_native.DFMP2(mf).run()
    # print('alpha=',mppp.alpha)
     
    # print('hf=', mf.e_tot) 
    # print('dft=', ks.e_tot)  
    # print('DFOBMP2=',mpp.e_tot)
    # print('mp2=',mp22.e_tot)
    #   export OPENBLAS_NUM_THREADS=1

    #-195.426163377709
    
    
