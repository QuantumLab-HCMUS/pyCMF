import time
import numpy
import scipy.linalg
from pyscf import lib, dft, scf
from pyscf.lib import logger
import tracemalloc

def make_veff(mp):
    nocc = mp.nocc
    nocca, noccb = mp.get_nocc()
    mo_coeff = mp.mo_coeff
    mo_occ = mp.mo_occ

    dm = mp._scf.make_rdm1(mo_coeff, mo_occ)
    veff_ao = mp._scf.get_veff(mp.mol, dm)

    veffa = numpy.matmul(mo_coeff[0].T, numpy.matmul(veff_ao[0], mo_coeff[0]))
    veffb = numpy.matmul(mo_coeff[1].T, numpy.matmul(veff_ao[1], mo_coeff[1]))

    c0_hf = 0.
    for i in range(nocc[0]):
        c0_hf -= veffa[i, i]
    for i in range(nocc[1]):
        c0_hf -= veffb[i, i]

    return veffa, veffb, c0_hf

def make_amp(mp):
    css = mp.css
    cos = mp.cos
    log = logger.new_logger(mp, verbose=5)

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa - nocca, nmob - noccb
    mo_energy = mp.mo_energy
    mo_coeff = mp.mo_coeff
    
    t0 = (time.process_time(), time.time())
    from pyscf.lib import current_memory
    tracemalloc.start()

    for istep, qov_a in enumerate(mp.loop_ao2mo(mo_coeff[0], nocca, nmoa)):
        pass
    for istep, qov_b in enumerate(mp.loop_ao2mo(mo_coeff[1], noccb, nmob)):
        pass

    log.debug("qov_ab memory: %.1f MiB", current_memory()[0])
    t1 = log.timer('making amplitude: integral transform', *t0)

    x_aa = numpy.tile(mo_energy[0][:nocca,None] - mo_energy[0][None,nocca:],(nocca,nvira,1,1))
    x_aa += numpy.einsum('ijkl -> klij', x_aa) - mp.shift
    tmp1_aa = 1. *css* numpy.dot(qov_a.T,qov_a).reshape(nocca,nvira,nocca,nvira)/x_aa
    del(x_aa)

    x_ab = numpy.einsum('ijkl -> klij',numpy.tile(mo_energy[0][:nocca,None] - mo_energy[0][None,nocca:],(noccb,nvirb,1,1)))
    x_ab += numpy.tile(mo_energy[1][:noccb,None] - mo_energy[1][None,noccb:],(nocca,nvira,1,1)) - mp.shift
    tmp1_ab = 1. *cos* numpy.dot(qov_a.T,qov_b).reshape(nocca,nvira,noccb,nvirb)/x_ab
    del(x_ab)

    x_ba = numpy.einsum('ijkl -> klij',numpy.tile(mo_energy[1][:noccb,None] - mo_energy[1][None,noccb:],(nocca,nvira,1,1)))
    x_ba += numpy.tile(mo_energy[0][:nocca,None] - mo_energy[0][None,nocca:],(noccb,nvirb,1,1)) - mp.shift
    tmp1_ba = 1. *cos* numpy.dot(qov_b.T,qov_a).reshape(noccb,nvirb,nocca,nvira)/x_ba
    del(x_ba)
    del(qov_a)

    x_bb = numpy.tile(mo_energy[1][:noccb,None] - mo_energy[1][None,noccb:],(noccb,nvirb,1,1))
    x_bb += numpy.einsum('ijkl -> klij', x_bb) - mp.shift
    tmp1_bb = 1. *css* numpy.dot(qov_b.T,qov_b).reshape(noccb,nvirb,noccb,nvirb)/x_bb
    del(x_bb)
    del(qov_b)
    
    tmp1_bar_aa = tmp1_aa - numpy.transpose(tmp1_aa,(0,3,2,1))
    tmp1_bar_bb = tmp1_bb - numpy.transpose(tmp1_bb,(0,3,2,1))
    tmp1_bar_ab = tmp1_ab
    tmp1_bar_ba = tmp1_ba

    log.debug("t_mp1 memory: %.1f MiB", current_memory()[0])
    
    tmp1 = (tmp1_aa, tmp1_bb, tmp1_ab, tmp1_ba)
    tmp1_bar = (tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab, tmp1_bar_ba)

    return tmp1, tmp1_bar

def first_BCH(mp, fock_hfa, fock_hfb, tmp1_bar, c0):
    log = logger.new_logger(mp, verbose=5)
    tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab, tmp1_bar_ba = tmp1_bar
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb
    mo_coeff = mp.mo_coeff
    naux = mp.with_df.get_naoaux()
    
    t0 = (time.process_time(), time.time())
    from pyscf.lib import current_memory
    tracemalloc.start()

    t1 = log.timer('first BCH: integral transform', *t0)
    c1_a = numpy.zeros((nmoa,nmoa), dtype=fock_hfa.dtype)
    c1_b = numpy.zeros((nmob,nmob), dtype=fock_hfb.dtype)

    for istep, qov_b in enumerate(mp.loop_ao2mo(mo_coeff[1], noccb, nmob)):
        pass

    for istep, qgv_a in enumerate(mp.loop_ao2mo_cgcv(mo_coeff[0], nocca, nmoa)):
        for i in range(nocca):
            c1_a[:,0:nocca] += 2. * lib.einsum("apb, ajb -> pj",numpy.dot(qgv_a.reshape(naux, nmoa, nvira)[:,:nocca,:nvira].reshape(naux, nocca*nvira)[:,i*nvira:(i+1)*nvira].T,qgv_a).reshape(nvira,nmoa,nvira),tmp1_bar_aa[i,:,:,:]) 
        for i in range(noccb):
            c1_a[:,0:nocca] += 2. * lib.einsum("apb, ajb -> pj",numpy.dot(qov_b[:,i*nvirb:(i+1)*nvirb].T,qgv_a).reshape(nvirb,nmoa,nvira),tmp1_bar_ba[i,:,:,:])
    
    del(qgv_a)
    
    for istep, qov_a in enumerate(mp.loop_ao2mo(mo_coeff[0], nocca, nmoa)):
        for i in range(nocca):
            c0 -= 1.*lib.einsum("ajb, ajb -> ", numpy.dot(qov_a[:,i*nvira:(i+1)*nvira].T, qov_a).reshape(nvira, nocca, nvira), tmp1_bar_aa[i,:,:,:])
    
    for i in range(nocca):
        c0 -= 1.*lib.einsum("ajb, ajb -> ", numpy.dot(qov_a[:,i*nvira:(i+1)*nvira].T, qov_b).reshape(nvira, noccb, nvirb), tmp1_bar_ab[i,:,:,:])
    
    for istep, qgv_b in enumerate(mp.loop_ao2mo_cgcv(mo_coeff[1], noccb, nmob)):
        for i in range(noccb):
            c1_b[:,0:noccb] += 2. * lib.einsum("apb, ajb -> pj",numpy.dot(qgv_b.reshape(naux, nmob, nvirb)[:,:noccb,:nvirb].reshape(naux, noccb*nvirb)[:,i*nvirb:(i+1)*nvirb].T,qgv_b).reshape(nvirb,nmob,nvirb),tmp1_bar_bb[i,:,:,:]) 
        for i in range(nocca):
            c1_b[:,0:noccb] += 2. * lib.einsum("apb, ajb -> pj",numpy.dot(qov_a[:,i*nvira:(i+1)*nvira].T,qgv_b).reshape(nvira,nmob,nvirb),tmp1_bar_ab[i,:,:,:])  
    
    del(qgv_b) 

    for i in range(noccb):
        c0 -= 1.*lib.einsum("ajb, ajb -> ", numpy.dot(qov_b[:,i*nvirb:(i+1)*nvirb].T, qov_a).reshape(nvirb, nocca, nvira), tmp1_bar_ba[i,:,:,:])
        c0 -= 1.*lib.einsum("ajb, ajb -> ", numpy.dot(qov_b[:,i*nvirb:(i+1)*nvirb].T, qov_b).reshape(nvirb, noccb, nvirb), tmp1_bar_bb[i,:,:,:])
    
    for istep, qog_a in enumerate(mp.loop_ao2mo_goog_cocg(mo_coeff[0], nocca, nmoa)):
        for i in range(nocca):
            c1_a[:,nocca:nmoa] -= 2.*lib.einsum("ajp, ajb -> pb", numpy.dot(qov_a[:,i*nvira:(i+1)*nvira].T,qog_a).reshape(nvira,nocca, nmoa),tmp1_bar_aa[i,:,:,:]) 
        for i in range(noccb):
            c1_a[:,nocca:nmoa] -= 2.*lib.einsum("ajp, ajb -> pb", numpy.dot(qov_b[:,i*nvirb:(i+1)*nvirb].T,qog_a).reshape(nvirb,nocca, nmoa),tmp1_bar_ba[i,:,:,:])

    del(qog_a)
    
    for istep, qog_b in enumerate(mp.loop_ao2mo_goog_cocg(mo_coeff[1], noccb, nmob)):
        for i in range(noccb):
            c1_b[:,noccb:nmob] -= 2.* lib.einsum("ajp, ajb -> pb", numpy.dot(qov_b[:,i*nvirb:(i+1)*nvirb].T,qog_b).reshape(nvirb,noccb, nmob),tmp1_bar_bb[i,:,:,:]) 
        for i in range(nocca):
            c1_b[:,noccb:nmob] -= 2.* lib.einsum("ajp, ajb -> pb", numpy.dot(qov_a[:,i*nvira:(i+1)*nvira].T,qog_b).reshape(nvira,noccb, nmob),tmp1_bar_ab[i,:,:,:]) 
    
    del(qog_b)

    c1_a[:nocca,nocca:] += 2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_aa,fock_hfa[:nocca,nocca:])
    c1_a[:nocca,nocca:] += 2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_ba,fock_hfb[:noccb,noccb:])
    c1_b[:noccb,noccb:] += 2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_bb,fock_hfb[:noccb,noccb:])
    c1_b[:noccb,noccb:] += 2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_ab,fock_hfa[:nocca,nocca:])

    return c0, c1_a, c1_b

def second_BCH(mp, fock_a, fock_b, fock_hfa, fock_hfb, tmp1, tmp1_bar, c0):
    tmp1_aa, tmp1_bb, tmp1_ab, tmp1_ba = tmp1
    tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab, tmp1_bar_ba = tmp1_bar
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()

    c1_a = numpy.zeros((nmoa,nmoa), dtype=fock_hfa.dtype)
    c1_b = numpy.zeros((nmob,nmob), dtype=fock_hfb.dtype)
    
    #[1]
    y1_a = lib.einsum('ij,ijkl -> kl', fock_hfa[:nocca,nocca:], tmp1_bar_aa)
    y1_a += lib.einsum('ij,ijkl -> kl', fock_hfb[:noccb,noccb:], tmp1_bar_ba)
    c1_a[:nocca,nocca:] += lib.einsum('ijkl,kl -> ij', tmp1_bar_aa,y1_a)
    c1_b[:noccb,noccb:] += lib.einsum('ijkl,kl -> ij', tmp1_bar_ba,y1_a)
    
    y1_b = lib.einsum('ij,ijkl -> kl', fock_hfb[:noccb,noccb:], tmp1_bar_bb)
    y1_b += lib.einsum('ij,ijkl -> kl', fock_hfa[:nocca,nocca:], tmp1_bar_ab)
    c1_a[:nocca,nocca:] += lib.einsum('ijkl,kl -> ij', tmp1_bar_ab,y1_b)
    c1_b[:noccb,noccb:] += lib.einsum('ijkl,kl -> ij', tmp1_bar_bb,y1_b)

    #[2]
    y1_aa = lib.einsum('ac,kcjb -> kajb',fock_hfa[nocca:,nocca:],tmp1_bar_aa)
    y1_ab = lib.einsum('ac,kcjb -> kajb',fock_hfa[nocca:,nocca:],tmp1_bar_ab)
    y1_ba = lib.einsum('ac,kcjb -> kajb',fock_hfb[noccb:,noccb:],tmp1_bar_ba)
    y1_bb = lib.einsum('ac,kcjb -> kajb',fock_hfb[noccb:,noccb:],tmp1_bar_bb)
    c1_a[:nocca,:nocca] += lib.einsum('iajb,iakb -> jk', tmp1_aa, y1_aa)
    c1_a[:nocca,:nocca] += lib.einsum('iajb,iakb -> jk', tmp1_ba, y1_ba)
    c1_b[:noccb,:noccb] += lib.einsum('iajb,iakb -> jk', tmp1_bb, y1_bb)
    c1_b[:noccb,:noccb] += lib.einsum('iajb,iakb -> jk', tmp1_ab, y1_ab)
    
    c0 -= lib.einsum('ijkl,ijkl->', tmp1_aa,y1_aa) + lib.einsum('ijkl,ijkl->', tmp1_bb,y1_bb)
    c0 -= lib.einsum('ijkl,ijkl->', tmp1_ab,y1_ab) + lib.einsum('ijkl,ijkl->', tmp1_ba,y1_ba)

    #[3]
    y1_aa = lib.einsum('ac,kcjb -> kajb',fock_hfa[nocca:,nocca:],tmp1_bar_aa)
    y1_ab = lib.einsum('ac,kcjb -> kajb',fock_hfa[nocca:,nocca:],tmp1_bar_ab)
    y1_ba = lib.einsum('ac,kcjb -> kajb',fock_hfb[noccb:,noccb:],tmp1_bar_ba)
    y1_bb = lib.einsum('ac,kcjb -> kajb',fock_hfb[noccb:,noccb:],tmp1_bar_bb)
    c1_a[:nocca,:nocca] += lib.einsum('iajb,kajb -> ik', tmp1_aa, y1_aa)
    c1_a[:nocca,:nocca] += lib.einsum('iajb,kajb -> ik', tmp1_ab, y1_ab)
    c1_b[:noccb,:noccb] += lib.einsum('iajb,kajb -> ik', tmp1_bb, y1_bb)
    c1_b[:noccb,:noccb] += lib.einsum('iajb,kajb -> ik', tmp1_ba, y1_ba)
                            
    #[4]
    y1_aa = lib.einsum('ik,kalb -> ialb',fock_hfa[:nocca,:nocca],tmp1_bar_aa)
    y1_ab = lib.einsum('ik,kalb -> ialb',fock_hfa[:nocca,:nocca],tmp1_bar_ab)
    y1_ba = lib.einsum('ik,kalb -> ialb',fock_hfb[:noccb,:noccb],tmp1_bar_ba)
    y1_bb = lib.einsum('ik,kalb -> ialb',fock_hfb[:noccb,:noccb],tmp1_bar_bb)
    c1_a[:nocca,:nocca] -= lib.einsum('iajb,ialb -> jl', tmp1_aa, y1_aa)
    c1_a[:nocca,:nocca] -= lib.einsum('iajb,ialb -> jl', tmp1_ba, y1_ba)
    c1_b[:noccb,:noccb] -= lib.einsum('iajb,ialb -> jl', tmp1_bb, y1_bb)
    c1_b[:noccb,:noccb] -= lib.einsum('iajb,ialb -> jl', tmp1_ab, y1_ab)
    
    c0 += lib.einsum('ijkl,ijkl->', tmp1_aa,y1_aa) + lib.einsum('ijkl,ijkl->', tmp1_bb,y1_bb)
    c0 += lib.einsum('ijkl,ijkl->', tmp1_ab,y1_ab) + lib.einsum('ijkl,ijkl->', tmp1_ba,y1_ba)

    #[5]
    y1_a  = lib.einsum('iajb,kajb -> ik', tmp1_aa, tmp1_bar_aa)
    y1_a += lib.einsum('iajb,kajb -> ik', tmp1_ab, tmp1_bar_ab)
    c1_a[:,:nocca] -= lib.einsum('pk,ik -> pi', fock_hfa[:,:nocca], y1_a)

    y1_b  = lib.einsum('iajb,kajb -> ik', tmp1_bb, tmp1_bar_bb)
    y1_b += lib.einsum('iajb,kajb -> ik', tmp1_ba, tmp1_bar_ba)
    c1_b[:,:noccb] -= lib.einsum('pk,ik -> pi', fock_hfb[:,:noccb], y1_b)
    
    #[6]
    y1_aa = lib.einsum('ik,kajd -> iajd',fock_hfa[:nocca,:nocca],tmp1_bar_aa)
    y1_ab = lib.einsum('ik,kajd -> iajd',fock_hfa[:nocca,:nocca],tmp1_bar_ab)
    y1_ba = lib.einsum('ik,kajd -> iajd',fock_hfb[:noccb,:noccb],tmp1_bar_ba)
    y1_bb = lib.einsum('ik,kajd -> iajd',fock_hfb[:noccb,:noccb],tmp1_bar_bb)
    c1_a[nocca:,nocca:] += lib.einsum('iajb,iajd -> bd', tmp1_aa, y1_aa)
    c1_a[nocca:,nocca:] += lib.einsum('iajb,iajd -> bd', tmp1_ba, y1_ba)
    c1_b[noccb:,noccb:] += lib.einsum('iajb,iajd -> bd', tmp1_bb, y1_bb)
    c1_b[noccb:,noccb:] += lib.einsum('iajb,iajd -> bd', tmp1_ab, y1_ab)

    #[7]
    y1_aa = lib.einsum('ik,kajd -> iajd',fock_hfa[:nocca,:nocca],tmp1_bar_aa)
    y1_ab = lib.einsum('ik,kajd -> iajd',fock_hfa[:nocca,:nocca],tmp1_bar_ab)
    y1_ba = lib.einsum('ik,kajd -> iajd',fock_hfb[:noccb,:noccb],tmp1_bar_ba)
    y1_bb = lib.einsum('ik,kajd -> iajd',fock_hfb[:noccb,:noccb],tmp1_bar_bb)
    c1_a[nocca:,nocca:] += lib.einsum('iajb,icjb -> ac', tmp1_aa, y1_aa)
    c1_a[nocca:,nocca:] += lib.einsum('iajb,icjb -> ac', tmp1_ab, y1_ab)
    c1_b[noccb:,noccb:] += lib.einsum('iajb,icjb -> ac', tmp1_bb, y1_bb)
    c1_b[noccb:,noccb:] += lib.einsum('iajb,icjb -> ac', tmp1_ba, y1_ba)

    #[8]
    y1_aa = lib.einsum('ac,icjd -> iajd',fock_hfa[nocca:,nocca:],tmp1_bar_aa)
    y1_ab = lib.einsum('ac,icjd -> iajd',fock_hfa[nocca:,nocca:],tmp1_bar_ab)
    y1_ba = lib.einsum('ac,icjd -> iajd',fock_hfb[noccb:,noccb:],tmp1_bar_ba)
    y1_bb = lib.einsum('ac,icjd -> iajd',fock_hfb[noccb:,noccb:],tmp1_bar_bb)
    c1_a[nocca:,nocca:] -= lib.einsum('iajb,iajd -> bd', tmp1_aa, y1_aa)
    c1_a[nocca:,nocca:] -= lib.einsum('iajb,iajd -> bd', tmp1_ba, y1_ba)
    c1_b[noccb:,noccb:] -= lib.einsum('iajb,iajd -> bd', tmp1_bb, y1_bb)
    c1_b[noccb:,noccb:] -= lib.einsum('iajb,iajd -> bd', tmp1_ab, y1_ab)

    #[9]
    y1_a  = lib.einsum('iajb,icjb -> ac', tmp1_aa, tmp1_bar_aa)
    y1_a += lib.einsum('iajb,icjb -> ac', tmp1_ab, tmp1_bar_ab)
    c1_a[:,nocca:] -= lib.einsum('pa,ac -> pc', fock_hfa[:,nocca:], y1_a)
    y1_b  = lib.einsum('iajb,icjb -> ac', tmp1_bb, tmp1_bar_bb)
    y1_b += lib.einsum('iajb,icjb -> ac', tmp1_ba, tmp1_bar_ba)
    c1_b[:,noccb:] -= lib.einsum('pa,ac -> pc', fock_hfb[:,noccb:], y1_b)

    return c0, c1_a, c1_b

def obmp2_iter(mp, mol, mf_emb, xc_code, v_emb=None, niter=1000):
    nmoa = mf_emb.mo_coeff[0].shape[1] 
    nmob = mf_emb.mo_coeff[1].shape[1]
    nocca = numpy.count_nonzero(mf_emb.mo_occ[0] > 0)
    noccb = numpy.count_nonzero(mf_emb.mo_occ[1] > 0)

    dm = mf_emb.make_rdm1(mf_emb.mo_coeff, mf_emb.mo_occ)
    s1e = mf_emb.get_ovlp(mol)
    h1e = mf_emb.get_hcore(mol)
    vhf = mf_emb.get_veff(mol, dm)
    nuc = mf_emb.energy_nuc()
    
    ks = dft.UKS(mol)
    ks.xc = xc_code
    ks.verbose = 0
    ks = ks.density_fit()

    F_list_a = []
    DIIS_RESID_a = []
    F_list_b = []
    DIIS_RESID_b = []
    
    ene_old = 0.0
    conv = False

    if v_emb is None:
        v_emb = [0, 0]

    for it in range(niter):
        h1ao = mf_emb.get_hcore(mol)
        h1mo_a = numpy.matmul(mf_emb.mo_coeff[0].T, numpy.matmul(h1ao, mf_emb.mo_coeff[0]))
        h1mo_b = numpy.matmul(mf_emb.mo_coeff[1].T, numpy.matmul(h1ao, mf_emb.mo_coeff[1]))
        
        fock_hfa = h1mo_a
        fock_hfb = h1mo_b

        mp.mo_coeff = mf_emb.mo_coeff
        mp.mo_occ = mf_emb.mo_occ
        mp.mo_energy = mf_emb.mo_energy
        mp._scf = mf_emb 
        mp._nocc = (nocca, noccb)
        mp._nmo = (nmoa, nmob)

        veffa, veffb, c0 = make_veff(mp) 
        
        fock_hfa += veffa
        fock_hfb += veffb

        fock_uobmp2_a = numpy.zeros((nmoa,nmoa), dtype=fock_hfa.dtype)
        fock_uobmp2_b = numpy.zeros((nmob,nmob), dtype=fock_hfb.dtype)

        fock_uobmp2_a += fock_hfa
        fock_uobmp2_b += fock_hfb

        ene_hf = 0.
        for i in range(nocca): ene_hf += fock_uobmp2_a[i,i]
        for i in range(noccb): ene_hf += fock_uobmp2_b[i,i]
        c0 *= 0.5
        ene_hf += c0

        vxc = ks.get_veff(mol, dm)
        fock_dft_raw = ks.get_fock(h1e, s1e, vxc, dm, diis_start_cycle=it)
        fock_dft = numpy.array([fock_dft_raw[0] + v_emb[0], fock_dft_raw[1] + v_emb[1]])
        
        fock_dft_a = numpy.matmul(mf_emb.mo_coeff[0].T, numpy.matmul(fock_dft[0], mf_emb.mo_coeff[0]))
        fock_dft_b = numpy.matmul(mf_emb.mo_coeff[1].T, numpy.matmul(fock_dft[1], mf_emb.mo_coeff[1]))
        ene_dft = ks.energy_elec(dm, h1e, vxc)[0] + nuc

        vhf = mf_emb.get_veff(mol, dm)
        fock_hf_pyscf = mf_emb.get_fock(h1e, s1e, vhf, dm, diis_start_cycle=it) 
        
        fock_hf_pyscf_a = numpy.matmul(mf_emb.mo_coeff[0].T, numpy.matmul(fock_hf_pyscf[0], mf_emb.mo_coeff[0]))
        fock_hf_pyscf_b = numpy.matmul(mf_emb.mo_coeff[1].T, numpy.matmul(fock_hf_pyscf[1], mf_emb.mo_coeff[1]))
        
        e_elec_hfpyscf = mf_emb.energy_elec(dm, h1e, vhf)[0]
        ene_hfpyscf = e_elec_hfpyscf + nuc

        tmp1, tmp1_bar = make_amp(mp) 
        tmp1_aa, tmp1_bb, tmp1_ab, tmp1_ba = tmp1
        tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab, tmp1_bar_ba = tmp1_bar

        if mp.second_order:
            mp.ampf = 1.0

        tmp1_bar_aa *= mp.ampf
        tmp1_bar_bb *= mp.ampf
        tmp1_bar_ab *= mp.ampf
        tmp1_bar_ba *= mp.ampf

        c0, c1_a, c1_b = first_BCH(mp, fock_hfa, fock_hfb, tmp1_bar, c0)
        
        fock_uobmp2_a += 0.5 * (c1_a + c1_a.T)
        fock_uobmp2_b += 0.5 * (c1_b + c1_b.T)  

        if mp.second_order:
            c0, c1_a, c1_b = second_BCH(mp, fock_uobmp2_a, fock_uobmp2_b, fock_hfa, fock_hfb, tmp1, tmp1_bar, c0)
            fock_uobmp2_a += 0.5 * (c1_a + c1_a.T)
            fock_uobmp2_b += 0.5 * (c1_b + c1_b.T) 

        ene = c0
        for i in range(nocca): ene += 1. * fock_uobmp2_a[i,i]
        for i in range(noccb): ene += 1. * fock_uobmp2_b[i,i]
        
        ene_uobmp2 = ene + nuc

        e_tot = (ene_dft) + (ene_uobmp2 - ene_hfpyscf) * mp.alphaa[1]
        e_corr = (ene_uobmp2 - ene_hfpyscf) * mp.alphaa[1]

        fock_udftobmp2_a = (fock_dft_a) + (fock_uobmp2_a - fock_hf_pyscf_a) * mp.alphaa[1] 
        fock_udftobmp2_b = (fock_dft_b) + (fock_uobmp2_b - fock_hf_pyscf_b) * mp.alphaa[1] 

        de = abs(e_corr - ene_old)
        ene_old = e_corr
        
        print(f"Iter {it}: E_corr={e_corr:.8f}, dE={de:.8e}")

        # DIIS
        F_eff_mo_a = fock_udftobmp2_a 
        F_eff_mo_b = fock_udftobmp2_b 
        
        F_a = s1e @ mf_emb.mo_coeff[0] @ F_eff_mo_a @ mf_emb.mo_coeff[0].T @ s1e
        F_b = s1e @ mf_emb.mo_coeff[1] @ F_eff_mo_b @ mf_emb.mo_coeff[1].T @ s1e
        
        C_occa = mp.mo_coeff[0][:, :nocca]
        C_occb = mp.mo_coeff[1][:, :noccb]
        
        D_a = numpy.einsum('pi,qi->pq', C_occa, C_occa, optimize=True)
        D_b = numpy.einsum('pi,qi->pq', C_occb, C_occb, optimize=True)

        err_a_ao = F_a.dot(D_a).dot(s1e) - s1e.dot(D_a).dot(F_a)
        err_ab_ao = F_a.dot(D_a).dot(s1e) - s1e.dot(D_b).dot(F_b)
        err_ba_ao = F_b.dot(D_b).dot(s1e) - s1e.dot(D_a).dot(F_a)
        err_a_mo = numpy.matmul(mp.mo_coeff[0].T,numpy.matmul(err_a_ao,mp.mo_coeff[0]))
        err_b_ao = F_b.dot(D_b).dot(s1e) - s1e.dot(D_b).dot(F_b)
        err_b_mo = numpy.matmul(mp.mo_coeff[1].T,numpy.matmul(err_b_ao,mp.mo_coeff[1]))
        err_ab_mo = mp.mo_coeff[0].T @ err_ab_ao @ mp.mo_coeff[0]
        err_ba_mo = mp.mo_coeff[0].T @ err_ba_ao @ mp.mo_coeff[0]

        diis_r_a = (err_a_mo + err_b_mo + 50*err_ab_mo + 50*err_ba_mo).real
        F_list_a.append(F_a)
        DIIS_RESID_a.append(diis_r_a) 

        if it >= 2:
            B_dim_a = len(F_list_a) + 1
            B_a = numpy.empty((B_dim_a, B_dim_a))
            B_a[-1, :] = -1
            B_a[:, -1] = -1
            B_a[-1, -1] = 0
            for i in range(len(F_list_a)):
                for j in range(len(F_list_a)):
                    B_a[i, j] = numpy.einsum('ij,ij->', DIIS_RESID_a[i], DIIS_RESID_a[j], optimize=True)

            rhs_a = numpy.zeros((B_dim_a))
            rhs_a[-1] = -1
            coeff_a = numpy.linalg.solve(B_a, rhs_a)
            
            F_a = numpy.zeros_like(F_a)
            for x in range(coeff_a.shape[0] - 1):
                F_a += coeff_a[x] * F_list_a[x]

        err_ab_mo_b = mp.mo_coeff[1].T @ err_ab_ao @ mp.mo_coeff[1]
        err_ba_mo_b = mp.mo_coeff[1].T @ err_ba_ao @ mp.mo_coeff[1]
        err_b_mo_b  = mp.mo_coeff[1].T @ err_b_ao  @ mp.mo_coeff[1]
        err_a_mo_b  = mp.mo_coeff[1].T @ err_a_ao  @ mp.mo_coeff[1]
        diis_r_b = (1*err_a_mo_b + 1*err_b_mo_b + 50*err_ab_mo_b + 50*err_ba_mo_b).real

        F_list_b.append(F_b)
        DIIS_RESID_b.append(diis_r_b)

        if it >= 2:
            B_dim_b = len(F_list_b) + 1
            B_b = numpy.empty((B_dim_b, B_dim_b))
            B_b[-1, :] = -1
            B_b[:, -1] = -1
            B_b[-1, -1] = 0
            for i in range(len(F_list_b)):
                for j in range(len(F_list_b)):
                    B_b[i, j] = numpy.einsum('ij,ij->', DIIS_RESID_b[i], DIIS_RESID_b[j], optimize=True)

            rhs_b = numpy.zeros((B_dim_b))
            rhs_b[-1] = -1
            coeff_b = numpy.linalg.solve(B_b, rhs_b)
            
            F_b = numpy.zeros_like(F_b)
            for x in range(coeff_b.shape[0] - 1):
                F_b += coeff_b[x] * F_list_b[x]
        
        F_a_mo = mf_emb.mo_coeff[0].T @ F_a @ mf_emb.mo_coeff[0] 
        F_b_mo = mf_emb.mo_coeff[1].T @ F_b @ mf_emb.mo_coeff[1] 

        eps_new_a, C_rot_a = scipy.linalg.eigh(F_a_mo)
        eps_new_b, C_rot_b = scipy.linalg.eigh(F_b_mo)

        mo_coeff_new_a = mf_emb.mo_coeff[0] @ C_rot_a
        mo_coeff_new_b = mf_emb.mo_coeff[1] @ C_rot_b

        mf_emb.mo_coeff  = (mo_coeff_new_a, mo_coeff_new_b)
        mf_emb.mo_energy = (eps_new_a,      eps_new_b)

        mp.mo_coeff  = mf_emb.mo_coeff
        mp.mo_energy = mf_emb.mo_energy

        if de <= mp.thresh:
            conv = True
            break
    
        dm = mf_emb.make_rdm1(mf_emb.mo_coeff, mf_emb.mo_occ)
        dm = lib.tag_array(dm, mo_coeff=mf_emb.mo_coeff, mo_occ=mf_emb.mo_occ)

    dm_total = mf_emb.make_rdm1(mf_emb.mo_coeff, mf_emb.mo_occ)
    return e_tot, ene_dft, (dm_total[0], dm_total[1])