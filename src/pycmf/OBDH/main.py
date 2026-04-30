import time
from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import gto, df, lib, scf, dft, ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf import __config__
from pyscf.lib import logger
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor

from pycmf.OBMP import DFOBMP2 
# import obmp2
from uobdh_solver import obmp2_iter, make_amp
from uobdh_embed import embed_kernel

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)

def make_S2(mp, tmp1_bar_ab):
    mo_coeff = mp.mo_coeff
    mo_occ   = mp.mo_occ
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()

    Sao = mp._scf.get_ovlp()
    ss_ref, s = mp._scf.spin_square((mo_coeff[0][:,mo_occ[0]>0], mo_coeff[1][:,mo_occ[1]>0]), Sao)
    Sib_AB_ = numpy.matmul(mo_coeff[0].T, numpy.matmul(Sao, mo_coeff[1]))
    Sja_BA_ = numpy.matmul(mo_coeff[1].T, numpy.matmul(Sao, mo_coeff[0]))
    Sib_AB = Sib_AB_[:nocca,noccb:nmob]
    Sja_BA = Sja_BA_[:noccb,nocca:nmoa]
    tmp = lib.einsum("ja,iajb -> ib", Sja_BA, tmp1_bar_ab)
    ss_res = ss_ref - 0.5*numpy.sum(Sib_AB*tmp)
    ss_prj = ss_ref - 1.0*numpy.sum(Sib_AB*tmp)

    return ss_ref, ss_res, ss_prj

def make_IPEA(mp):
    eV = 27.2114
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    mo_energy = mp.mo_energy
    mo_coeff  = mp.mo_coeff
    naux = mp.with_df.get_naoaux()

    for istep, qgg_a in enumerate(mp.loop_ao2mo_ggoo_cgcg(mo_coeff[0], nocca)):
        qgg_a = qgg_a.reshape(naux, nmoa, nmoa)
    qov_a = qgg_a[:,0:nocca,nocca:]
    qoo_a = qgg_a[:,0:nocca,0:nocca]

    for istep, qgg_b in enumerate(mp.loop_ao2mo_ggoo_cgcg(mo_coeff[1], noccb)):
        qgg_b = qgg_b.reshape(naux, nmob, nmob)
    qov_b = qgg_b[:,:noccb,noccb:]
    qvv_b = qgg_b[:,noccb:,noccb:]

    ipea = []
    for h in range(nocca):
        x_a = mo_energy[0][:nocca,None,None] + mo_energy[0][None,None, :nocca] - mo_energy[0][None,nocca:,None] - mo_energy[0][None,h,None]  - mp.shift
        x_b = mo_energy[1][:noccb,None,None] + mo_energy[0][None,None, :nocca] - mo_energy[1][None,noccb:,None] - mo_energy[0][None,h,None]  - mp.shift

        tmp1_aa = mp.css * numpy.einsum("Lia, Lj -> iaj", qov_a, qoo_a[:,:,h]) / x_a
        tmp1_ba = mp.cos * numpy.einsum("Lia, Lj -> iaj", qov_b, qoo_a[:,:,h]) / x_b

        tmp1_bar_aa = mp.ampf * (tmp1_aa - numpy.transpose(tmp1_aa,(2,1,0)))
        tmp1_bar_ba = mp.ampf * tmp1_ba

        tmp2 = numpy.einsum("iaj, iaj -> ", tmp1_bar_aa, numpy.einsum("Lia, Lj -> iaj", qov_a, qoo_a[:,:,h]))
        tmp2 += numpy.einsum("iaj, iaj -> ", tmp1_bar_ba, numpy.einsum("Lia, Lj -> iaj", qov_b, qoo_a[:,:,h]))

        ip_obmp2 = eV*(-mo_energy[0][h] + 1.*tmp2)
        ipea.append(ip_obmp2)

    L = noccb
    x_b = mo_energy[1][:noccb,None,None] + mo_energy[1][None,L,None] - mo_energy[1][None,noccb:,None] - mo_energy[1][None,None,noccb:]  - mp.shift
    tmp1_bb = mp.css * numpy.einsum("Lia, Lb -> iab", qov_b, qvv_b[:,0,:]) / x_b

    x_a = mo_energy[0][:nocca,None,None] + mo_energy[1][None,L,None] - mo_energy[0][None,nocca:,None] - mo_energy[1][None,None,noccb:]  - mp.shift
    tmp1_ab = mp.cos * numpy.einsum("Lia, Lb -> iab", qov_a, qvv_b[:,0,:]) / x_a

    tmp1_bar_bb = mp.ampf * (tmp1_bb - numpy.transpose(tmp1_bb,(0,2,1)))
    tmp1_bar_ab = mp.ampf * tmp1_ab 

    tmp2 = numpy.einsum("iab, iab -> ", tmp1_bar_bb, numpy.einsum("Lia, Lb -> iab", qov_b, qvv_b[:,0,:]))
    tmp2 += numpy.einsum("iab, iab -> ", tmp1_bar_ab, numpy.einsum("Lia, Lb -> iab", qov_a, qvv_b[:,0,:]))

    ea_obmp2 = eV*(-mo_energy[1][L] - 1.*tmp2)
    ipea.append(ea_obmp2)

    return ipea

def get_nocc(mp):
    if mp._nocc is not None: return mp._nocc
    nocca = numpy.count_nonzero(mp.mo_occ[0] > 0)
    noccb = numpy.count_nonzero(mp.mo_occ[1] > 0)
    return nocca, noccb

def get_nmo(mp):
    if mp._nmo is not None: return mp._nmo
    nmoa = mp.mo_occ[0].size
    nmob = mp.mo_occ[1].size
    return nmoa, nmob

class UB2PLYPDFUOBMP2(DFOBMP2):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        # Bổ sung các cờ kiểm soát luồng chạy
        self.use_embed = False  # Nếu False: Chạy OBDH thông thường trên toàn hệ
        self.use_cl = False     # Bật/tắt Concentric Localization 
        self.active_atoms = []
        self.n_shells = 1
        self.mu = 1e6
        self.alphaa = (0.5, 0.5)
        self.niter = 50
        self.thresh = 1e-6
        self.second_order = True
        self.eval_IPEA = True
        self.cos = 1.
        self.css = 1.
        self._nocc = None
        self._nmo = None

    get_nocc = get_nocc
    get_nmo = get_nmo
    make_S2 = make_S2
    make_amp = make_amp
    make_IPEA = make_IPEA

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        if self.use_embed:
            self.ene_tot, self.e_ref = embed_kernel(self)
        else:
            self.ene_tot, self.e_dft, self.gamma = self.standard_kernel()
        return self.ene_tot

    def standard_kernel(self):
        log = logger.new_logger(self, self.verbose)
        print('\n' + '='*70)
        print('RUNNING STANDARD UOBMP2 (NO EMBEDDING)')
        print('='*70)
        
        xc_code = f"{self.alphaa[0]}*HF + {1-self.alphaa[0]}*B88, {1-self.alphaa[1]}*LYP"
        
        mf_std = copy.copy(self._scf)
        mf_std.mo_coeff = (self._scf.mo_coeff[0].copy(), self._scf.mo_coeff[1].copy())
        mf_std.mo_energy = (self._scf.mo_energy[0].copy(), self._scf.mo_energy[1].copy())
        mf_std.mo_occ = (self._scf.mo_occ[0].copy(), self._scf.mo_occ[1].copy())
        
        # Gọi obmp2_iter với v_emb = None (hoặc [0, 0])
        e_tot, e_dft, gamma = obmp2_iter(self, self.mol, mf_std, xc_code, v_emb=None, niter=self.niter)
        
        print("-" * 60)
        print(f"Total Standard UOBMP2 Energy = {e_tot:.8f} Eh")
        
        dip_mom = numpy.linalg.norm(scf.hf.dip_moment(self.mol, gamma, unit='Debye'))
        print(f"Norm of Dipole Moment        = {dip_mom}")
        print("=" * 60)
        
        return e_tot, e_dft, gamma

OBMP2 = UB2PLYPDFUOBMP2