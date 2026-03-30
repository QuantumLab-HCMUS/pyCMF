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
OB-MP2
'''

import time
from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import gto, df
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf import __config__
from pyscf.mp import mp2
from pycmf.OBMP import DFOBMP2, OBMP2_slow, _ChemistsERIs
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor
from pyscf.tools import cubegen
##
from pyscf import dft, scf, lo

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


import time
import numpy as np
import scipy.linalg
from pyscf import lib, dft, scf
from pyscf.lib import logger

def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=True,
           verbose=logger.NOTE, alphaa=(0.5, 0.5)):

    # ----------------------------------------------------------------------
    # 1. INITIALIZATION & SETUP
    # ----------------------------------------------------------------------
    log = logger.new_logger(mp, verbose)
    t0 = (time.process_time(), time.time())
    
    mol = mp.mol
    if hasattr(mp, 'alphaa'):
        alphaa = mp.alphaa
    
    # Overlap matrix & Orthogonalization matrix
    S = mp._scf.get_ovlp()
    A = scipy.linalg.fractional_matrix_power(S, -0.5)
    mu= mp.mu
    # ----------------------------------------------------------------------
    # 2. FULL SYSTEM DFT REFERENCE
    # ----------------------------------------------------------------------
    print('\n' + '='*70)
    print('UOBMP2-IN-DFT EMBEDDING WITH SPADE PARTITIONING')
    print('='*70)
    print('\n--- STEP 1: Running Full System DFT ---')
    
    #---- Step 1: Full DFT
    xc_code = f"{alphaa[0]}*HF + {1-alphaa[0]}*B88, {1-alphaa[1]}*LYP"
    logger.info(mp, f'Custom XC: {xc_code}')
    
    ks_full = run_full_dft(mol, xc_code)
    print(f"Full DFT Energy: {ks_full.e_tot:.8f} Eh")

    # Get H_Core Full (include all potential energy of nuclear)
    h_core_full = ks_full.get_hcore()

    #---- Step 2: SPADE
    C_occ_a = ks_full.mo_coeff[0][:, ks_full.mo_occ[0] > 0]
    C_occ_b = ks_full.mo_coeff[1][:, ks_full.mo_occ[1] > 0]
    atom_indices_A = mp.active_atoms # OH group

    print("\n --- Partitioning ---")
    C_A_a, C_B_a = spade_partition(mol, S, C_occ_a, atom_indices_A,True, "Alpha")
    C_A_b, C_B_b = spade_partition(mol, S, C_occ_b, atom_indices_A, False, "Beta")

    na_act = C_A_a.shape[1]
    nb_act = C_A_b.shape[1]
    
    gamma_A = (build_density_matrix(C_A_a), build_density_matrix(C_A_b))
    gamma_B = (build_density_matrix(C_B_a), build_density_matrix(C_B_b))

    #---- Step 3: Energy Components & Embedding Potential
    print("\n--- Constructing Potentials ---")
    h_core_A_iso = get_subsystem_hcore(mol, atom_indices_A)

    # e_dft_A_iso  = calculate_dft_energy_isolated(mol, xc_code, 
    #                                              gamma_A, h_core_A_iso, 
    #                                              atom_indices_A)
    #print(f"E_DFT[A] (Isolated Total): {e_dft_A_iso:.8f} Eh")

    # Embedding potential
    v_emb, P_B = build_embedding_potential(mol, xc_code, S, mu, 
                                           ks_full, gamma_B, gamma_A)
    # --- Step 4: Embedded UOBMP2 ---
    print("\n--- Running UOBMP2 in DFT Environment ---")    
    # E_WF_A_Internal (Isolated)
    e_wf_A_internal, e_dft_A_relax, gamma_uobmp2 = run_embed_uobmp2(
                                        mp, mol, xc_code,
                                        h_core_full, h_core_A_iso, v_emb,
                                        gamma_A, (na_act, nb_act), atom_indices_A,
                                        use_cl=mp.use_cl,       # True: CL truncation; False: full virtual space
                                        cl_n_shells=mp.n_shells, # số CL shells (chỉ dùng khi use_cl=True)
                                        cl_mu_threshold=1e5 # ngưỡng loại orbital bị đẩy bởi mu*P_B
                                    )

    gamma_uobmp2_a, gamma_uobmp2_b = gamma_uobmp2
    
    # Baseline correction: E_DFT[A_relax + B] - E_DFT[A_relax]
    gamma_relax = (gamma_uobmp2[0] + gamma_B[0], 
                   gamma_uobmp2[1] + gamma_B[1])

    e_dft_full_relax = ks_full.energy_tot(dm = gamma_relax)
    e_baseline = e_dft_full_relax - e_dft_A_relax

    # Relaxation correction
    v_emb_np_a = v_emb[0] - mp.mu * P_B[0]
    v_emb_np_b = v_emb[1] - mp.mu * P_B[1]

    #e_relax = np.einsum('ij,ji', gamma_uobmp2_a - gamma_A[0], v_emb_np_a)+ \
    #          np.einsum('ij,ji', gamma_uobmp2_b - gamma_A[1], v_emb_np_b)

    # Orthogonality correction
    e_ortho = mp.mu * (np.einsum('ij,ji', gamma_uobmp2_a, P_B[0]) + \
                          np.einsum('ij,ji', gamma_uobmp2_b, P_B[1]))

    #e_final = e_wf_A_internal + e_baseline + e_relax + e_ortho
    e_final = e_wf_A_internal + e_baseline + e_ortho

    print("-" * 60)
    print(f"E_WF[A] (Internal, Recalculated): {e_wf_A_internal:.8f}")
    print(f"Baseline (Full - Iso)           : {e_baseline:.8f}")
    #print(f"Relaxation Correction           : {e_relax:.8f}")
    print(f"Orthogonality Correction        : {e_ortho:.8f}")
    print("-" * 60)
    print(f"Total UOBMP2-in-DFT Energy        : {e_final:.8f} Eh")
    print(f"Ref DFT Energy                  : {ks_full.e_tot:.8f} Eh")
    print(f"Difference (Gain)               : {(e_final - ks_full.e_tot)*1e6:.2f} uEh")
    print("=" * 60)

    dm_total_a = gamma_uobmp2[0] + gamma_B[0]
    dm_total_b = gamma_uobmp2[1] + gamma_B[1]
    dm_total_final = (dm_total_a, dm_total_b)
    dip_mom = np.linalg.norm(make_dipole(mol, dm_total_final))
    print(f"Norm of Dipole Moment       : {(dip_mom)}")
    print("=" * 60)
    return e_final, ks_full.e_tot
import numpy as np

def dot(A, B):
    return np.einsum('ij, ij', A, B)


def run_full_dft(mol, xc):
    ks = dft.UKS(mol)
    ks.xc = xc
    ks.verbose = 0
    ks = ks.density_fit() # Tăng tốc
    ks.kernel()
    return ks


def spade_partition(mol, S, C_occ, atom_indices_A, plot=False, label="Alpha"):
    from scipy.linalg import fractional_matrix_power
    import numpy as np
    
    """Bước 2: Phân chia orbital bằng SPADE (SVD)."""
    # Chuyển sang cơ sở trực giao
    S_half = fractional_matrix_power(S, 0.5)
    C_bar = S_half @ C_occ
    
    # Lấy chỉ số AO của nguyên tử A
    ao_indices_A = []
    ao_slices = mol.aoslice_by_atom()
    for atom_idx in atom_indices_A:
        start, end = ao_slices[atom_idx][2], ao_slices[atom_idx][3]
        ao_indices_A.extend(range(start, end))
        
    C_bar_A = C_bar[ao_indices_A, :]
    
    # SVD
    # sigma là mảng các giá trị từ lớn đến nhỏ (thường từ 1.0 xuống 0.0)
    U, sigma, Vh = np.linalg.svd(C_bar_A, full_matrices=False)
    V = Vh.T
    C_spade = C_occ @ V
    
    # --- Xử lý Gap và chọn Active Space ---
    num_A_orbs = 0
    gap_val = 0.0
    
    # Trường hợp 1: Full System (Tất cả sigma ~ 1.0)
    if np.min(sigma) > 0.99:
        if plot: print(f"   [SPADE-{label}] Detected Full System (All sigma ~ 1.0). Selecting ALL orbitals.")
        num_A_orbs = C_occ.shape[1] # Lấy toàn bộ số cột (NOCC)
        
    # Trường hợp 2: Empty System (Tất cả sigma ~ 0.0)
    elif np.max(sigma) < 0.01:
        if plot: print(f"   [SPADE-{label}] Detected Empty System. Selecting 0 orbitals.")
        num_A_orbs = 0
        
    # Trường hợp 3: Hệ thống con bình thường (Có Gap)
    else:
        if len(sigma) > 1:
            gaps = sigma[:-1] - sigma[1:]
            gap_idx = np.argmax(gaps)
            num_A_orbs = gap_idx + 1
            gap_val = gaps[gap_idx]
        else:
            num_A_orbs = len(sigma)
            gap_val = 0.0 # Không có gap nếu chỉ có 1 orbital
            
        if plot:
            print(f"   [SPADE-{label}] Active orbitals: {num_A_orbs} (Gap: {gap_val:.4f})")

    C_A = C_spade[:, :num_A_orbs]
    C_B = C_spade[:, num_A_orbs:]
    
    # [FIX] Đã xóa dòng raise ValueError ở đây
    return C_A, C_B


def build_density_matrix(C_occ):
    return C_occ @ C_occ.T


def get_subsystem_hcore(mol, active_atoms):
    """Lấy H_core chỉ chứa hạt nhân của hệ A (Isolated Hamiltonian)."""
    t = mol.intor('int1e_kin')
    v_nuc = np.zeros_like(t)
    # Chỉ cộng thế hạt nhân của các nguyên tử trong A
    for i in active_atoms:
        with mol.with_rinv_origin(mol.atom_coord(i)):
            v_nuc += -mol.atom_charge(i) * mol.intor('int1e_rinv')
    return t + v_nuc


def calculate_dft_energy_isolated(mol, xc, gamma_A_tuple, h_core_A, 
                                  active_atoms):
    """Tính năng lượng DFT của hệ A cô lập (Baseline)."""

    mf_tmp = dft.UKS(mol)
    mf_tmp.xc = xc
    mf_tmp.verbose = 0
    mf_tmp = mf_tmp.density_fit()
    
    e_elec, _ = mf_tmp.energy_elec([gamma_A_tuple[0], gamma_A_tuple[1]], h1e=h_core_A)
    
    #---- Nuclear Repulsion of Subsystem A
    e_nuc_A = 0.0
    coords = mol.atom_coords()
    charges = mol.atom_charges()
    for i in range(len(active_atoms)):
        for j in range(i + 1, len(active_atoms)):
            at_i = active_atoms[i]
            at_j = active_atoms[j]
            dist = np.linalg.norm(coords[at_i] - coords[at_j])
            e_nuc_A += (charges[at_i] * charges[at_j]) / dist
            
    return e_elec + e_nuc_A


def build_embedding_potential(mol, xc_code, S, mu, mf_full, 
                              gamma_B_tuple, gamma_A_tuple):
    """Xây dựng v_emb = v_eff[A+B] - v_eff[A] + mu*P_B."""
    dm_full = mf_full.make_rdm1()
    dm_A = [gamma_A_tuple[0], gamma_A_tuple[1]]
    
    # V_eff (Total) từ Full DFT
    veff_full = mf_full.get_veff(mol, dm_full)
    
    # V_eff (A) từ mật độ A (dùng DFT operator của cả hệ)
    mf_tmp = dft.UKS(mol)
    mf_tmp.xc = xc_code
    mf_tmp.verbose = 0
    mf_tmp = mf_tmp.density_fit()
    veff_A = mf_tmp.get_veff(mol, dm_A)
    
    # Projection Operator
    P_B_a = S @ gamma_B_tuple[0] @ S
    P_B_b = S @ gamma_B_tuple[1] @ S
    
    # v_emb = v_eff[Total] - v_eff[A] + mu * P_B
    v_emb_a = veff_full[0] - veff_A[0] + mu * P_B_a
    v_emb_b = veff_full[1] - veff_A[1] + mu * P_B_b
    
    return [v_emb_a, v_emb_b], [P_B_a, P_B_b]


def run_embed_uobmp2(mp, mol, xc, h_core_full, h_core_A_iso, v_emb,
                     gamma_init, num_active_orbs, atom_indices_A,
                     use_cl=False, cl_n_shells=1, cl_mu_threshold=1e5):
    """
    Tham số:
        use_cl          : True  → dùng Concentric Localization để thu gọn virtual space
                          False → dùng trực tiếp mo_coeff từ UHF có v_emb (hành vi gốc)
        cl_n_shells     : số CL shells (mặc định 1, tương đương "double-zeta CL")
        cl_mu_threshold : ngưỡng loại orbital bị đẩy bởi mu*P_B (mặc định 1e5)
    """

    print(f"   [Embedded UOBMP2] Initializing UHF with Embedding Potential...")

    # Adapt the number of electrons of subsystem A
    na, nb      = num_active_orbs
    mol.nelectron = na + nb
    mol.spin    = na - nb

    # Make a UHF shell
    mf_emb = scf.UHF(mol)
    mf_emb.verbose = 0

    # Original get_veff module of mf_emb
    original_get_veff   = mf_emb.get_veff

    # Override get_veff để thêm Embedding Potential
    def get_veff_emb(mol, dm, dm_last=0, vhf_last=0):
        veff = original_get_veff(mol, dm, dm_last, vhf_last)
        return np.array([veff[0] + v_emb[0], veff[1] + v_emb[1]])

    mf_emb.get_veff = get_veff_emb

    # Keep H_core as full core hamiltonian for sth
    mf_emb.get_hcore = lambda *args: h_core_full

    # Run UHF
    try:
        mf_emb.kernel(dm0=gamma_init)
    except Exception as e:
        print(f"   [Warning] UHF kernel failed: {e}. Trying without dm0...")
        mf_emb.kernel()

    if not mf_emb.converged:
        print("   [Warning] Embedded UHF did not converge!")

    print(f"   [Embedded UOBMP2] UHF Reference Energy: {mf_emb.e_tot:.8f}")

    # =========================================================================
    # [LỰA CHỌN: CL TRUNCATION hoặc dùng thẳng MO từ UHF]
    # =========================================================================
    if use_cl:
        print(f"   [Embedded UOBMP2] Performing Concentric Localization "
              f"(n_shells={cl_n_shells}) to truncate virtual space...")
        from .CL_embed import concentric_localization
        import scipy.linalg as la

        # Lấy danh sách AO indices từ atom_indices_A
        active_aos = []
        aoslice = mol.aoslice_by_atom()
        for atom_id in atom_indices_A:
            p0, p1 = aoslice[atom_id][2], aoslice[atom_id][3]
            active_aos.extend(range(p0, p1))

        S_mat = mf_emb.get_ovlp()
        F_mat = mf_emb.get_fock()   # (F_a, F_b)

        new_mo_coeff  = []
        new_mo_energy = []
        new_mo_occ    = []

        for s in [0, 1]:    # Alpha (0) và Beta (1)
            C_s   = mf_emb.mo_coeff[s]
            occ_s = mf_emb.mo_occ[s]
            eps_s = mf_emb.mo_energy[s]
            F_s   = F_mat[s]

            # 1. Tách Occupied
            idx_occ   = occ_s > 0
            C_occ_A   = C_s[:, idx_occ]
            eps_occ_A = eps_s[idx_occ]

            # 2. Tách Virtuals thực sự (loại orbital bị đẩy bởi mu*P_B)
            idx_vir_eff = (occ_s == 0) & (eps_s < cl_mu_threshold)
            C_vir_eff   = C_s[:, idx_vir_eff]

            # 3. CL truncation (verbose=True tự in shell sizes + max shells)
            C_vir_CL = concentric_localization(
                C_vir_eff, S_mat, F_s, active_aos,
                n_shells=cl_n_shells, verbose=True)

            # 4. Pseudo-canonicalize: chéo hóa Fock trong không gian ảo CL
            F_vir = C_vir_CL.T.conj() @ F_s @ C_vir_CL
            evals_vir, evecs_vir = la.eigh(F_vir)
            C_vir_CL_canon = C_vir_CL @ evecs_vir

            # 5. Ghép lại Occ + Vir
            C_new_s   = np.hstack([C_occ_A, C_vir_CL_canon])
            eps_new_s = np.concatenate([eps_occ_A, evals_vir])
            occ_new_s = np.concatenate([
                np.ones(C_occ_A.shape[1]),
                np.zeros(C_vir_CL_canon.shape[1])
            ])

            new_mo_coeff.append(C_new_s)
            new_mo_energy.append(eps_new_s)
            new_mo_occ.append(occ_new_s)

        # Cập nhật mf_emb và mp với không gian CL đã thu gọn
        mf_emb.mo_coeff  = (new_mo_coeff[0],  new_mo_coeff[1])
        mf_emb.mo_energy = (new_mo_energy[0], new_mo_energy[1])
        mf_emb.mo_occ    = (new_mo_occ[0],    new_mo_occ[1])

        nmo_new = new_mo_coeff[0].shape[1]
        mp.mo_coeff  = mf_emb.mo_coeff
        mp.mo_occ    = mf_emb.mo_occ
        mp.mo_energy = mf_emb.mo_energy
        mp._nmo      = (nmo_new, nmo_new)
        mp.nocc      = (np.count_nonzero(mf_emb.mo_occ[0]),
                        np.count_nonzero(mf_emb.mo_occ[1]))

        nmo_a = mf_emb.mo_coeff[0].shape[1]
        nmo_b = mf_emb.mo_coeff[1].shape[1]
        print(f"   [Embedded UOBMP2] CL truncation done. "
              f"NMO alpha={nmo_a}, beta={nmo_b}")

    # Run UOBMP2
    print(f"   [Embedded UOBMP2] Running UOBMP2...")
    e_tot, e_dft, gamma_uobmp2 = obmp2_iter(mp, mol, mf_emb, xc, v_emb)

    # --- Step 5: Final Energy Correction 
    print("\n--- Final Energy Correction ---")

    gamma_uobmp2_a, gamma_uobmp2_b = gamma_uobmp2

    # 1. 1-electron Energy (Isolated)
    e_1e_iso = np.einsum('ij,ji', h_core_A_iso, 
                         gamma_uobmp2_a + gamma_uobmp2_b)
    
    # 2. 2-electron Energy (Coulomb + HF Exchange + XC)
    # Dùng DFT helper để tính chính xác phần này từ Gamma_CC
    mf_tmp = dft.UKS(mol)
    mf_tmp.xc = xc
    mf_tmp.verbose = 0
    mf_tmp = mf_tmp.density_fit()

    e_elec_meanfield, _ = mf_tmp.energy_elec([gamma_uobmp2_a, gamma_uobmp2_b], 
                                             h1e=h_core_A_iso)

    coords = mol.atom_coords()
    charges = mol.atom_charges()
    e_nuc_A = 0.0
    for i in range(len(atom_indices_A)):
        for j in range(i + 1, len(atom_indices_A)):
            at_i = atom_indices_A[i]
            at_j = atom_indices_A[j]
            dist = np.linalg.norm(coords[at_i] - coords[at_j])
            e_nuc_A += (charges[at_i] * charges[at_j]) / dist

    #return e_elec_meanfield + e_nuc_A + e_corr, gamma_uobmp2
    return e_tot, e_dft, gamma_uobmp2


def obmp2_iter(mp, mol, mf_emb, xc_code, v_emb, niter=1000):

    nmoa = mf_emb.mo_coeff[0].shape[1] 
    nmob = mf_emb.mo_coeff[1].shape[1]
    

    nocca = numpy.count_nonzero(mf_emb.mo_occ[0] > 0)
    noccb = numpy.count_nonzero(mf_emb.mo_occ[1] > 0)
    

    dm = mf_emb.make_rdm1(mf_emb.mo_coeff, mf_emb.mo_occ)
    s1e = mf_emb.get_ovlp(mol)
    h1e = mf_emb.get_hcore(mol)
    vhf = mf_emb.get_veff(mol, dm)
    nuc = mf_emb.energy_nuc()
    
    A = scipy.linalg.fractional_matrix_power(s1e, -0.5)
    
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

    for it in range(niter):
        h1ao = mf_emb.get_hcore(mol)
        h1mo_a = numpy.matmul(mf_emb.mo_coeff[0].T, numpy.matmul(h1ao, mf_emb.mo_coeff[0]))
        h1mo_b = numpy.matmul(mf_emb.mo_coeff[1].T, numpy.matmul(h1ao, mf_emb.mo_coeff[1]))
        
        fock_hfa = h1mo_a
        fock_hfb = h1mo_b

        # ==============================================================================
        # [QUAN TRỌNG] ĐỒNG BỘ HÓA mp VỚI mf_emb
        # Để mp biết được orbital và năng lượng hiện tại của vòng lặp embedding
        # ==============================================================================
        mp.mo_coeff = mf_emb.mo_coeff
        mp.mo_occ = mf_emb.mo_occ
        mp.mo_energy = mf_emb.mo_energy
        # Cập nhật _scf trong mp để các hàm con dùng đúng đối tượng tích phân
        mp._scf = mf_emb 
        # Cập nhật số lượng orbital để get_nmo() hoạt động đúng
        mp._nocc = (nocca, noccb)
        mp._nmo = (nmoa, nmob)

        # [FIX] Truyền 'mp' (đã đồng bộ) vào make_veff thay vì 'mf_emb'
        veffa, veffb, c0 = make_veff(mp) 
        
        fock_hfa += veffa
        fock_hfb += veffb

        fock_uobmp2_a = numpy.zeros((nmoa,nmoa), dtype=fock_hfa.dtype)
        fock_uobmp2_b = numpy.zeros((nmob,nmob), dtype=fock_hfb.dtype)

        fock_uobmp2_a += fock_hfa
        fock_uobmp2_b += fock_hfb

        # ... (Đoạn tính ene_hf giữ nguyên) ...
        ene_hf = 0.
        for i in range(nocca):
            ene_hf += fock_uobmp2_a[i,i]
        for i in range(noccb):
            ene_hf += fock_uobmp2_b[i,i]
        c0 *= 0.5
        ene_hf += c0

        # DFT part
        vxc = ks.get_veff(mol, dm)
        fock_dft = ks.get_fock(h1e, s1e, vxc, dm, diis_start_cycle=it) + v_emb
        fock_dft_a = numpy.matmul(mf_emb.mo_coeff[0].T, numpy.matmul(fock_dft[0], mf_emb.mo_coeff[0]))
        fock_dft_b = numpy.matmul(mf_emb.mo_coeff[1].T, numpy.matmul(fock_dft[1], mf_emb.mo_coeff[1]))
        ene_dft = ks.energy_elec(dm, h1e, vxc)[0] + nuc

        # [FIX] Gọi phương thức từ mf_emb
        vhf = mf_emb.get_veff(mol, dm)
        fock_hf_pyscf = mf_emb.get_fock(h1e, s1e, vhf, dm, diis_start_cycle=it) # check tham số diis
        
        fock_hf_pyscf_a = numpy.matmul(mf_emb.mo_coeff[0].T, numpy.matmul(fock_hf_pyscf[0], mf_emb.mo_coeff[0]))
        fock_hf_pyscf_b = numpy.matmul(mf_emb.mo_coeff[1].T, numpy.matmul(fock_hf_pyscf[1], mf_emb.mo_coeff[1]))
        
        # [FIX] Gọi phương thức từ mf_emb
        e_elec_hfpyscf = mf_emb.energy_elec(dm, h1e, vhf)[0]
        ene_hfpyscf = e_elec_hfpyscf + nuc

        #---- MP1 amplitude 
        # [FIX] Truyền 'mp' vào make_amp
        tmp1, tmp1_bar = make_amp(mp) 
        tmp1_aa, tmp1_bb, tmp1_ab, tmp1_ba = tmp1
        tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab, tmp1_bar_ba = tmp1_bar

        if mp.second_order:
            mp.ampf = 1.0

        tmp1_bar_aa *= mp.ampf
        tmp1_bar_bb *= mp.ampf
        tmp1_bar_ab *= mp.ampf
        tmp1_bar_ba *= mp.ampf

        # [FIX] Truyền 'mp' vào first_BCH
        c0, c1_a, c1_b = first_BCH(mp, fock_hfa, fock_hfb, tmp1_bar, c0)
        
        fock_uobmp2_a += 0.5 * (c1_a + c1_a.T)
        fock_uobmp2_b += 0.5 * (c1_b + c1_b.T)  

        if mp.second_order:
            # [FIX] Truyền 'mp' vào second_BCH
            c0, c1_a, c1_b = second_BCH(mp, fock_uobmp2_a, 
                                        fock_uobmp2_b, fock_hfa, 
                                        fock_hfb, tmp1, tmp1_bar, c0)
            
            fock_uobmp2_a += 0.5 * (c1_a + c1_a.T)
            fock_uobmp2_b += 0.5 * (c1_b + c1_b.T) 

        ene = c0
        for i in range(nocca):
            ene += 1. * fock_uobmp2_a[i,i]
        for i in range(noccb):
            ene += 1. * fock_uobmp2_b[i,i]
        
        ene_uobmp2 = ene + nuc

        e_tot = (ene_dft) + (ene_uobmp2 - ene_hfpyscf) * mp.alphaa[1]
        e_corr = (ene_uobmp2 - ene_hfpyscf) * mp.alphaa[1]


        fock_udftobmp2_a = (fock_dft_a) + \
            (fock_uobmp2_a - fock_hf_pyscf_a) * mp.alphaa[1] 
        fock_udftobmp2_b = (fock_dft_b) + \
            (fock_uobmp2_b - fock_hf_pyscf_b) * mp.alphaa[1] 

        print()

        de = abs(e_corr - ene_old)
        ene_old = e_corr
        
        print(f"Iter {it}: E_corr={e_corr:.8f}, dE={de:.8e}")

        # --- DIIS & Update Orbital ---
        # Tính Fock tổng hợp trong không gian MO
        F_eff_mo_a =  fock_udftobmp2_a 
        F_eff_mo_b =  fock_udftobmp2_b 
        
        # Chuyển về AO để chéo hóa
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

        # Build DIIS Residual
        diis_r_a = (err_a_mo + err_b_mo + 50*err_ab_mo + 50*err_ba_mo)
        diis_r_a = diis_r_a.real
        #diis_r_a = A.real.dot(err_a_mo).dot(A.real) + A.real.dot(err_b_mo).dot(A.real) + A.real.dot(err_ab_mo).dot(A.real) + A.real.dot(err_ba_mo).dot(A.real) 

        # Append trial & residual vectors to lists
        F_list_a.append(F_a)
        DIIS_RESID_a.append(diis_r_a) 

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

        # Build DIIS Residual
        err_ab_mo_b = mp.mo_coeff[1].T @ err_ab_ao @ mp.mo_coeff[1]
        err_ba_mo_b = mp.mo_coeff[1].T @ err_ba_ao @ mp.mo_coeff[1]
        err_b_mo_b  = mp.mo_coeff[1].T @ err_b_ao  @ mp.mo_coeff[1]
        err_a_mo_b  = mp.mo_coeff[1].T @ err_a_ao  @ mp.mo_coeff[1]
        diis_r_b = (1*err_a_mo_b + 1*err_b_mo_b + 50*err_ab_mo_b + 50*err_ba_mo_b)
        diis_r_b = diis_r_b.real

        #diis_r_b = A.real.dot(err_a_mo).dot(A.real) + A.real.dot(err_b_mo).dot(A.real) + A.real.dot(err_ab_mo).dot(A.real) + A.real.dot(err_ba_mo).dot(A.real)
        F_list_b.append(F_b)
        DIIS_RESID_b.append(diis_r_b)
        dRMS = numpy.mean(diis_r_b**2)**0.5

        if it >= 2:
        # Build B matrix
            B_dim_b = len(F_list_b) + 1
            B_b = numpy.empty((B_dim_b, B_dim_b))
            B_b[-1, :] = -1
            B_b[:, -1] = -1
            B_b[-1, -1] = 0
            for i in range(len(F_list_b)):
                for j in range(len(F_list_b)):
                    B_b[i, j] = numpy.einsum('ij,ij->', DIIS_RESID_b[i], DIIS_RESID_b[j], optimize=True)


            # Build RHS of Pulay equation 
            rhs_b = numpy.zeros((B_dim_b))
            rhs_b[-1] = -1
            
            # Solve Pulay equation for c_i's with NumPy
            coeff_b = numpy.linalg.solve(B_b, rhs_b)
            
            # Build DIIS Fock matrix
            F_b = numpy.zeros_like(F_b)
            for x in range(coeff_b.shape[0] - 1):
                F_b += coeff_b[x] * F_list_b[x]
        

        # Diagonalize Fock đã DIIS-extrapolate trong MO space
        # F_a/F_b hiện đang ở AO space (nao×nao) — cần project về MO space trước
        F_a_mo = mf_emb.mo_coeff[0].T @ F_a @ mf_emb.mo_coeff[0]  # (nmoa, nmoa)
        F_b_mo = mf_emb.mo_coeff[1].T @ F_b @ mf_emb.mo_coeff[1]  # (nmob, nmob)

        eps_new_a, C_rot_a = scipy.linalg.eigh(F_a_mo)
        eps_new_b, C_rot_b = scipy.linalg.eigh(F_b_mo)

        # Xoay MO coefficients trong AO basis: C_new = C_old @ C_rot
        mo_coeff_new_a = mf_emb.mo_coeff[0] @ C_rot_a
        mo_coeff_new_b = mf_emb.mo_coeff[1] @ C_rot_b

        # Gán lại bằng tuple mới — tránh lỗi "tuple object does not support item assignment"
        mf_emb.mo_coeff  = (mo_coeff_new_a, mo_coeff_new_b)
        mf_emb.mo_energy = (eps_new_a,      eps_new_b)

        # Đồng bộ mp với trạng thái mới
        mp.mo_coeff  = mf_emb.mo_coeff
        mp.mo_energy = mf_emb.mo_energy

        if de <= mp.thresh:
            conv = True
            break
    
        dm = mf_emb.make_rdm1(mf_emb.mo_coeff, mf_emb.mo_occ)
        dm = lib.tag_array(dm, mo_coeff=mf_emb.mo_coeff, mo_occ=mf_emb.mo_occ)

    dm_total = mf_emb.make_rdm1(mf_emb.mo_coeff, mf_emb.mo_occ)
    
    # Trả về kết quả
    return e_tot, ene_dft, (dm_total[0], dm_total[1])


def make_dipole(mol, dm_embed):
    return scf.hf.dip_moment(mol, dm_embed, unit='Debye')


def make_S2(mp, tmp1_bar_ab):
    mo_coeff = mp.mo_coeff
    mo_occ   = mp._scf.mo_occ
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb

    Sao = mp._scf.get_ovlp()
    ss_ref, s = mp._scf.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                    mo_coeff[1][:,mo_occ[1]>0]), Sao)
    Sib_AB_ = numpy.matmul(mo_coeff[0].T, numpy.matmul(Sao, mo_coeff[1]))
    Sja_BA_ = numpy.matmul(mo_coeff[1].T, numpy.matmul(Sao, mo_coeff[0]))
    Sib_AB = Sib_AB_[:nocca,noccb:nmob]
    Sja_BA = Sja_BA_[:noccb,nocca:nmoa]
    tmp = lib.einsum("ja,iajb -> ib", Sja_BA, tmp1_bar_ab)
    ss_res = ss_ref - 0.5*numpy.sum(Sib_AB*tmp)
    ss_prj = ss_ref - 1.0*numpy.sum(Sib_AB*tmp)

    return ss_ref, ss_res, ss_prj

def make_veff(mp):
    nmo  = mp.nmo
    nocc = mp.nocc
    nocca, noccb = mp.get_nocc()
    mo_coeff = mp.mo_coeff
    mo_occ = mp.mo_occ
    naux = mp.with_df.get_naoaux()


    dm = mp._scf.make_rdm1(mo_coeff,mo_occ)
    veff_ao = mp._scf.get_veff(mp.mol, dm)

    veffa = numpy.matmul(mo_coeff[0].T,numpy.matmul(veff_ao[0], mo_coeff[0]))
    veffb = numpy.matmul(mo_coeff[1].T,numpy.matmul(veff_ao[1], mo_coeff[1]))

    c0_hf = 0.
    for i in range(nocc[0]):
        c0_hf -=  veffa[i,i]
    for i in range(nocc[1]):
        c0_hf -=  veffb[i,i]

    return veffa, veffb, c0_hf


def make_amp(mp):
    css = mp.css
    cos = mp.cos
    log = logger.new_logger(mp, verbose=5)

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    # print("", nocca, "", noccb)
    # print("", nmoa, "", nmob)
    # raise ValueError
    nvira, nvirb = nmoa-nocca, nmob-noccb
    mo_energy = mp.mo_energy
    mo_coeff  = mp.mo_coeff
    
    t0 = (time.process_time(), time.time())

    from pyscf.lib import current_memory

    import tracemalloc
    tracemalloc.start()

    

    for istep, qov_a in enumerate(mp.loop_ao2mo(mo_coeff[0], nocca, nmoa)):
        qov_a = qov_a
    for istep, qov_b in enumerate(mp.loop_ao2mo(mo_coeff[1], noccb, nmob)):
        qov_b = qov_b

    print(f"DEBUG qov_a shape = {qov_a.shape}  (expected (naux, {nocca}*{nmoa-nocca}) = (naux, {nocca*(nmoa-nocca)}))")
    print(f"DEBUG qov_b shape = {qov_b.shape}  (expected (naux, {noccb}*{nmob-noccb}) = (naux, {noccb*(nmob-noccb)}))")
    print("qov_ab memory: %.1f MiB" % current_memory()[0])


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

    print("t_mp1 memory: %.1f MiB" % current_memory()[0])
    
    tmp1 = (tmp1_aa, tmp1_bb, tmp1_ab, tmp1_ba)
    tmp1_bar = (tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab, tmp1_bar_ba)

    return tmp1, tmp1_bar

############################################ 
def first_BCH(mp, fock_hfa, fock_hfb, tmp1_bar, c0):
    
    log = logger.new_logger(mp, verbose=5)

    tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab, tmp1_bar_ba = tmp1_bar
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    # print("nocca shape =", nocca, "noccb shape =", noccb)
    # print("nmoa shape =", nmoa, "nmob shape =", nmob)
    nvira, nvirb = nmoa-nocca, nmob-noccb
    mo_energy = mp.mo_energy
    mo_coeff  = mp.mo_coeff
    naux = mp.with_df.get_naoaux()
    
    t0 = (time.process_time(), time.time())

    from pyscf.lib import current_memory

    import tracemalloc
    tracemalloc.start()
    

    ##########################################################

    t1 = log.timer('first BCH: integral transform', *t0)

    c1_a = numpy.zeros((nmoa,nmoa), dtype=fock_hfa.dtype)
    c1_b = numpy.zeros((nmob,nmob), dtype=fock_hfb.dtype)

    for istep, qov_b in enumerate(mp.loop_ao2mo(mo_coeff[1], noccb, nmob)):
        qov_b = qov_b

    for istep, qgv_a in enumerate(mp.loop_ao2mo_cgcv(mo_coeff[0], nocca, nmoa)):
        for i in range(nocca):
            c1_a[:,0:nocca] += 2. * lib.einsum("apb, ajb -> pj",numpy.dot(qgv_a.reshape(naux, nmoa, nvira)[:,:nocca,:nvira].reshape(naux, nocca*nvira)[:,i*nvira:(i+1)*nvira].T,qgv_a).reshape(nvira,nmoa,nvira),tmp1_bar_aa[i,:,:,:]) 
        for i in range(noccb):
            c1_a[:,0:nocca] += 2. * lib.einsum("apb, ajb -> pj",numpy.dot(qov_b[:,i*nvirb:(i+1)*nvirb].T,qgv_a).reshape(nvirb,nmoa,nvira),tmp1_bar_ba[i,:,:,:])
    
    print("qgv_a memory: %.1f MiB" % current_memory()[0])
    del(qgv_a)
    
    
    
    for istep, qov_a in enumerate(mp.loop_ao2mo(mo_coeff[0], nocca, nmoa)):
        for i in range(nocca):
            c0 -= 1.*lib.einsum("ajb, ajb -> ", numpy.dot(qov_a[:,i*nvira:(i+1)*nvira].T, qov_a).reshape(nvira, nocca, nvira), tmp1_bar_aa[i,:,:,:])
    
    for i in range(nocca):
        c0 -= 1.*lib.einsum("ajb, ajb -> ", numpy.dot(qov_a[:,i*nvira:(i+1)*nvira].T, qov_b).reshape(nvira, noccb, nvirb), tmp1_bar_ab[i,:,:,:])
    
    print("qov memory: %.1f MiB" % current_memory()[0])
    for istep, qgv_b in enumerate(mp.loop_ao2mo_cgcv(mo_coeff[1], noccb, nmob)):
        for i in range(noccb):
            c1_b[:,0:noccb] += 2. * lib.einsum("apb, ajb -> pj",numpy.dot(qgv_b.reshape(naux, nmob, nvirb)[:,:noccb,:nvirb].reshape(naux, noccb*nvirb)[:,i*nvirb:(i+1)*nvirb].T,qgv_b).reshape(nvirb,nmob,nvirb),tmp1_bar_bb[i,:,:,:]) 
        for i in range(nocca):
            c1_b[:,0:noccb] += 2. * lib.einsum("apb, ajb -> pj",numpy.dot(qov_a[:,i*nvira:(i+1)*nvira].T,qgv_b).reshape(nvira,nmob,nvirb),tmp1_bar_ab[i,:,:,:])  
    print("qgv memory: %.1f MiB" % current_memory()[0])
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

    print("qgv memory: %.1f MiB" % current_memory()[0])
    '''#c0 -= 1.*numpy.sum(h2mo_aa_ovov*tmp1_bar_aa)
    for i in range(nocca):
        c0 -= 1.*lib.einsum("ajb, ajb -> ", numpy.dot(qov_a[:,i*nvira:(i+1)*nvira].T, qov_a).reshape(nvira, nocca, nvira), tmp1_bar_aa[i,:,:,:])
    #c0 -= 1.*numpy.sum(h2mo_ab_ovov*tmp1_bar_ab)
    for i in range(nocca):
        c0 -= 1.*lib.einsum("ajb, ajb -> ", numpy.dot(qov_a[:,i*nvira:(i+1)*nvira].T, qov_b).reshape(nvira, noccb, nvirb), tmp1_bar_ab[i,:,:,:])
    #c0 -= 1.*numpy.sum(h2mo_ba_ovov*tmp1_bar_ba)
    for i in range(noccb):
        c0 -= 1.*lib.einsum("ajb, ajb -> ", numpy.dot(qov_b[:,i*nvirb:(i+1)*nvirb].T, qov_a).reshape(nvirb, nocca, nvira), tmp1_bar_ba[i,:,:,:])
    #c0 -= 1.*numpy.sum(h2mo_bb_ovov*tmp1_bar_bb)
    for i in range(noccb):
        c0 -= 1.*lib.einsum("ajb, ajb -> ", numpy.dot(qov_b[:,i*nvirb:(i+1)*nvirb].T, qov_b).reshape(nvirb, noccb, nvirb), tmp1_bar_bb[i,:,:,:])'''

    c1_a[:nocca,nocca:] += 2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_aa,fock_hfa[:nocca,nocca:])
    c1_a[:nocca,nocca:] += 2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_ba,fock_hfb[:noccb,noccb:])

    c1_b[:noccb,noccb:] += 2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_bb,fock_hfb[:noccb,noccb:])
    c1_b[:noccb,noccb:] += 2.*lib.einsum('ijkl,ij -> kl',tmp1_bar_ab,fock_hfa[:nocca,nocca:])



    '''#c1_a[:,:nocca] += 2.*lib.einsum('ijpl,ijkl -> pk',h2mo_aa_ovgv,tmp1_bar_aa)
    for i in range(nocca):
        c1_a[:,0:nocca] += 2. * lib.einsum("apb, ajb -> pj",numpy.dot(qov_a[:,i*nvira:(i+1)*nvira].T,qgv_a).reshape(nvira,nmoa,nvira),tmp1_bar_aa[i,:,:,:]) 
    #c1_a[:,:nocca] += 2.*lib.einsum('ijpl,ijkl -> pk',h2mo_ba_ovgv,tmp1_bar_ba) 
    for i in range(noccb):
        c1_a[:,0:nocca] += 2. * lib.einsum("apb, ajb -> pj",numpy.dot(qov_b[:,i*nvirb:(i+1)*nvirb].T,qgv_a).reshape(nvirb,nmoa,nvira),tmp1_bar_ba[i,:,:,:])

    #c1_a[:,nocca:nmoa] -= 2.*lib.einsum('ijkp,ijkl -> pl',h2mo_aa_ovog,tmp1_bar_aa)
    for i in range(nocca):
        c1_a[:,nocca:nmoa] -= 2.*lib.einsum("ajp, ajb -> pb", numpy.dot(qov_a[:,i*nvira:(i+1)*nvira].T,qog_a).reshape(nvira,nocca, nmoa),tmp1_bar_aa[i,:,:,:]) 
    #c1_a[:,nocca:nmoa] -= 2.*lib.einsum('ijkp,ijkl -> pl',h2mo_ba_ovog,tmp1_bar_ba) 
    for i in range(noccb):
        c1_a[:,nocca:nmoa] -= 2.*lib.einsum("ajp, ajb -> pb", numpy.dot(qov_b[:,i*nvirb:(i+1)*nvirb].T,qog_a).reshape(nvirb,nocca, nmoa),tmp1_bar_ba[i,:,:,:])


    #c1_b[:,:noccb] += 2.*lib.einsum('ijpl,ijkl -> pk',h2mo_bb_ovgv,tmp1_bar_bb)
    for i in range(noccb):
        c1_b[:,0:noccb] += 2. * lib.einsum("apb, ajb -> pj",numpy.dot(qov_b[:,i*nvirb:(i+1)*nvirb].T,qgv_b).reshape(nvirb,nmob,nvirb),tmp1_bar_bb[i,:,:,:]) 
    #c1_b[:,:noccb] += 2.*lib.einsum('ijpl,ijkl -> pk',h2mo_ab_ovgv,tmp1_bar_ab)
    for i in range(nocca):
        c1_b[:,0:noccb] += 2. * lib.einsum("apb, ajb -> pj",numpy.dot(qov_a[:,i*nvira:(i+1)*nvira].T,qgv_b).reshape(nvira,nmob,nvirb),tmp1_bar_ab[i,:,:,:])  

    #c1_b[:,noccb:nmob] -= 2.*lib.einsum('ijkp,ijkl -> pl',h2mo_bb_ovog,tmp1_bar_bb)
    for i in range(noccb):
        c1_b[:,noccb:nmob] -= 2.* lib.einsum("ajp, ajb -> pb", numpy.dot(qov_b[:,i*nvirb:(i+1)*nvirb].T,qog_b).reshape(nvirb,noccb, nmob),tmp1_bar_bb[i,:,:,:]) 
    #c1_b[:,noccb:nmob] -= 2.*lib.einsum('ijkp,ijkl -> pl',h2mo_ab_ovog,tmp1_bar_ab) 
    for i in range(nocca):
        c1_b[:,noccb:nmob] -= 2.* lib.einsum("ajp, ajb -> pb", numpy.dot(qov_a[:,i*nvira:(i+1)*nvira].T,qog_b).reshape(nvira,noccb, nmob),tmp1_bar_ab[i,:,:,:])''' 
    
    print("1st memory: %.1f MiB" % current_memory()[0])
    return c0, c1_a, c1_b


def second_BCH(mp, fock_a, fock_b, fock_hfa, fock_hfb, tmp1, tmp1_bar, c0):

    tmp1_aa, tmp1_bb, tmp1_ab, tmp1_ba = tmp1
    tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab, tmp1_bar_ba = tmp1_bar

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb

    c1_a = numpy.zeros((nmoa,nmoa), dtype=fock_hfa.dtype)
    c1_b = numpy.zeros((nmob,nmob), dtype=fock_hfb.dtype)
    
    #[1]
    y1_a = lib.einsum('ij,ijkl -> kl', fock_hfa[:nocca,nocca:], tmp1_bar_aa)#
    y1_a += lib.einsum('ij,ijkl -> kl', fock_hfb[:noccb,noccb:], tmp1_bar_ba)#
    c1_a[:nocca,nocca:] += lib.einsum('ijkl,kl -> ij', tmp1_bar_aa,y1_a)#
    c1_b[:noccb,noccb:] += lib.einsum('ijkl,kl -> ij', tmp1_bar_ba,y1_a)#
    
    y1_b = lib.einsum('ij,ijkl -> kl', fock_hfb[:noccb,noccb:], tmp1_bar_bb)#
    y1_b += lib.einsum('ij,ijkl -> kl', fock_hfa[:nocca,nocca:], tmp1_bar_ab)#
    c1_a[:nocca,nocca:] += lib.einsum('ijkl,kl -> ij', tmp1_bar_ab,y1_b)#
    c1_b[:noccb,noccb:] += lib.einsum('ijkl,kl -> ij', tmp1_bar_bb,y1_b)#

    #[2]
    y1_aa = lib.einsum('ac,kcjb -> kajb',fock_hfa[nocca:,nocca:],tmp1_bar_aa)#
    y1_ab = lib.einsum('ac,kcjb -> kajb',fock_hfa[nocca:,nocca:],tmp1_bar_ab)#
    y1_ba = lib.einsum('ac,kcjb -> kajb',fock_hfb[noccb:,noccb:],tmp1_bar_ba)#
    y1_bb = lib.einsum('ac,kcjb -> kajb',fock_hfb[noccb:,noccb:],tmp1_bar_bb)#
    c1_a[:nocca,:nocca] += lib.einsum('iajb,iakb -> jk', tmp1_aa, y1_aa)
    c1_a[:nocca,:nocca] += lib.einsum('iajb,iakb -> jk', tmp1_ba, y1_ba)
    c1_b[:noccb,:noccb] += lib.einsum('iajb,iakb -> jk', tmp1_bb, y1_bb)
    c1_b[:noccb,:noccb] += lib.einsum('iajb,iakb -> jk', tmp1_ab, y1_ab)
    
    c0 -= lib.einsum('ijkl,ijkl->', tmp1_aa,y1_aa) +\
        lib.einsum('ijkl,ijkl->', tmp1_bb,y1_bb)
    c0 -= lib.einsum('ijkl,ijkl->', tmp1_ab,y1_ab) +\
        lib.einsum('ijkl,ijkl->', tmp1_ba,y1_ba)

    #[3]
    y1_aa = lib.einsum('ac,kcjb -> kajb',fock_hfa[nocca:,nocca:],tmp1_bar_aa)#
    y1_ab = lib.einsum('ac,kcjb -> kajb',fock_hfa[nocca:,nocca:],tmp1_bar_ab)#
    y1_ba = lib.einsum('ac,kcjb -> kajb',fock_hfb[noccb:,noccb:],tmp1_bar_ba)#
    y1_bb = lib.einsum('ac,kcjb -> kajb',fock_hfb[noccb:,noccb:],tmp1_bar_bb)#
    c1_a[:nocca,:nocca] += lib.einsum('iajb,kajb -> ik', tmp1_aa, y1_aa)
    c1_a[:nocca,:nocca] += lib.einsum('iajb,kajb -> ik', tmp1_ab, y1_ab)
    c1_b[:noccb,:noccb] += lib.einsum('iajb,kajb -> ik', tmp1_bb, y1_bb)
    c1_b[:noccb,:noccb] += lib.einsum('iajb,kajb -> ik', tmp1_ba, y1_ba)
                            
    #[4]
    y1_aa = lib.einsum('ik,kalb -> ialb',fock_hfa[:nocca,:nocca],tmp1_bar_aa)#
    y1_ab = lib.einsum('ik,kalb -> ialb',fock_hfa[:nocca,:nocca],tmp1_bar_ab)#
    y1_ba = lib.einsum('ik,kalb -> ialb',fock_hfb[:noccb,:noccb],tmp1_bar_ba)#
    y1_bb = lib.einsum('ik,kalb -> ialb',fock_hfb[:noccb,:noccb],tmp1_bar_bb)#
    c1_a[:nocca,:nocca] -= lib.einsum('iajb,ialb -> jl', tmp1_aa, y1_aa)
    c1_a[:nocca,:nocca] -= lib.einsum('iajb,ialb -> jl', tmp1_ba, y1_ba)
    c1_b[:noccb,:noccb] -= lib.einsum('iajb,ialb -> jl', tmp1_bb, y1_bb)
    c1_b[:noccb,:noccb] -= lib.einsum('iajb,ialb -> jl', tmp1_ab, y1_ab)
    
    c0 += lib.einsum('ijkl,ijkl->', tmp1_aa,y1_aa) +\
        lib.einsum('ijkl,ijkl->', tmp1_bb,y1_bb)
    c0 += lib.einsum('ijkl,ijkl->', tmp1_ab,y1_ab) +\
        lib.einsum('ijkl,ijkl->', tmp1_ba,y1_ba)

    #[5]
    y1_a  = lib.einsum('iajb,kajb -> ik', tmp1_aa, tmp1_bar_aa)#
    y1_a += lib.einsum('iajb,kajb -> ik', tmp1_ab, tmp1_bar_ab)#
    c1_a[:,:nocca] -= lib.einsum('pk,ik -> pi', fock_hfa[:,:nocca], y1_a)

    y1_b  = lib.einsum('iajb,kajb -> ik', tmp1_bb, tmp1_bar_bb)#
    y1_b += lib.einsum('iajb,kajb -> ik', tmp1_ba, tmp1_bar_ba)#
    c1_b[:,:noccb] -= lib.einsum('pk,ik -> pi', fock_hfb[:,:noccb], y1_b)
    
    #[6]
    y1_aa = lib.einsum('ik,kajd -> iajd',fock_hfa[:nocca,:nocca],tmp1_bar_aa)#
    y1_ab = lib.einsum('ik,kajd -> iajd',fock_hfa[:nocca,:nocca],tmp1_bar_ab)#
    y1_ba = lib.einsum('ik,kajd -> iajd',fock_hfb[:noccb,:noccb],tmp1_bar_ba)#
    y1_bb = lib.einsum('ik,kajd -> iajd',fock_hfb[:noccb,:noccb],tmp1_bar_bb)#
    c1_a[nocca:,nocca:] += lib.einsum('iajb,iajd -> bd', tmp1_aa, y1_aa)
    c1_a[nocca:,nocca:] += lib.einsum('iajb,iajd -> bd', tmp1_ba, y1_ba)
    c1_b[noccb:,noccb:] += lib.einsum('iajb,iajd -> bd', tmp1_bb, y1_bb)
    c1_b[noccb:,noccb:] += lib.einsum('iajb,iajd -> bd', tmp1_ab, y1_ab)

    #[7]
    y1_aa = lib.einsum('ik,kajd -> iajd',fock_hfa[:nocca,:nocca],tmp1_bar_aa)#
    y1_ab = lib.einsum('ik,kajd -> iajd',fock_hfa[:nocca,:nocca],tmp1_bar_ab)#
    y1_ba = lib.einsum('ik,kajd -> iajd',fock_hfb[:noccb,:noccb],tmp1_bar_ba)#
    y1_bb = lib.einsum('ik,kajd -> iajd',fock_hfb[:noccb,:noccb],tmp1_bar_bb)#
    c1_a[nocca:,nocca:] += lib.einsum('iajb,icjb -> ac', tmp1_aa, y1_aa)
    c1_a[nocca:,nocca:] += lib.einsum('iajb,icjb -> ac', tmp1_ab, y1_ab)
    c1_b[noccb:,noccb:] += lib.einsum('iajb,icjb -> ac', tmp1_bb, y1_bb)
    c1_b[noccb:,noccb:] += lib.einsum('iajb,icjb -> ac', tmp1_ba, y1_ba)

    #[8]
    y1_aa = lib.einsum('ac,icjd -> iajd',fock_hfa[nocca:,nocca:],tmp1_bar_aa)#
    y1_ab = lib.einsum('ac,icjd -> iajd',fock_hfa[nocca:,nocca:],tmp1_bar_ab)#
    y1_ba = lib.einsum('ac,icjd -> iajd',fock_hfb[noccb:,noccb:],tmp1_bar_ba)#
    y1_bb = lib.einsum('ac,icjd -> iajd',fock_hfb[noccb:,noccb:],tmp1_bar_bb)#
    c1_a[nocca:,nocca:] -= lib.einsum('iajb,iajd -> bd', tmp1_aa, y1_aa)
    c1_a[nocca:,nocca:] -= lib.einsum('iajb,iajd -> bd', tmp1_ba, y1_ba)
    c1_b[noccb:,noccb:] -= lib.einsum('iajb,iajd -> bd', tmp1_bb, y1_bb)
    c1_b[noccb:,noccb:] -= lib.einsum('iajb,iajd -> bd', tmp1_ab, y1_ab)

    #[9]
    y1_a  = lib.einsum('iajb,icjb -> ac', tmp1_aa, tmp1_bar_aa)#
    y1_a += lib.einsum('iajb,icjb -> ac', tmp1_ab, tmp1_bar_ab)#
    c1_a[:,nocca:] -= lib.einsum('pa,ac -> pc', fock_hfa[:,nocca:], y1_a)
    y1_b  = lib.einsum('iajb,icjb -> ac', tmp1_bb, tmp1_bar_bb)#
    y1_b += lib.einsum('iajb,icjb -> ac', tmp1_ba, tmp1_bar_ba)#
    c1_b[:,noccb:] -= lib.einsum('pa,ac -> pc', fock_hfb[:,noccb:], y1_b)

    return c0, c1_a, c1_b    

def make_IPEA(mp):
    eV = 27.2114
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb
    mo_energy = mp.mo_energy
    mo_coeff  = mp.mo_coeff
    naux = mp.with_df.get_naoaux()

    ####### evaluating IP 

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

        tmp1_aa = mp.css * numpy.einsum("Lia, Lj -> iaj", qov_a, qoo_a[:,:,h])
        tmp1_aa = tmp1_aa/x_a

        tmp1_ba = mp.cos * numpy.einsum("Lia, Lj -> iaj", qov_b, qoo_a[:,:,h])
        tmp1_ba = tmp1_ba/x_b

        tmp1_bar_aa = mp.ampf * (tmp1_aa - numpy.transpose(tmp1_aa,(2,1,0)))
        tmp1_bar_ba = mp.ampf * tmp1_ba

        tmp2 = 0.
        tmp2 += numpy.einsum("iaj, iaj -> ", tmp1_bar_aa, numpy.einsum("Lia, Lj -> iaj", qov_a, qoo_a[:,:,h]))
        tmp2 += numpy.einsum("iaj, iaj -> ", tmp1_bar_ba, numpy.einsum("Lia, Lj -> iaj", qov_b, qoo_a[:,:,h]))

        ip_obmp2 = eV*(-mo_energy[0][h] + 1.*tmp2)
        ipea.append(ip_obmp2)
        logger.info(mp, "obmp2 orb energy %8.6f (eV) ip_obmp2 %8.6f (eV)", -eV*mo_energy[0][h], ip_obmp2)

    ###### evaluating EA ######
    
    #for L in range(noccb,min(noccb+10,nvirb)): 
    L = noccb
    
    x_b = mo_energy[1][:noccb,None,None] + mo_energy[1][None,L,None] - mo_energy[1][None,noccb:,None] - mo_energy[1][None,None,noccb:]  - mp.shift
    tmp1_bb = mp.css * numpy.einsum("Lia, Lb -> iab", qov_b, qvv_b[:,0,:])
    tmp1_bb = tmp1_bb/x_b

    x_a = mo_energy[0][:nocca,None,None] + mo_energy[1][None,L,None] - mo_energy[0][None,nocca:,None] - mo_energy[1][None,None,noccb:]  - mp.shift
    tmp1_ab = mp.cos * numpy.einsum("Lia, Lb -> iab", qov_a, qvv_b[:,0,:])
    tmp1_ab = tmp1_ab/x_a

    tmp1_bar_bb = mp.ampf * (tmp1_bb - numpy.transpose(tmp1_bb,(0,2,1)))
    tmp1_bar_ab = mp.ampf * tmp1_ab 

    tmp2 = 0.
    
    tmp2 += numpy.einsum("iab, iab -> ", tmp1_bar_bb, numpy.einsum("Lia, Lb -> iab", qov_b, qvv_b[:,0,:]))
    tmp2 += numpy.einsum("iab, iab -> ", tmp1_bar_ab, numpy.einsum("Lia, Lb -> iab", qov_a, qvv_b[:,0,:]))

    ea_obmp2 = eV*(-mo_energy[1][L] - 1.*tmp2)
    print(ea_obmp2)
    ipea.append(ea_obmp2)
    logger.info(mp, "obmp2 lumo %8.6f (eV) ea_obmp2 %8.6f (eV)", -eV*mo_energy[1][L], ea_obmp2)

    fname = 'ipea-'+str(mp.css)+'-'+str(mp.cos)+'.txt'
    with open(fname, 'w') as f:
        for i in range(len(ipea)):
            f.write(str(ipea[i])+'\n')

    return ipea

def int_transform_ss(eri_ao, mo_coeff):
    nao = mo_coeff.shape[0]
    nmo = mo_coeff.shape[1]
    eri_mo = numpy.dot(mo_coeff.T, eri_ao.reshape(nao,-1))
    eri_mo = numpy.dot(eri_mo.reshape(-1,nao), mo_coeff)
    eri_mo = eri_mo.reshape(nmo,nao,nao,nmo).transpose(1,0,3,2)
    eri_mo = numpy.dot(mo_coeff.T, eri_mo.reshape(nao,-1))
    eri_mo = numpy.dot(eri_mo.reshape(-1,nao), mo_coeff)
    eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
    

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
    frozen = mp.frozen
    if mp._nocc is not None:
        return mp._nocc
    elif frozen is None:
        nocca = numpy.count_nonzero(mp.mo_occ[0] > 0)
        noccb = numpy.count_nonzero(mp.mo_occ[1] > 0)
    elif isinstance(frozen, (int, numpy.integer)):
        nocca = numpy.count_nonzero(mp.mo_occ[0] > 0) - frozen
        noccb = numpy.count_nonzero(mp.mo_occ[1] > 0) - frozen
        #assert(nocca > 0 and noccb > 0)
    elif isinstance(frozen[0], (int, numpy.integer, list, numpy.ndarray)):
        if len(frozen) > 0 and isinstance(frozen[0], (int, numpy.integer)):
            # The same frozen orbital indices for alpha and beta orbitals
            frozen = [frozen, frozen]
        occidxa = mp.mo_occ[0] > 0
        occidxa[list(frozen[0])] = False
        occidxb = mp.mo_occ[1] > 0
        occidxb[list(frozen[1])] = False
        nocca = numpy.count_nonzero(occidxa)
        noccb = numpy.count_nonzero(occidxb)
    else:
        raise NotImplementedError
    return nocca, noccb

def get_nmo(mp):
    frozen = mp.frozen
    if mp._nmo is not None:
        return mp._nmo
    elif frozen is None:
        nmoa = mp.mo_occ[0].size
        nmob = mp.mo_occ[1].size
    elif isinstance(frozen, (int, numpy.integer)):
        nmoa = mp.mo_occ[0].size - frozen
        nmob = mp.mo_occ[1].size - frozen
    elif isinstance(frozen[0], (int, numpy.integer, list, numpy.ndarray)):
        if isinstance(frozen[0], (int, numpy.integer)):
            frozen = (frozen, frozen)
        nmoa = len(mp.mo_occ[0]) - len(set(frozen[0]))
        nmob = len(mp.mo_occ[1]) - len(set(frozen[1]))
    else:
        raise NotImplementedError
    return nmoa, nmob


def get_frozen_mask(mp):
    '''Get boolean mask for the unrestricted reference orbitals.

    In the returned boolean (mask) array of frozen orbital indices, the
    element is False if it corresonds to the frozen orbital.
    '''
    moidxa = numpy.ones(mp.mo_occ[0].size, dtype=bool)
    moidxb = numpy.ones(mp.mo_occ[1].size, dtype=bool)

    frozen = mp.frozen
    if mp._nmo is not None:
        moidxa[mp._nmo[0]:] = False
        moidxb[mp._nmo[1]:] = False
    elif frozen is None:
        pass
    elif isinstance(frozen, (int, numpy.integer)):
        moidxa[:frozen] = False
        moidxb[:frozen] = False
    elif isinstance(frozen[0], (int, numpy.integer, list, numpy.ndarray)):
        if isinstance(frozen[0], (int, numpy.integer)):
            frozen = (frozen, frozen)
        moidxa[list(frozen[0])] = False
        moidxb[list(frozen[1])] = False
    else:
        raise NotImplementedError
    return moidxa,moidxb

def mom_reorder(mp, mo_coeff):
    import copy
    mo_coeff_save = copy.copy(mo_coeff)
    #mo_energy_save = copy.copy(mo_energy)
    #mo_energy = mp.mo_energy
    mo_coeff = copy.copy(mo_coeff_save)
    print("before")
    print(mo_coeff[0][:,:4])
    ia, ib = mp.occ_exc
    aa, ab = mp.vir_exc
    #print("ia ", ia)
    #print(mo_coeff_save[0][:,ia])
    mo_coeff[0][:,ia] = mo_coeff_save[0][:,aa]
    #mo_energy[0][ia]  = mo_energy_save[0][aa]
    #print("test")
    #print(mo_coeff_save[0][:,ia])
    mo_coeff[0][:,aa] = mo_coeff_save[0][:,ia]
    #mo_energy[0][aa]  = mo_energy_save[0][ia]
    if (ib is not None) and (ab is not None):
        mo_coeff[1][:,ib] = mo_coeff_save[1][:,ab]
        mo_coeff[1][:,ab] = mo_coeff_save[1][:,ib]
    print("after")
    print(mo_coeff[0][:,:4])
    return mo_coeff #, mo_energy
        
def mom_select(mp, mo_coeff_init, mo_coeff_new):
    #print("old")
    #print(mo_coeff_init[0][:,:4])
    #print("new")
    #print(mo_coeff_new[0][:,:4])
    ovi = mp._scf.get_ovlp()
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    ia, ib = mp.occ_exc
    aa, ab = mp.vir_exc
    Oa = numpy.matmul(mo_coeff_init[0][:,0:nocca].T,
                    numpy.matmul(ovi,mo_coeff_new[0][:,:]))
    #print("Oa")
    #print(Oa)
    Pa = []
    for j in range(nmoa):
        tmp = 0.
        for i in range(nocca):
            tmp += Oa[i,j]
        Pa.append(abs(tmp))
        #print("Paj = ", Pa[j])
    max_el = max(Pa)
    indxa = 0
    for j in range(nmoa):
        if Pa[j] == max_el:
            indxa = j
    if (ib is not None) and (ab is not None):
        Ob = numpy.matmul(mo_coeff_init[1][:,0:noccb].T,
                        numpy.matmul(ovi,mo_coeff_new[1][:,:]))
        Pb = []
        for j in range(nmob):
            tmp = 0.
            for i in range(noccb):
                tmp += Ob[i,j]
            Pb.append(abs(tmp))
        max_el = max(Pa)
        indxb = 0
        for j in range(nmob):
            if Pa[j] == max_el:
                indxb = j
    else:
        indxb = None

    print("indxa = %d"%indxa, "Pa = %8.6f"%Pa[indxa])
    if indxb is not None:
        print("indxb = %d"%indxb, "Pb = %8.6f"%Pb[indxb])
    mp.vir_exc = [indxa, indxb]
    #mp.ib = indxb
    #return indxa, indxb

def make_rdm1(mp, use_t2=True, use_ao=False, **kwargs):
    '''One-particle density matrix

    Returns:
        A list of 2D ndarrays for alpha and beta spins
    '''
    mo_coeff = mp.mo_coeff
    mo_occ   = mp._scf.mo_occ
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb
    mo_ea, mo_eb = mp.mo_energy
    eia_a = mo_ea[:nocca,None] - mo_ea[None,nocca:]
    eia_b = mo_eb[:noccb,None] - mo_eb[None,noccb:]

    if not use_t2:
        mo_a = mo_coeff[0]
        mo_b = mo_coeff[1]
        dm_a = numpy.dot(mo_a*mo_occ[0], mo_a.conj().T)
        dm_b = numpy.dot(mo_b*mo_occ[1], mo_b.conj().T)
        return numpy.array((dm_a,dm_b))
    else:
        from pyscf.cc import uccsd_rdm

        eris = mp.ao2mo(mo_coeff)

        dtype = eris.ovov.dtype
        t2aa = numpy.empty((nocca,nocca,nvira,nvira), dtype=dtype)
        t2ab = numpy.empty((nocca,noccb,nvira,nvirb), dtype=dtype)
        t2bb = numpy.empty((noccb,noccb,nvirb,nvirb), dtype=dtype)
        t2 = (t2aa,t2ab,t2bb)
        
        for i in range(nocca):
            if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
                # When mf._eri is a custom integrals wiht the shape (n,n,n,n), the
                # ovov integrals might be in a 4-index tensor.
                eris_ovov = eris.ovov[i]
            else:
                eris_ovov = numpy.asarray(eris.ovov[i*nvira:(i+1)*nvira])

            eris_ovov = eris_ovov.reshape(nvira,nocca,nvira).transpose(1,0,2)
            t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_a[i], eia_a)
            t2aa[i] = t2i - t2i.transpose(0,2,1)
            #print("t2aa")
            #print(t2aa[i])


            if isinstance(eris.ovOV, numpy.ndarray) and eris.ovOV.ndim == 4:
                # When mf._eri is a custom integrals wiht the shape (n,n,n,n), the
                # ovov integrals might be in a 4-index tensor.
                eris_ovov = eris.ovOV[i]
            else:
                eris_ovov = numpy.asarray(eris.ovOV[i*nvira:(i+1)*nvira])
            eris_ovov = eris_ovov.reshape(nvira,noccb,nvirb).transpose(1,0,2)
            t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_a[i], eia_b)
            t2ab[i] = t2i
            

        for i in range(noccb):
            if isinstance(eris.OVOV, numpy.ndarray) and eris.OVOV.ndim == 4:
                # When mf._eri is a custom integrals wiht the shape (n,n,n,n), the
                # ovov integrals might be in a 4-index tensor.
                eris_ovov = eris.OVOV[i]
            else:
                eris_ovov = numpy.asarray(eris.OVOV[i*nvirb:(i+1)*nvirb])
            eris_ovov = eris_ovov.reshape(nvirb,noccb,nvirb).transpose(1,0,2)
            t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_b[i], eia_b)
            t2bb[i] = t2i - t2i.transpose(0,2,1)
            
        doo, dvv = _gamma1_intermediates(mp, t2)
        nocca, noccb, nvira, nvirb = t2[1].shape
        dov = numpy.zeros((nocca,nvira))
        dOV = numpy.zeros((noccb,nvirb))
        d1 = (doo, (dov, dOV), (dov.T, dOV.T), dvv)
        rdm1 = uccsd_rdm._make_rdm1(mp, d1, with_frozen=False, ao_repr=use_ao)
        #if use_ao:
        #    rdm1_ao =  (reduce(numpy.dot, (mo_coeff[0], rdm1[0], mo_coeff[0].T)), 
        #                reduce(numpy.dot, (mo_coeff[1], rdm1[1], mo_coeff[1].T)))
        #    return rdm1_ao
        #else:
        return rdm1
# DO NOT make tag_array for DM here because the DM arrays may be modified and
# passed to functions like get_jk, get_vxc.  These functions may take the tags
# (mo_coeff, mo_occ) to compute the potential if tags were found in the DM
# arrays and modifications to DM arrays may be ignored.

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


# spin-orbital rdm2 in Chemist's notation
def make_rdm2(mp, t2=None):
    r'''
    Two-particle spin density matrices dm2aa, dm2ab, dm2bb in MO basis

    dm2aa[p,q,r,s] = <q_alpha^\dagger s_alpha^\dagger r_alpha p_alpha>
    dm2ab[p,q,r,s] = <q_alpha^\dagger s_beta^\dagger r_beta p_alpha>
    dm2bb[p,q,r,s] = <q_beta^\dagger s_beta^\dagger r_beta p_beta>

    (p,q correspond to one particle and r,s correspond to another particle)
    Two-particle density matrix should be contracted to integrals with the
    pattern below to compute energy

    E = numpy.einsum('pqrs,pqrs', eri_aa, dm2_aa)
    E+= numpy.einsum('pqrs,pqrs', eri_ab, dm2_ab)
    E+= numpy.einsum('pqrs,rspq', eri_ba, dm2_ab)
    E+= numpy.einsum('pqrs,pqrs', eri_bb, dm2_bb)

    where eri_aa[p,q,r,s] = (p_alpha q_alpha | r_alpha s_alpha )
    eri_ab[p,q,r,s] = ( p_alpha q_alpha | r_beta s_beta )
    eri_ba[p,q,r,s] = ( p_beta q_beta | r_alpha s_alpha )
    eri_bb[p,q,r,s] = ( p_beta q_beta | r_beta s_beta )
    '''
    mo_coeff = mp.mo_coeff
    mo_occ   = mp._scf.mo_occ
    nocca, noccb = nocca0, noccb0 = mp.get_nocc()
    nmoa, nmob = nmoa0, nmob0 = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb
    mo_ea, mo_eb = mp.mo_energy
    eia_a = mo_ea[:nocca,None] - mo_ea[None,nocca:]
    eia_b = mo_eb[:noccb,None] - mo_eb[None,noccb:]
    eris = mp.ao2mo(mp.mo_coeff)

    dtype = eris.ovov.dtype
    t2aa = numpy.empty((nocca,nocca,nvira,nvira), dtype=dtype)
    t2ab = numpy.empty((nocca,noccb,nvira,nvirb), dtype=dtype)
    t2bb = numpy.empty((noccb,noccb,nvirb,nvirb), dtype=dtype)
    t2 = (t2aa,t2ab,t2bb)
    
    for i in range(nocca):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals wiht the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.ovov[i]
        else:
            eris_ovov = numpy.asarray(eris.ovov[i*nvira:(i+1)*nvira])

        eris_ovov = eris_ovov.reshape(nvira,nocca,nvira).transpose(1,0,2)
        t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_a[i], eia_a)
        t2aa[i] = t2i - t2i.transpose(0,2,1)
        #print("t2aa")
        #print(t2aa[i])


        if isinstance(eris.ovOV, numpy.ndarray) and eris.ovOV.ndim == 4:
            # When mf._eri is a custom integrals wiht the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.ovOV[i]
        else:
            eris_ovov = numpy.asarray(eris.ovOV[i*nvira:(i+1)*nvira])
        eris_ovov = eris_ovov.reshape(nvira,noccb,nvirb).transpose(1,0,2)
        t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_a[i], eia_b)
        t2ab[i] = t2i

    if not (mp.frozen == 0 or mp.frozen is None):
        nmoa0 = mp.mo_occ[0].size
        nmob0 = mp.mo_occ[1].size
        nocca0 = numpy.count_nonzero(mp.mo_occ[0] > 0)
        noccb0 = numpy.count_nonzero(mp.mo_occ[1] > 0)
        moidxa, moidxb = mp.get_frozen_mask()
        oidxa = numpy.where(moidxa & (mp.mo_occ[0] > 0))[0]
        vidxa = numpy.where(moidxa & (mp.mo_occ[0] ==0))[0]
        oidxb = numpy.where(moidxb & (mp.mo_occ[1] > 0))[0]
        vidxb = numpy.where(moidxb & (mp.mo_occ[1] ==0))[0]

        dm2aa = numpy.zeros((nmoa0,nmoa0,nmoa0,nmoa0), dtype=t2aa.dtype)
        dm2ab = numpy.zeros((nmoa0,nmoa0,nmob0,nmob0), dtype=t2aa.dtype)
        dm2bb = numpy.zeros((nmob0,nmob0,nmob0,nmob0), dtype=t2aa.dtype)

        tmp = t2aa.transpose(0,2,1,3)
        dm2aa[oidxa[:,None,None,None],vidxa[:,None,None],oidxa[:,None],vidxa] = tmp
        dm2aa[vidxa[:,None,None,None],oidxa[:,None,None],vidxa[:,None],oidxa] = tmp.conj().transpose(1,0,3,2)

        tmp = t2bb.transpose(0,2,1,3)
        dm2bb[oidxb[:,None,None,None],vidxb[:,None,None],oidxb[:,None],vidxb] = tmp
        dm2bb[vidxb[:,None,None,None],oidxb[:,None,None],vidxb[:,None],oidxb] = tmp.conj().transpose(1,0,3,2)

        dm2ab[oidxa[:,None,None,None],vidxa[:,None,None],oidxb[:,None],vidxb] = t2ab.transpose(0,2,1,3)
        dm2ab[vidxa[:,None,None,None],oidxa[:,None,None],vidxb[:,None],oidxb] = t2ab.conj().transpose(2,0,3,1)

    else:
        dm2aa = numpy.zeros((nmoa0,nmoa0,nmoa0,nmoa0), dtype=t2aa.dtype)
        dm2ab = numpy.zeros((nmoa0,nmoa0,nmob0,nmob0), dtype=t2aa.dtype)
        dm2bb = numpy.zeros((nmob0,nmob0,nmob0,nmob0), dtype=t2aa.dtype)

#:tmp = (t2aa.transpose(0,2,1,3) - t2aa.transpose(0,3,1,2)) * .5
#: t2aa.transpose(0,2,1,3) == -t2aa.transpose(0,3,1,2)
        tmp = t2aa.transpose(0,2,1,3)
        dm2aa[:nocca0,nocca0:,:nocca0,nocca0:] = tmp
        dm2aa[nocca0:,:nocca0,nocca0:,:nocca0] = tmp.conj().transpose(1,0,3,2)

        tmp = t2bb.transpose(0,2,1,3)
        dm2bb[:noccb0,noccb0:,:noccb0,noccb0:] = tmp
        dm2bb[noccb0:,:noccb0,noccb0:,:noccb0] = tmp.conj().transpose(1,0,3,2)

        dm2ab[:nocca0,nocca0:,:noccb0,noccb0:] = t2ab.transpose(0,2,1,3)
        dm2ab[nocca0:,:nocca0,noccb0:,:noccb0] = t2ab.transpose(2,0,3,1).conj()

    dm1a, dm1b = make_rdm1(mp, t2, use_ao=False)
    dm1a[numpy.diag_indices(nocca0)] -= 1
    dm1b[numpy.diag_indices(noccb0)] -= 1

    for i in range(nocca0):
        dm2aa[i,i,:,:] += dm1a.T
        dm2aa[:,:,i,i] += dm1a.T
        dm2aa[:,i,i,:] -= dm1a.T
        dm2aa[i,:,:,i] -= dm1a
        dm2ab[i,i,:,:] += dm1b.T
    for i in range(noccb0):
        dm2bb[i,i,:,:] += dm1b.T
        dm2bb[:,:,i,i] += dm1b.T
        dm2bb[:,i,i,:] -= dm1b.T
        dm2bb[i,:,:,i] -= dm1b
        dm2ab[:,:,i,i] += dm1a.T

    for i in range(nocca0):
        for j in range(nocca0):
            dm2aa[i,i,j,j] += 1
            dm2aa[i,j,j,i] -= 1
    for i in range(noccb0):
        for j in range(noccb0):
            dm2bb[i,i,j,j] += 1
            dm2bb[i,j,j,i] -= 1
    for i in range(nocca0):
        for j in range(noccb0):
            dm2ab[i,i,j,j] += 1

    return dm2aa, dm2ab, dm2bb


def spin_square_general(mp, rdm1, rdm2, mo_coeff, ovlp=None):
    r'''General spin square operator.

    ... math::

        <CI|S_+*S_-|CI> &= n_\alpha + \delta_{ik}\delta_{jl}Gamma_{i\alpha k\beta ,j\beta l\alpha } \\
        <CI|S_-*S_+|CI> &= n_\beta + \delta_{ik}\delta_{jl}Gamma_{i\beta k\alpha ,j\alpha l\beta } \\
        <CI|S_z*S_z|CI> &= \delta_{ik}\delta_{jl}(Gamma_{i\alpha k\alpha ,j\alpha l\alpha }
                        - Gamma_{i\alpha k\alpha ,j\beta l\beta }
                        - Gamma_{i\beta k\beta ,j\alpha l\alpha}
                        + Gamma_{i\beta k\beta ,j\beta l\beta})
                        + (n_\alpha+n_\beta)/4

    Given the overlap betwen non-degenerate alpha and beta orbitals, this
    function can compute the expectation value spin square operator for
    UHF-FCI wavefunction
    '''

    #mo_coeff = (numpy.eye(norb),) * 2

    ovlp = mp._scf.get_ovlp()

    dm1a, dm1b = rdm1
    dm2aa, dm2ab, dm2bb = rdm2
    #print("dm1a")
    #print(dm1a)
    #print("dm2aa")
    #print(dm2aa)

#if isinstance(mo_coeff, numpy.ndarray) and mo_coeff.ndim == 2:
    #    mo_coeff = (mo_coeff, mo_coeff)

# projected overlap matrix elements for partial trace
    if isinstance(ovlp, numpy.ndarray):
        ovlpaa = reduce(numpy.dot, (mo_coeff[0].T, ovlp, mo_coeff[0]))
        ovlpbb = reduce(numpy.dot, (mo_coeff[1].T, ovlp, mo_coeff[1]))
        ovlpab = reduce(numpy.dot, (mo_coeff[0].T, ovlp, mo_coeff[1]))
        ovlpba = reduce(numpy.dot, (mo_coeff[1].T, ovlp, mo_coeff[0]))
    else:
        ovlpaa = numpy.dot(mo_coeff[0].T, mo_coeff[0])
        ovlpbb = numpy.dot(mo_coeff[1].T, mo_coeff[1])
        ovlpab = numpy.dot(mo_coeff[0].T, mo_coeff[1])
        ovlpba = numpy.dot(mo_coeff[1].T, mo_coeff[0])

    # if ovlp=1, ssz = (neleca-nelecb)**2 * .25
    ssz =(numpy.einsum('ijkl,ij,kl->', dm2aa, ovlpaa, ovlpaa)
        - numpy.einsum('ijkl,ij,kl->', dm2ab, ovlpaa, ovlpbb)
        + numpy.einsum('ijkl,ij,kl->', dm2bb, ovlpbb, ovlpbb)
        - numpy.einsum('ijkl,ij,kl->', dm2ab, ovlpaa, ovlpbb)) * .25 \
        +(numpy.einsum('ji,ij->', dm1a, ovlpaa)
        + numpy.einsum('ji,ij->', dm1b, ovlpbb)) *.25

    dm2abba = -dm2ab.transpose(0,3,2,1)  # alpha^+ beta^+ alpha beta
    dm2baab = -dm2ab.transpose(2,1,0,3)  # beta^+ alpha^+ beta alpha
    ssxy =(numpy.einsum('ijkl,ij,kl->', dm2baab, ovlpba, ovlpab)
        + numpy.einsum('ijkl,ij,kl->', dm2abba, ovlpab, ovlpba)
        + numpy.einsum('ji,ij->', dm1a, ovlpaa)
        + numpy.einsum('ji,ij->', dm1b, ovlpbb)) * .5
    ss = ssxy + ssz

    s = numpy.sqrt(ss+.25) - .5
    multip = s*2+1
    return ss, multip

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
    e_gyro = .5 * nist.G_ELECTRON
    nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
    au2MHz = nist.HARTREE2J / nist.PLANCK * 1e-6
    fac = nist.ALPHA**2 / 2 / effspin * e_gyro * au2MHz

    hfc = []
    for i, atm_id in enumerate(hfc_nuc):
        nuc_gyro = get_nuc_g_factor(mol.atom_symbol(atm_id)) * nuc_mag
        h1fc = _get_integrals_fc(mol, atm_id)
        fc = numpy.einsum('ij,ji', h1fc, spindm)

        #sd = fcsd + numpy.eye(3) * fc

        logger.info(mp, 'FC of atom %d : %8.6f (in MHz)', atm_id, (2.*fac * nuc_gyro * fc))
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


def mom_occ_(mp, occorb, setocc):
    '''Use maximum overlap method to determine occupation number for each orbital in every
    iteration. It can be applied to unrestricted HF/KS and restricted open-shell
    HF/KS.'''
    from pyscf.scf import uhf, rohf
    if isinstance(mf, uhf.UHF):
        coef_occ_a = occorb[0][:, setocc[0]>0]
        coef_occ_b = occorb[1][:, setocc[1]>0]
    elif isinstance(mf, rohf.ROHF):
        if mf.mol.spin != (numpy.sum(setocc[0]) - numpy.sum(setocc[1])):
            raise ValueError('Wrong occupation setting for restricted open-shell calculation.')
        coef_occ_a = occorb[:, setocc[0]>0]
        coef_occ_b = occorb[:, setocc[1]>0]
    else:
        raise RuntimeError('Cannot support this class of instance %s' % mf)
    log = logger.Logger(mf.stdout, mf.verbose)
    def get_occ(mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if isinstance(mf, rohf.ROHF): mo_coeff = numpy.array([mo_coeff, mo_coeff])
        mo_occ = numpy.zeros_like(setocc)
        nocc_a = int(numpy.sum(setocc[0]))
        nocc_b = int(numpy.sum(setocc[1]))
        s_a = reduce(numpy.dot, (coef_occ_a.T, mf.get_ovlp(), mo_coeff[0]))
        s_b = reduce(numpy.dot, (coef_occ_b.T, mf.get_ovlp(), mo_coeff[1]))
        #choose a subset of mo_coeff, which maximizes <old|now>
        idx_a = numpy.argsort(numpy.einsum('ij,ij->j', s_a, s_a))[::-1]
        idx_b = numpy.argsort(numpy.einsum('ij,ij->j', s_b, s_b))[::-1]
        mo_occ[0][idx_a[:nocc_a]] = 1.
        mo_occ[1][idx_b[:nocc_b]] = 1.

        log.debug(' New alpha occ pattern: %s', mo_occ[0])
        log.debug(' New beta occ pattern: %s', mo_occ[1])
        if isinstance(mf.mo_energy, numpy.ndarray) and mf.mo_energy.ndim == 1:
            log.debug1(' Current mo_energy(sorted) = %s', mo_energy)
        else:
            log.debug1(' Current alpha mo_energy(sorted) = %s', mo_energy[0])
            log.debug1(' Current beta mo_energy(sorted) = %s', mo_energy[1])

        if (int(numpy.sum(mo_occ[0])) != nocc_a):
            log.error('mom alpha electron occupation numbers do not match: %d, %d',
                    nocc_a, int(numpy.sum(mo_occ[0])))
        if (int(numpy.sum(mo_occ[1])) != nocc_b):
            log.error('mom alpha electron occupation numbers do not match: %d, %d',
                    nocc_b, int(numpy.sum(mo_occ[1])))

        #output 1-dimension occupation number for restricted open-shell
        if isinstance(mf, rohf.ROHF): mo_occ = mo_occ[0, :] + mo_occ[1, :]
        return mo_occ
    mf.get_occ = get_occ
    return mf
mom_occ = mom_occ_


class UB2PLYPDFUOBMP2(DFOBMP2):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)


    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    int_transform_ss = int_transform_ss
    int_transform_os = int_transform_os
    mom_select = mom_select
    mom_reorder = mom_reorder
    break_sym = False
    #use_t2 = False
    cos = 1.
    css = 1.
########################13/5/25 them spin contaminent   
    make_S2 = make_S2
    make_amp = make_amp
    

    @lib.with_doc(OBMP2_slow.kernel.__doc__)
    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, _kernel=kernel):

        # self.ene_tot =_kernel(self, mo_energy, mo_coeff, eris, with_t2, alphaa)
        
        if hasattr(self, 'alphaa'):
            self.ene_tot =_kernel(self, mo_energy, mo_coeff, eris, with_t2, self.alphaa)
        else:
            self.ene_tot =_kernel(self, mo_energy, mo_coeff, eris, with_t2,  verbose=logger.NOTE)

        return self.ene_tot

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return _make_eris(self, mo_coeff, verbose=self.verbose)

    make_rdm1 = make_rdm1
    make_rdm2 = make_rdm2
    spin_square_general = spin_square_general
    make_fc = make_fc
    eval_fc = False
    make_IPEA = make_IPEA
    eval_IPEA = True
    ip = 0

    def nuc_grad_method(self):
        from pyscf.grad import ump2
        return ump2.Gradients(self)

OBMP2 = UB2PLYPDFUOBMP2

#from pyscf import scf
#scf.uhf.UHF.MP2 = lib.class_as_method(MP2)


class _ChemistsERIs(_ChemistsERIs):
    def __init__(self, mp, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        moidx = mp.get_frozen_mask()
        self.mo_coeff = mo_coeff = \
                (mo_coeff[0][:,moidx[0]], mo_coeff[1][:,moidx[1]])

def _make_eris(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    log = logger.new_logger(mp, verbose)
    time0 = (time.process_time(), time.time())
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
    time0 = (time.process_time(), time.time())
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
    disk = (nocca**2.*(nao*(nao+dmax)/2+nvira**2) +
            noccb**2.*(nao*(nao+dmax)/2+nvirb**2) +
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
                    max(4, 250/nocca, max_memory*.9e6/8/(nao**2.*nocca)/5)))

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
    import os
    # Khóa OpenBLAS/MKL lại ở 1 luồng để tránh xung đột
    os.environ['OPENBLAS_NUM_THREADS'] = '6'
    # os.environ['MKL_NUM_THREADS'] = '1'   

    # CẤP QUYỀN ĐA LUỒNG CHO PySCF ở đây (Ví dụ: dùng 8 luồng CPU)
    os.environ['OMP_NUM_THREADS'] = '6'

    from pyscf import scf
    from pyscf import gto
    #from pyscf.mp import dfuobmp2_faster_ram , dfump2_native,ump2
    from pyscf.mp import dfump2_native,ump2
    from pycmf.OBMP import dfuobmp2



    mol = gto.Mole()

    mol.atom = '''
    C 0 0 0 
    O 0 0 1.2
    H 0 0.93 -0.58
    H 0 -0.93 -0.58
    '''

    mol.spin = 0
    mol.verbose= 0
    mol.basis = 'aug-cc-pvtz'
    mol.build()
    # mf = scf.UHF(mol).run()
    mf = scf.UHF(mol).density_fit().run()
    mppp = UB2PLYPDFUOBMP2(mf)
    mppp.alphaa= (0.53,0.39)
    mppp.thresh=1e-06
    mppp.active_atoms = [1,2,3]
    mppp.mu = 1e6
    mppp.use_cl = False
    mppp.n_shells = 1

    # =========================================================================
    # BENCHMARK
    # =========================================================================
    """
    from CL_benchmark import CLBenchmark

    bench = CLBenchmark(mppp, mol, mf)

    # Chạy no-CL (None) và CL với n_shells = 0, 1, 2, 3
    bench.run(n_shells_list=[None, 0, 1, 2])

    # In bảng tổng hợp
    bench.report()

    # Vẽ và lưu đồ thị
    bench.plot('/home/bghuy1309/Result/CL_benchmark/Rcl_benchmark.png')

    # Lưu CSV để phân tích thêm
    bench.save_csv('/home/bghuy1309/Result/CL_benchmark/cl_benchmark.csv')
    # =========================================================================
    """
    