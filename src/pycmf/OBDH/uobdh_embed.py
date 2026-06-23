import numpy as np
import scipy.linalg as la
from scipy.linalg import fractional_matrix_power
from pyscf import lib, dft, scf
from pyscf.lib import logger
from .uobdh_solver import obmp2_iter

def make_dipole(mol, dm_embed):
    return scf.hf.dip_moment(mol, dm_embed, unit='Debye')

def run_full_dft(mol, xc, df_obj=None):
    ks = dft.UKS(mol)
    ks.xc = xc
    ks.verbose = 0
    if df_obj is not None:
        ks.with_df = df_obj
    else:
        ks = ks.density_fit() 
    ks.kernel()
    return ks

def spade_partition(mol, S, C_occ, atom_indices_A, plot=False, label="Alpha"):
    S_half = fractional_matrix_power(S, 0.5)
    C_bar = S_half @ C_occ
    
    ao_indices_A = []
    ao_slices = mol.aoslice_by_atom()
    for atom_idx in atom_indices_A:
        start, end = ao_slices[atom_idx][2], ao_slices[atom_idx][3]
        ao_indices_A.extend(range(start, end))
        
    C_bar_A = C_bar[ao_indices_A, :]
    U, sigma, Vh = np.linalg.svd(C_bar_A, full_matrices=True)
    V = Vh.T
    C_spade = C_occ @ V
    
    num_A_orbs = 0
    gap_val = 0.0
    
    if np.min(sigma) > 0.99:
        if plot: print(f"   [SPADE-{label}] Detected Full System. Selecting ALL orbitals.")
        num_A_orbs = C_occ.shape[1]
    elif np.max(sigma) < 0.01:
        if plot: print(f"   [SPADE-{label}] Detected Empty System. Selecting 0 orbitals.")
        num_A_orbs = 0
    else:
        if len(sigma) > 1:
            gaps = sigma[:-1] - sigma[1:]
            gap_idx = np.argmax(gaps)
            num_A_orbs = gap_idx + 1
            gap_val = gaps[gap_idx]
        else:
            num_A_orbs = len(sigma)
            gap_val = 0.0 
            
        if plot:
            print(f"   [SPADE-{label}] Active orbitals: {num_A_orbs} (Gap: {gap_val:.4f})")

    C_A = C_spade[:, :num_A_orbs]
    C_B = C_spade[:, num_A_orbs:]
    return C_A, C_B

def build_density_matrix(C_occ):
    return C_occ @ C_occ.T

def get_subsystem_hcore(mol, active_atoms):
    t = mol.intor('int1e_kin')
    v_nuc = np.zeros_like(t)
    for i in active_atoms:
        with mol.with_rinv_origin(mol.atom_coord(i)):
            v_nuc += -mol.atom_charge(i) * mol.intor('int1e_rinv')
    return t + v_nuc

def calculate_dft_energy_isolated(mol, xc, gamma_A_tuple, h_core_A, active_atoms):
    mf_tmp = dft.UKS(mol)
    mf_tmp.xc = xc
    mf_tmp.verbose = 0
    mf_tmp = mf_tmp.density_fit()
    e_elec, _ = mf_tmp.energy_elec([gamma_A_tuple[0], gamma_A_tuple[1]], h1e=h_core_A)
    
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

def build_embedding_potential(mol, xc_code, S, mu, mf_full, gamma_B_tuple, gamma_A_tuple):
    dm_full = mf_full.make_rdm1()
    dm_A = [gamma_A_tuple[0], gamma_A_tuple[1]]
    
    veff_full = mf_full.get_veff(mol, dm_full)
    
    mf_tmp = dft.UKS(mol)
    mf_tmp.xc = xc_code
    mf_tmp.verbose = 0
    mf_tmp.with_df = mf_full.with_df
    veff_A = mf_tmp.get_veff(mol, dm_A)
    
    P_B_a = S @ gamma_B_tuple[0] @ S
    P_B_b = S @ gamma_B_tuple[1] @ S
    
    v_emb_a = veff_full[0] - veff_A[0] + mu * P_B_a
    v_emb_b = veff_full[1] - veff_A[1] + mu * P_B_b
    
    return [v_emb_a, v_emb_b], [P_B_a, P_B_b]

def run_embed_uobmp2(mp, mol, xc, h_core_full, h_core_A_iso, v_emb, gamma_init, num_active_orbs, atom_indices_A, use_cl=False, cl_n_shells=1, cl_mu_threshold=1e5):
    print(f"   [Embedded UOBMP2] Initializing UHF with Embedding Potential...")
    mol_emb = mol.copy()
    na, nb = num_active_orbs
    mol_emb.nelectron = na + nb
    mol_emb.spin = na - nb

    mf_emb = scf.UHF(mol_emb)
    mf_emb.verbose = 0
    mf_emb.with_df = mp.with_df
    mf_emb.with_df.mol = mol_emb
    original_get_veff = mf_emb.get_veff

    def get_veff_emb(mol, dm, dm_last=0, vhf_last=0):
        veff = original_get_veff(mol, dm, dm_last, vhf_last)
        return np.array([veff[0] + v_emb[0], veff[1] + v_emb[1]])

    mf_emb.get_veff = get_veff_emb
    mf_emb.get_hcore = lambda *args: h_core_full

    try:
        mf_emb.kernel(dm0=gamma_init)
    except Exception as e:
        print(f"   [Warning] UHF kernel failed: {e}. Trying without dm0...")
        mf_emb.kernel()

    print(f"   [Embedded UOBMP2] UHF Reference Energy: {mf_emb.e_tot:.8f}")

    if use_cl:
        print(f"   [Embedded UOBMP2] Performing Concentric Localization (n_shells={cl_n_shells})...")
        try:
            from .CL_embed import concentric_localization
            active_aos = []
            aoslice = mol.aoslice_by_atom()
            for atom_id in atom_indices_A:
                p0, p1 = aoslice[atom_id][2], aoslice[atom_id][3]
                active_aos.extend(range(p0, p1))

            S_mat = mf_emb.get_ovlp()
            F_mat = mf_emb.get_fock()

            new_mo_coeff = []
            new_mo_energy = []
            new_mo_occ = []

            for s in [0, 1]:
                C_s = mf_emb.mo_coeff[s]
                occ_s = mf_emb.mo_occ[s]
                eps_s = mf_emb.mo_energy[s]
                F_s = F_mat[s]

                idx_occ = occ_s > 0
                C_occ_A = C_s[:, idx_occ]
                eps_occ_A = eps_s[idx_occ]

                idx_vir_eff = (occ_s == 0) & (eps_s < cl_mu_threshold)
                C_vir_eff = C_s[:, idx_vir_eff]

                C_vir_CL = concentric_localization(C_vir_eff, S_mat, F_s, active_aos, n_shells=cl_n_shells, verbose=True)

                F_vir = C_vir_CL.T.conj() @ F_s @ C_vir_CL
                evals_vir, evecs_vir = la.eigh(F_vir)
                C_vir_CL_canon = C_vir_CL @ evecs_vir

                C_new_s = np.hstack([C_occ_A, C_vir_CL_canon])
                eps_new_s = np.concatenate([eps_occ_A, evals_vir])
                occ_new_s = np.concatenate([np.ones(C_occ_A.shape[1]), np.zeros(C_vir_CL_canon.shape[1])])

                new_mo_coeff.append(C_new_s)
                new_mo_energy.append(eps_new_s)
                new_mo_occ.append(occ_new_s)

            mf_emb.mo_coeff = (new_mo_coeff[0], new_mo_coeff[1])
            mf_emb.mo_energy = (new_mo_energy[0], new_mo_energy[1])
            mf_emb.mo_occ = (new_mo_occ[0], new_mo_occ[1])

            nmo_new = new_mo_coeff[0].shape[1]
            mp.mo_coeff = mf_emb.mo_coeff
            mp.mo_occ = mf_emb.mo_occ
            mp.mo_energy = mf_emb.mo_energy
            mp._nmo = (nmo_new, nmo_new)
            mp.nocc = (np.count_nonzero(mf_emb.mo_occ[0]), np.count_nonzero(mf_emb.mo_occ[1]))
            print(f"   [Embedded UOBMP2] CL truncation done. NMO alpha={mf_emb.mo_coeff[0].shape[1]}, beta={mf_emb.mo_coeff[1].shape[1]}")
        except ImportError:
            print("   [Error] Could not import CL_embed. Falling back to non-CL virtual space.")

    print(f"   [Embedded UOBMP2] Running DIIS...")
    e_tot_or_corr, e_dft, gamma_uobmp2 = obmp2_iter(mp, mol_emb, mf_emb, xc, v_emb, niter=mp.niter)
    
    is_hybrid = getattr(mp, 'is_hybrid', True)
    if is_hybrid:
        return e_tot_or_corr, e_dft, gamma_uobmp2
    else:
        # PURE OBMP2 Energy logic
        gamma_uobmp2_a, gamma_uobmp2_b = gamma_uobmp2
        mf_tmp = dft.UKS(mol)
        mf_tmp.xc = xc
        mf_tmp.verbose = 0
        mf_tmp.with_df = mp.with_df
        
        # 1e- and 2e- energy from UOBMP2 density but isolated nuclei core
        e_elec_meanfield, _ = mf_tmp.energy_elec([gamma_uobmp2_a, gamma_uobmp2_b], h1e=h_core_A_iso)

        coords = mol.atom_coords()
        charges = mol.atom_charges()
        e_nuc_A = 0.0
        for i in range(len(atom_indices_A)):
            for j in range(i + 1, len(atom_indices_A)):
                at_i = atom_indices_A[i]
                at_j = atom_indices_A[j]
                dist = np.linalg.norm(coords[at_i] - coords[at_j])
                e_nuc_A += (charges[at_i] * charges[at_j]) / dist

        e_wf_A_internal = e_elec_meanfield + e_nuc_A + e_tot_or_corr # e_tot_or_corr là e_corr
        return e_wf_A_internal, None, gamma_uobmp2

def embed_kernel(mp):
    mol = mp.mol
    alphaa = mp.alphaa
    xc_code = f"{alphaa[0]}*HF + {1-alphaa[0]}*B88, {1-alphaa[1]}*LYP"
    S = mp._scf.get_ovlp()
    mu = mp.mu
    is_hybrid = getattr(mp, 'is_hybrid', True)
    method_name = "OBDH" if is_hybrid else "OBMP2"

    print('\n' + '='*70)
    print(f'{method_name}-IN-DFT EMBEDDING WITH SPADE PARTITIONING')
    print('='*70)
    print('\n--- STEP 1: Running Full System DFT ---')
    ks_full = run_full_dft(mol, xc_code, df_obj=mp.with_df)
    print(f"Full DFT Energy: {ks_full.e_tot:.8f} Eh")
    h_core_full = ks_full.get_hcore()

    C_occ_a = ks_full.mo_coeff[0][:, ks_full.mo_occ[0] > 0]
    C_occ_b = ks_full.mo_coeff[1][:, ks_full.mo_occ[1] > 0]
    atom_indices_A = mp.active_atoms 

    print("\n --- Partitioning ---")
    C_A_a, C_B_a = spade_partition(mol, S, C_occ_a, atom_indices_A, True, "Alpha")
    C_A_b, C_B_b = spade_partition(mol, S, C_occ_b, atom_indices_A, False, "Beta")

    na_act, nb_act = C_A_a.shape[1], C_A_b.shape[1]
    gamma_A = (build_density_matrix(C_A_a), build_density_matrix(C_A_b))
    gamma_B = (build_density_matrix(C_B_a), build_density_matrix(C_B_b))

    print("\n--- Constructing Potentials ---")
    h_core_A_iso = get_subsystem_hcore(mol, atom_indices_A)
    
    # Riêng OBMP2 cần in và tính E_DFT[A] Isolated làm baseline
    if not is_hybrid:
        e_dft_A_iso = calculate_dft_energy_isolated(mol, xc_code, gamma_A, h_core_A_iso, atom_indices_A)
        print(f"E_DFT[A] (Isolated Total): {e_dft_A_iso:.8f} Eh")

    v_emb, P_B = build_embedding_potential(mol, xc_code, S, mu, ks_full, gamma_B, gamma_A)

    print(f"\n--- Running {method_name} in DFT Environment ---")    
    e_wf_A_internal, e_dft_A_relax, gamma_uobmp2 = run_embed_uobmp2(
                                        mp, mol, xc_code, h_core_full, h_core_A_iso, v_emb,
                                        gamma_A, (na_act, nb_act), atom_indices_A,
                                        use_cl=mp.use_cl, cl_n_shells=mp.n_shells, cl_mu_threshold=1e5)

    gamma_uobmp2_a, gamma_uobmp2_b = gamma_uobmp2

    if is_hybrid:
        # --- Logic Final Energy của OBDH ---
        gamma_relax = (gamma_uobmp2[0] + gamma_B[0], gamma_uobmp2[1] + gamma_B[1])
        e_dft_full_relax = ks_full.energy_tot(dm = gamma_relax)
        e_baseline = e_dft_full_relax - e_dft_A_relax
        e_ortho = mp.mu * (np.einsum('ij,ji', gamma_uobmp2_a, P_B[0]) + np.einsum('ij,ji', gamma_uobmp2_b, P_B[1]))
        e_final = e_wf_A_internal + e_baseline + e_ortho

        print("-" * 60)
        print(f"E_WF[A] (Internal, Recalculated): {e_wf_A_internal:.8f}")
        print(f"Baseline (Full - Iso)           : {e_baseline:.8f}")
        print(f"Orthogonality Correction        : {e_ortho:.8f}")

    else:
        # --- Logic Final Energy của OBMP2 ---
        e_baseline = ks_full.e_tot - e_dft_A_iso
        v_emb_np_a = v_emb[0] - mp.mu * P_B[0]
        v_emb_np_b = v_emb[1] - mp.mu * P_B[1]

        e_relax = np.einsum('ij,ji', gamma_uobmp2_a - gamma_A[0], v_emb_np_a) + \
                  np.einsum('ij,ji', gamma_uobmp2_b - gamma_A[1], v_emb_np_b)
        
        e_ortho = mp.mu * (np.einsum('ij,ji', gamma_uobmp2_a, P_B[0]) + np.einsum('ij,ji', gamma_uobmp2_b, P_B[1]))
        
        e_final = e_wf_A_internal + e_baseline + e_relax + e_ortho

        print("-" * 60)
        print(f"E_WF[A] (Internal, Recalculated): {e_wf_A_internal:.8f}")
        print(f"Baseline (Full - Iso)           : {e_baseline:.8f}")
        print(f"Relaxation Correction           : {e_relax:.8f}")
        print(f"Orthogonality Correction        : {e_ortho:.8f}")

    print("-" * 60)
    print(f"Total {method_name}-in-DFT Energy   : {e_final:.8f} Eh")
    print(f"Ref DFT Energy                  : {ks_full.e_tot:.8f} Eh")
    print(f"Difference (Gain)               : {(e_final - ks_full.e_tot)*1e6:.2f} uEh")
    print("=" * 60)
    
    mp._gamma = (gamma_uobmp2[0] + gamma_B[0], gamma_uobmp2[1] + gamma_B[1])

    return e_final, ks_full.e_tot