#!/usr/bin/env python3
import os
import numpy as np
import scipy.linalg
from functools import reduce
from pyscf import gto, scf, dft, mp, cc
from pycmf.OBMP import DFUOBMP2_mom_diis, UOBMP2_mom_diis, UHF_mom_diis

# ------------------------------------------------------------ #
# Units
# ------------------------------------------------------------ #
# 1 a.u. = 2.541746 Debye    (for Transition dipole)
# 1 a.u. = 27.211408 eV      (for Energy)
har_to_eV = 27.211408  # (eV)
har_to_D = 2.541746  # (Debye)

# ------------------------------------------------------------ #
# Hàm con Tính P_uv
# ------------------------------------------------------------ #


def pseudo_solve_transpose(oT, B, svd_threshold=1e-5):
    # Giải oT·X = B an toàn (nếu oT gần kỳ dị thì dùng pseudo-inverse) #
    try:
        X = np.linalg.solve(oT, B)
    except np.linalg.LinAlgError:
        u, s, vt = scipy.linalg.svd(oT, full_matrices=False)
        s_inv = np.array([1.0 / x if x > svd_threshold else 0.0 for x in s])
        X = (vt.T * s_inv) @ u.T @ B
    return X


def calc_transition_density_term(s_ao, c_m_occ, c_n_occ, svd_threshold=1e-5, gsc_threshold=1e-3):
    """
    Tính ma trận mật độ chuyển tiếp (P_trans) cho một spin.
    Sử dụng SVD-stabilized pseudo-inverse (DML) HOẶC GSC nếu phát hiện 1 khác biệt.
    Tham khảo SVD
    """
    # 1. Tính Ma trận chồng chập MO (O)
    ovlp_mat = c_m_occ.conj().T @ s_ao @ c_n_occ

    # 2. SVD: O = U @ Sigma @ V.T
    u, s, vt = scipy.linalg.svd(ovlp_mat)
    det_u = np.linalg.det(u)
    det_vt = np.linalg.det(vt)
    phase = det_u * det_vt
    det_val = phase * np.prod(s)

    N_occ = s.size

    # Phát hiện 1-khác biệt: Tìm điểm kì dị xấp xỉ 0
    idx_diff = np.argmin(s)
    s_min = s[idx_diff]

    # Kiểm tra điều kiện GSC
    s_spectator = np.prod(s[np.arange(N_occ) != idx_diff])

    # Phân tích GSC:
    if s_min < gsc_threshold and np.isclose(s_spectator, 1.0, atol=gsc_threshold**0.5):
        # TRƯỜNG HỢP 1: PHÁT HIỆN GSC (1 khác biệt) -> Dùng công thức GSC ổn định
        phi_i = c_m_occ @ u[:, idx_diff]
        v_j = vt[idx_diff, :].conj()
        phi_j = c_n_occ @ v_j
        p_trans = (phase * s_spectator) * np.outer(phi_i, phi_j.conj())
        mode = 'GSC'
    else:
        # TRƯỜNG HỢP 2: DML (Dùng cho nhiều khác biệt hoặc tính toán ổn định)
        s_inv = np.where(s > svd_threshold, 1.0 / s, 0.0)
        x_matrix = vt.conj().T @ np.diag(s_inv) @ u.conj().T
        p_trans = c_m_occ @ x_matrix @ c_n_occ.conj().T
        mode = 'DML'

    return det_val, p_trans, mode, s_spectator


def contract_tdm_ao(p_trans, r_ao):
    # Thực hiện phép co giữa Ma trận mật độ chuyển tiếp (P_trans) và Tích phân Dipole AO (r_ao).
    term_vec = np.zeros(3)
    for i in range(3):  # Loop qua x, y, z
        term_vec[i] = np.einsum('ij,ji->', p_trans, r_ao[i])
    return term_vec


# ------------------------------------------------------------ #
# Hàm con tính transition dipole
# (Tuy nhiên em không dùng trực tiếp trong main mà code lại tương tự
#  vì em muốn coi nhiều thông số trong quá trình tính toán để kiểm chứng
#  hàm sóng có bị sụp về GS hay không)
# ------------------------------------------------------------ #


def calc_tdm(mol, mf_m, mf_n, svd_threshold=1e-5, gsc_threshold=1e-3):
    """
    Tính TDM giữa hai trạng thái UHF không trực giao (OB-MP2)
    sử dụng DML hoặc GSC dựa trên phát hiện SVD.
    (Đã đồng bộ tên biến với main loop)
    """
    S = mol.intor('int1e_ovlp')
    rx_tdm = mol.intor('int1e_r', comp=3)

    nelec_a, nelec_b = mol.nelec

    c_m_a_occ = mf_m.mo_coeff[0][:, :nelec_a]
    c_m_b_occ = mf_m.mo_coeff[1][:, :nelec_b]
    c_n_a_occ = mf_n.mo_coeff[0][:, :nelec_a]
    c_n_b_occ = mf_n.mo_coeff[1][:, :nelec_b]

    det_a, p_trans_a, mode_a, sa_spectator = calc_transition_density_term(
        S, c_m_a_occ, c_n_a_occ, svd_threshold, gsc_threshold
    )
    det_b, p_trans_b, mode_b, sb_spectator = calc_transition_density_term(
        S, c_m_b_occ, c_n_b_occ, svd_threshold, gsc_threshold
    )

    det_ovlpMO = det_a * det_b

    term_a = contract_tdm_ao(p_trans_a, rx_tdm)
    term_b = contract_tdm_ao(p_trans_b, rx_tdm)
    mu_elec = -1.0 * ((det_b * term_a) + (det_a * term_b))

    charges = mol.atom_charges()
    coords_bohr = mol.atom_coords(unit='Bohr')
    nucl_dip_vec_au = np.einsum('i,ix->x', charges, coords_bohr)
    mu_nucl = nucl_dip_vec_au * det_ovlpMO

    mu_total_raw = mu_elec + mu_nucl
    mu_trans = 1 / np.sqrt(2) * mu_total_raw * 2
    mu_sq = np.sum(mu_trans**2)

    return mu_trans, mu_sq


# ============================================================================================================================================

# ------------------------------------------------------------ #
# Main calculation
# ------------------------------------------------------------ #

# --- Molecule setup ---
print('\n' + '=' * 30 + ' MOLECULE SETUP ' + '=' * 30)
Molecule = '02-H2O-H2O'
mol = gto.M(
    atom="""
        O    -0.066999140    0.000000000    1.494354740
        H     0.815734270    0.000000000    1.865866390
        H     0.068855100    0.000000000    0.539142770
        O     0.062547750    0.000000000   -1.422632080
        H    -0.406965400   -0.760178410   -1.771744500
        H    -0.406965400    0.760178410   -1.771744500
    """,
    basis='cc-pVDZ',
    unit='A',
    spin=0,
    verbose=0,
)
mol.build()

# Pre-calculate integrals for Dipole (Global usage)
rx = mol.intor('int1e_r', comp=3)

# Dictionary để lưu trữ các đối tượng OB-MP2 sau khi tính xong
calc_results = {}
state_energies = {}
state_info = {}

# ======================================================================
# 1. GROUND STATE CALCULATION
# ======================================================================
print(f'\n=== Ground state (SCF + OB-MP2) ===')

hf_gs = scf.UHF(mol)
hf_gs.scf()
mo0_gs = hf_gs.mo_coeff
occ_gs = hf_gs.mo_occ

# OB-MP2 GS
ob_gs = UOBMP2_mom_diis(hf_gs)
ob_gs.second_order = True
ob_gs.mom_select = True
ob_gs.thresh = 1e-8

print('> GS: Running final calculation (css=1, cos=1.0)...')
ob_gs.css = 1
ob_gs.cos = 1
e_obmp2_gs = ob_gs.kernel()

# Lưu kết quả GS
calc_results['GS'] = ob_gs
state_energies['GS'] = e_obmp2_gs

n_ao_gs = ob_gs.mo_coeff[0].shape[0]
n_mo_gs = ob_gs.mo_coeff[0].shape[1]
print(f'[GS] AO = {n_ao_gs}, MO = {n_mo_gs}')

# ======================================================================
# 2. EXCITED STATES CALCULATION (LOOP)
# ======================================================================
transitions = [(9, 11, 'ES1'), (8, 10, 'ES2'), (9, 10, 'ES3'), (8, 12, 'ES4'), (7, 10, 'ES5'), (6, 10, 'ES6')]
es_summary = []

for hole, electron, label in transitions:
    # 1. Setup Occupations
    mo0_es = hf_gs.mo_coeff.copy()
    occ_es = hf_gs.mo_occ.copy()

    # Excitation logic
    occ_es[1][hole] = 0
    occ_es[1][electron] = 1

    # 2. MOM - SCF
    scf_mom = scf.UHF(mol)
    scf_mom = scf.addons.mom_occ(scf_mom, mo0_es, occ_es)
    dm_init = scf_mom.make_rdm1(mo0_es, occ_es)
    scf_mom.scf(dm_init)
    hf = UHF_mom_diis(scf_mom).run()

    print(f'\n=== Excited states (MOM + OB-MP2) ===')
    print(f'\n--- Processing {label}: Hole {hole} -> Elec {electron} ---')

    # 3. OB-MP2
    ob_es = UOBMP2_mom_diis(scf_mom)
    ob_es.second_order = True
    ob_es.mom_select = True
    ob_es.thresh = 1e-8
    ob_es.niter = 500

    print(f'  -> {label}: Final optimization (css=1, cos=1.0)...')
    ob_es.css = 1
    ob_es.cos = 1
    e_obmp2 = ob_es.kernel()  # Lấy năng lượng cuối cùng tại đây

    # 4. Lưu đối tượng tính toán
    calc_results[label] = ob_es

    n_ao = ob_es.mo_coeff[0].shape[0]
    n_mo = ob_es.mo_coeff[0].shape[1]

    es_summary.append({'label': label, 'hole': hole, 'elec': electron, 'energy': e_obmp2, 'n_ao': n_ao, 'n_mo': n_mo})

# ===================== SUMMARY & COMPARISON =====================
print('\n' + '=' * 20 + ' ENERGIES SUMMARY ' + '=' * 20)

# 1. Xử lý Ground State (GS)
print(f'GS  Energy: {e_obmp2_gs:10.8f} a.u.')
print('-' * 60)

# 2. Xử lý Excited States (ES)
for res in es_summary:
    delta_ev = (res['energy'] - e_obmp2_gs) * har_to_eV
    print(f'[{res["label"]}] Hole {res["hole"]} -> Elec {res["elec"]}')
    print(f'      Energy  : {res["energy"]:10.8f} a.u. (ΔE = {delta_ev:10.4f} eV)')
    print('-' * 60)

# ======================================================================
# TRANSITION DIPOLE MOMENTS
# ======================================================================
# Cấu hình danh sách cặp trạng thái (GS -> ES)
state_objs = calc_results
for res in es_summary:
    state_energies[res['label']] = res['energy']

pairs = [('GS', item[2]) for item in transitions]

# Gán các giá trị tham số
SVD_THRESHOLD = 1e-5
GSC_THRESHOLD = 0.05

# 1. Chuẩn bị các tích phân cơ bản
mol = ob_gs.mol
S = mol.intor('int1e_ovlp')
rx_tdm = mol.intor('int1e_r', comp=3)

# 2. Tính Vector Dipole Hạt nhân =  Sum(Z_i * R_i)
charges = mol.atom_charges()
coords_bohr = mol.atom_coords(unit='Bohr')
nucl_dip_vec_au = np.einsum('i,ix->x', charges, coords_bohr)

# 3. Lấy số lượng electron
nelec_a, nelec_b = mol.nelec

# 4. Vòng lặp tính toán TDM
for m, n in pairs:
    print('\n' + '=' * 70)
    print(f' TRANSITION: {m} <- {n}')
    print('=' * 70)

    mf_m = state_objs[m]
    mf_n = state_objs[n]
    E_m = state_energies[m]
    E_n = state_energies[n]

    # Trích xuất hệ số MO bị chiếm (OCCUPIED)
    c_m_a_occ = mf_m.mo_coeff[0][:, :nelec_a]
    c_m_b_occ = mf_m.mo_coeff[1][:, :nelec_b]
    c_n_a_occ = mf_n.mo_coeff[0][:, :nelec_a]
    c_n_b_occ = mf_n.mo_coeff[1][:, :nelec_b]

    # Xử lý Spin Alpha và Beta (m -> n) - Dùng hàm tích hợp GSC/DML
    det_a, p_trans_a, mode_a, sa_spectator = calc_transition_density_term(
        S, c_m_a_occ, c_n_a_occ, SVD_THRESHOLD, GSC_THRESHOLD
    )
    det_b, p_trans_b, mode_b, sb_spectator = calc_transition_density_term(
        S, c_m_b_occ, c_n_b_occ, SVD_THRESHOLD, GSC_THRESHOLD
    )

    # Tổng hợp các thông số chẩn đoán
    det_ovlpMO = det_a * det_b  # Đây chính là S_12 (Total Overlap)

    # In thông tin chẩn đoán
    print(f'Overlap Alpha (Det)          : {det_a: .8f} ({mode_a})')
    print(f'Overlap Beta (Det)           : {det_b: .8f} ({mode_b})')
    print(f'sa_spectator                 : {sa_spectator: .8f}')
    print(f'sb_spectator                 : {sb_spectator: .8f}')
    print(f'Total Wavefunction Overlap   : {det_ovlpMO: .8f}')
    print('Max values of Transition Density matrices:')
    print(f'  max|P_trans_a| = {np.max(np.abs(p_trans_a)):>12.8f}')
    print(f'  max|P_trans_b| = {np.max(np.abs(p_trans_b)):>12.8f}')

    # --- TÍNH TOÁN TRANSITION DIPOLE ---

    # A. Phần điện tử (Electronic Component)
    term_a = contract_tdm_ao(p_trans_a, rx_tdm)
    term_b = contract_tdm_ao(p_trans_b, rx_tdm)
    mu_elec = -1.0 * ((det_b * term_a) + (det_a * term_b))

    # B. Phần hạt nhân (Nuclear Component)
    mu_nucl = nucl_dip_vec_au * det_ovlpMO

    # C. Tổng Dipole Raw (Chưa scale spin)
    mu_total_raw = mu_elec + mu_nucl

    # D. Scale theo giả định Singlet
    mu_trans = 1 / np.sqrt(2) * mu_total_raw * 2

    # Tính bình phương độ lớn (Dipole Strength)
    mu_sq = np.sum(mu_trans**2)
    mu_mag = np.sqrt(mu_sq)

    print('\nTransition dipole components (a.u. - unscaled):')
    print(f'  Electronic contribution: [{mu_elec[0]:.6f}  {mu_elec[1]:.6f}  {mu_elec[2]:.6f}]')
    print(f'  Nuclear contribution:    [{mu_nucl[0]:.6f}  {mu_nucl[1]:.6f}  {mu_nucl[2]:.6f}]')

    print('\nFinal Transition dipole (a.u.) [Scaled for Singlet]:')
    print(f'  μ_x,y,z = [{mu_trans[0].real:12.6f}  {mu_trans[1].real:12.6f}  {mu_trans[2].real:12.6f}]')
    print(f'  |μ|     = {mu_mag:.8f}')

    # Energy difference
    deltaE = E_n - E_m

    # Oscillator strength (length gauge)
    f_length = (2.0 / 3.0) * deltaE * mu_sq

    print('\nOscillator strength (length gauge):')
    print(f' ΔE ({n}-{m})  = {deltaE: .8f} a.u.   ({deltaE * har_to_eV:.8f} eV)')
    print(f' |μ|^2        = {mu_sq: .8f}  a.u.')
    print(f' f_length     = {f_length: .8f}')
