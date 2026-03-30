#!/usr/bin/env python3
import os
import numpy as np
import scipy.linalg
from functools import reduce
from pyscf import gto, scf, dft, mp, cc
from pyscf.mp import dfuobmp2_mom_conv, uobmp2_mom_conv, hf_mom_conv
from pyscf.tools import molden
from pyscf.tools import cubegen

# ------------------------------------------------------------ #
# Units
# ------------------------------------------------------------ #
# 1 a.u. = 2.541746 Debye    (for Transition dipole)
# 1 a.u. = 27.211408 eV      (for Energy)
har_to_eV = 27.211408        #(eV)
har_to_D = 2.541746          #(Debye)

# ------------------------------------------------------------ #
# Hàm con Vẽ orbital file .cube và file .vti/.vtk
# ------------------------------------------------------------ #
def export_orbitals_to_cube(
    mf,                 # Đối tượng Mean Field (đã chạy scf)
    mol,                # Đối tượng Mole
    mol_name,           # Tên phân tử (string) dùng đặt tên file
    job_label,          # Nhãn công việc (VD: "GS", "ES")
    target_indices=None,# Danh sách các MO cụ thể muốn vẽ

    # --- CÁC LỰA CHỌN VẼ (True/False) ---
    export_all=False,       # Vẽ TOÀN BỘ MO từ 0 đến hết
    export_range=False,     # Vẽ từ MO 0 đến MO cao nhất trong target_indices
    export_homo_lumo=False, # Tự động tìm và vẽ HOMO/LUMO
    export_specific=True    # Chỉ vẽ các MO trong target_indices
):
    """
    Hàm xuất file .cube cho các Orbital phân tử (MO).
    Đã loại bỏ chế độ vẽ trung bình spin.
    """

    # 1. Khởi tạo và xử lý đầu vào
    if target_indices is None: target_indices = []
    output_dir = "art-orb-cube"
    os.makedirs(output_dir, exist_ok=True)

    # Lấy hệ số và độ chiếm đóng
    mo_coeffs = mf.mo_coeff
    mo_occs = mf.mo_occ

    # Kiểm tra UHF hay RHF
    is_uhf = (len(mo_coeffs.shape) == 3)
    num_spins = 2 if is_uhf else 1

    # Chuẩn hóa input thành dạng list để dễ loop
    if not is_uhf:
        mo_coeffs = [mo_coeffs]
        mo_occs = [mo_occs]
        spin_names = ['RHF']
    else:
        spin_names = ['Alpha', 'Beta']

    nmo = mo_coeffs[0].shape[1] # Tổng số MO

    # 2. Cấu hình lưới (Grid) - dx ~ 0.2 Bohr
    GRID_SIZE = (35, 44, 46)
    print(f"\n--- Exporting Cubes: {mol_name} | {job_label} ---")

    # Hàm nội bộ lưu file
    def _save_cube(tag, data_vec):
        filename = f"{mol_name}_{job_label}_{tag}.cube"
        filepath = os.path.join(output_dir, filename)
        cubegen.orbital(mol, filepath, data_vec,
                        nx=GRID_SIZE[0], ny=GRID_SIZE[1], nz=GRID_SIZE[2])
        print(f" Saved: {filename}")

    # 3. Vòng lặp chính qua từng Spin
    for s_idx in range(num_spins):
        spin_name = spin_names[s_idx]
        spin_tag = f"_{spin_name}" if is_uhf else ""

        c_matrix = mo_coeffs[s_idx] # Ma trận hệ số
        o_vector = mo_occs[s_idx]   # Vector chiếm đóng

        print(f" Processing Spin: {spin_name}...")

        # --- CHẾ ĐỘ 1: XUẤT TẤT CẢ (FULL) ---
        if export_all:
            for i in range(nmo):
                occ = o_vector[i]
                tag = f"FULL{spin_tag}_MO_{i:03d}_occ_{occ:.3f}"
                _save_cube(tag, c_matrix[:, i])
            continue # Xong spin này, chuyển spin sau

        # Xác định phạm vi cho chế độ Range
        max_target = max(target_indices) if target_indices else -1
        limit_idx = min(max_target, nmo - 1)

        # --- CHẾ ĐỘ 2: XUẤT THEO DẢI (RANGE) ---
        if export_range and limit_idx >= 0:
            for i in range(limit_idx + 1):
                occ = o_vector[i]
                status = 'occ' if occ > 1e-3 else 'virt'
                tag = f"RANGE{spin_tag}_MO_{i:03d}_{status}_{occ:.3f}"
                _save_cube(tag, c_matrix[:, i])

        # --- CHẾ ĐỘ 3: HOMO / LUMO ---
        if export_homo_lumo:
            occ_idx = np.where(o_vector > 1e-3)[0]
            if len(occ_idx) > 0:
                homo = occ_idx.max()
                lumo = homo + 1

                # HOMO
                tag_h = f"HOMO{spin_tag}_MO_{homo:03d}_occ_{o_vector[homo]:.3f}"
                _save_cube(tag_h, c_matrix[:, homo])

                # LUMO
                if lumo < nmo:
                    tag_l = f"LUMO{spin_tag}_MO_{lumo:03d}_occ_{o_vector[lumo]:.3f}"
                    _save_cube(tag_l, c_matrix[:, lumo])

        # --- CHẾ ĐỘ 4: CÁC MO CỤ THỂ (SPECIFIC) ---
        if export_specific and target_indices:
            for i in target_indices:
                if 0 <= i < nmo:
                    # Tránh trùng lặp nếu đã vẽ ở chế độ Range
                    if export_range and i <= limit_idx:
                        continue

                    occ = o_vector[i]
                    tag = f"SPEC{spin_tag}_MO_{i:03d}_occ_{occ:.3f}"
                    _save_cube(tag, c_matrix[:, i])


# ------------------------------------------------------------ #
# Hàm con Tính P_uv
# ------------------------------------------------------------ #

def charge_center(mol):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    return np.einsum('z,zr->r', charges, coords)/charges.sum()

def pseudo_solve_transpose(oT, B, svd_threshold=1e-5):
    # Giải oT·X = B an toàn (nếu oT gần kỳ dị thì dùng pseudo-inverse) #
    try:
        X = np.linalg.solve(oT, B)
    except np.linalg.LinAlgError:
        u, s, vt = scipy.linalg.svd(oT, full_matrices=False)
        s_inv = np.array([1.0/x if x > svd_threshold else 0.0 for x in s])
        X = (vt.T * s_inv) @ u.T @ B
    return X

def calc_transition_density_term(s_ao, c_m_occ, c_n_occ, svd_threshold=1e-5, gsc_threshold=1e-3):
    """
    Tính ma trận mật độ chuyển tiếp (P_trans) cho một spin.
    Sử dụng SVD-stabilized pseudo-inverse (DML) HOẶC GSC nếu phát hiện 1 khác biệt.

    Returns: det_val, p_trans, mode_used ('GSC' hoặc 'DML')
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

    # Phát hiện 1-khác biệt: Tìm singular value gần 0 (giá trị kỳ dị nhỏ nhất)
    # Lấy chỉ số của singular value nhỏ nhất
    idx_diff = np.argmin(s)
    s_min = s[idx_diff]

    # Kiểm tra điều kiện GSC: Có 1 giá trị s nhỏ hơn gsc_threshold
    # VÀ N-1 giá trị còn lại lớn (gần 1)

    # Tích các s còn lại (Spectator overlap)
    s_spectator = np.prod(s[np.arange(N_occ) != idx_diff])

    # Phân tích GSC:
    if s_min < gsc_threshold and np.isclose(s_spectator, 1.0, atol=gsc_threshold**0.5):

        # TRƯỜNG HỢP 1: PHÁT HIỆN GSC (1 khác biệt) -> Dùng công thức GSC ổn định

        # Orbital bị hủy (i) trong cơ sở AO (từ tập m) - tương ứng với vector riêng trái U
        phi_i = c_m_occ @ u[:, idx_diff]

        # Orbital được tạo (j) trong cơ sở AO (từ tập n) - tương ứng với vector riêng phải V
        v_j = vt[idx_diff, :].conj()
        phi_j = c_n_occ @ v_j

        # Tính P_trans GSC: P_trans = (Phase * S_spectator) * |i><j|
        p_trans = (phase * s_spectator) * np.outer(phi_i, phi_j.conj())
        mode = "GSC"

    else:
        # TRƯỜNG HỢP 2: DML (Dùng cho nhiều khác biệt hoặc tính toán ổn định)

        # 3. Tính nghịch đảo giả (O^-1) bằng SVD-stabilized pseudo-inverse
        s_inv = np.where(s > svd_threshold, 1.0 / s, 0.0)
        x_matrix = vt.conj().T @ np.diag(s_inv) @ u.conj().T

        # 4. Tính P_trans trên cơ sở AO: P_trans = C_gs @ O^-1 @ C_es.T
        p_trans = c_m_occ @ x_matrix @ c_n_occ.conj().T
        mode = "DML"

    return det_val, p_trans, mode, s_spectator


def contract_tdm_ao(p_trans, r_ao):
    # Thực hiện phép co giữa Ma trận mật độ chuyển tiếp (P_trans) và Tích phân Dipole AO (r_ao).
    term_vec = np.zeros(3)
    for i in range(3): # Loop qua x, y, z
        term_vec[i] = np.einsum('ij,ji->', p_trans, r_ao[i])
    return term_vec


# ------------------------------------------------------------ #
# Hàm con tính transition dipole
# (Tuy nhiên em không dùng trực tiếp trong main mà code lại tương tự
#  vì em muốn coi nhiều thông số trong quá trình tính toán để kiểm chứng
#  hàm sóng có bị sụp về GS hay không)
# ------------------------------------------------------------ #
def calc_tdm(mol, mf_gs, mf_es, svd_threshold=1e-5, gsc_threshold=1e-3):
    """
    Tính TDM giữa hai trạng thái UHF không trực giao (OB-MP2)
    sử dụng DML hoặc GSC dựa trên phát hiện SVD.
    """
    # 1. Lấy các tích phân AO cần thiết
    s_ao = mol.intor('int1e_ovlp')
    r_ao = mol.intor('int1e_r')

    # 2. Lấy số lượng electron alpha và beta
    nelec_a, nelec_b = mol.nelec

    # 3. Trích xuất hệ số MO bị chiếm
    ca_gs_occ = mf_gs.mo_coeff[0][:, :nelec_a]
    ca_es_occ = mf_es.mo_coeff[0][:, :nelec_a]
    cb_gs_occ = mf_gs.mo_coeff[1][:, :nelec_b]
    cb_es_occ = mf_es.mo_coeff[1][:, :nelec_b]

    # 4. Xử lý Spin Alpha và Beta bằng hàm ổn định
    # Cập nhật: Hàm trả về det, P_trans, và mode
    det_a, p_trans_a, mode_a, sa_spectator = calc_transition_density_term(s_ao, ca_gs_occ, ca_es_occ, svd_threshold, gsc_threshold)
    det_b, p_trans_b, mode_b, sb_spectator = calc_transition_density_term(s_ao, cb_gs_occ, cb_es_occ, svd_threshold, gsc_threshold)

    # 5. Phần electron:
    term_a = contract_tdm_ao(p_trans_a, r_ao)
    term_b = contract_tdm_ao(p_trans_b, r_ao)
    tdm_elec = -1.0 * ((det_b * term_a) + (det_a * term_b))

    # 6. Phần hạt nhân:
    charges = mol.atom_charges()
    coords_bohr = mol.atom_coords(unit='Bohr')
    nucl_dip_vec = np.einsum('i,ix->x', charges, coords_bohr)
    total_overlap = det_a * det_b
    tdm_nucl = nucl_dip_vec * total_overlap

    # 7. Tổng hợp TDM cuối cùng:
    tdm_spinor = tdm_elec + tdm_nucl
    tdm = 1/np.sqrt(2) * tdm_spinor * 2
    tdm_sq = np.sum(tdm**2)

    return tdm, tdm_sq


# ------------------------------------------------------------ #
# Hàm con Dipole cả phân tử và từng orbital (dipole + energy)
# (Em xuất các thông tin này để đối chiếu với thực nghiệm khi giải một số molecule
#  như CO bằng OB-MP2, đồng thời để coi suy biến)
# ------------------------------------------------------------ #
def dipole_from_rdm(mf_obj, rx, label):
    """
    Tính dipole phân tử từ 1-RDM:
        dip_elec  = -Tr[γ r]
        dip_nuc   = Σ Z_A R_A
        dip_total = dip_elec + dip_nuc
    """
    # Tạo 1-RDM và ẩn Warning của đoạn này
    old_verbose = mf_obj.verbose
    mf_obj.verbose = 0
    rdm_ao = mf_obj.make_rdm1(use_t2=True, use_ao=True)
    mf_obj.verbose = old_verbose

    # ----- Electronic dipole: d_elec = -Tr[γ r]
    rdm_total = rdm_ao[0] + rdm_ao[1]
    dip_elec = -np.array([np.trace(rdm_total @ rx_i) for rx_i in rx])
    dip_elec_mag = np.linalg.norm(dip_elec)

    # ----- Nuclear dipole: d_nuc = Σ Z_A R_A
    dip_nuc = np.zeros(3)
    for A in range(mf_obj.mol.natm):
        Z = mf_obj.mol.atom_charge(A)
        R = np.array(mf_obj.mol.atom_coord(A))
        dip_nuc += Z * R
    dip_nuc_mag = np.linalg.norm(dip_nuc)

    # ----- Total dipole
    dip_total = dip_elec + dip_nuc
    dip_total_mag = np.linalg.norm(dip_total)

    # ----- Print results
    print(f"\n⟨d⟩ ({label})")
    print(f"  Electronic dipole  d_elec = "
        f"[{dip_elec[0]:12.6f}  {dip_elec[1]:12.6f}  {dip_elec[2]:12.6f}]    "
        f"|d_elec|  = {dip_elec_mag:.8f} a.u. = {dip_elec_mag*har_to_D:.8f} D")
    print(f"  Nuclear dipole     d_nuc  = "
        f"[{dip_nuc[0]:12.6f}  {dip_nuc[1]:12.6f}  {dip_nuc[2]:12.6f}]    "
        f"|d_nucl|  = {dip_nuc_mag:.8f} a.u. = {dip_nuc_mag*har_to_D:.8f} D")
    print(f"  Total dipole       d_tot  = "
        f"[{dip_total[0]:12.6f}  {dip_total[1]:12.6f}  {dip_total[2]:12.6f}]    "
        f"|d_total| = {dip_total_mag:.8f} a.u. = {dip_total_mag*har_to_D:.8f} D\n")

    return dip_elec, dip_nuc, dip_total


def dipole_orbitals(mf_obj, rx, label):
    """
    In KQ tính Energy + Dipole của từng orbital phân tử để kiểm tra nhanh suy biến
    """
    print(f"\n" + "="*80)
    print(f" ORBITAL PROPERTIES: {label}")
    print("=" * 80)

    # Lấy dữ liệu
    mo_coeffs = getattr(mf_obj, 'mo_coeff', mf_obj._scf.mo_coeff)
    mo_energies = getattr(mf_obj, 'mo_energy', mf_obj._scf.mo_energy)
    mo_occs = getattr(mf_obj, 'mo_occ', getattr(mf_obj._scf, 'mo_occ', None))

    nmo = mo_coeffs[0].shape[1]
    spin_names = ['Alpha', 'Beta']

    # --- 1. In chi tiết từng spin ---
    for s_idx in range(2):
        print(f"\n Spin: {spin_names[s_idx]}")
        # Tiêu đề bảng
        print(f"{'MO':>4} | {'Occ':>6} | {'Energy (a.u.)':>15} | {'Dipole <r> (x, y, z) a.u.':>35}")
        print("-" * 80)

        C = mo_coeffs[s_idx]
        E = mo_energies[s_idx]
        Occ = mo_occs[s_idx]

        for i in range(nmo):
            # Tính dipole cho từng orbital
            r_val = [- C[:, i].T @ rx[comp] @ C[:, i] for comp in range(3)]

            # Căn chỉnh: dùng :>8.4f để các số có độ rộng 8, thẳng hàng thập phân
            print(f"{i:>4} | {Occ[i]:>6.2f} | {E[i]:>15.4f} | [{r_val[0]:>8.4f}, {r_val[1]:>8.4f}, {r_val[2]:>8.4f}]")

    # --- 2. In bảng kết hợp trung bình ---
    print(f"\n" + "="*80)
    print(f"COMBINED (AVERAGED) PROPERTIES: {label}")
    print("=" * 80)
    print(f"{'MO':>4} | {'Occ':>6} | {'Avg Energy':>15} | {'Avg Dipole <r> (x, y, z)':>35}")
    print("-" * 80)

    for i in range(nmo):
        # Tính dipole từng spin để trung bình
        r_a = np.array([- mo_coeffs[0][:, i].T @ rx[comp] @ mo_coeffs[0][:, i] for comp in range(3)])
        r_b = np.array([- mo_coeffs[1][:, i].T @ rx[comp] @ mo_coeffs[1][:, i] for comp in range(3)])

        occ_sum = mo_occs[0][i] + mo_occs[1][i]
        e_avg   = (mo_energies[0][i] + mo_energies[1][i]) / 2.0
        r_avg   = (r_a + r_b) / 2.0

        print(f"{i:>4} | {occ_sum:>6.2f} | {e_avg:>15.4f} | [{r_avg[0]:>8.4f}, {r_avg[1]:>8.4f}, {r_avg[2]:>8.4f}]")



#=============================================================================================================================================

# ------------------------------------------------------------ #
# Main calculation
# ------------------------------------------------------------ #

# --- Molecule setup ---
print("\n" + "="*30 + " MOLECULE SETUP " + "="*30)
Molecule = '02-H2O-H2O'
mol = gto.M(
    atom='''
        O    -0.066999140    0.000000000    1.494354740
        H     0.815734270    0.000000000    1.865866390
        H     0.068855100    0.000000000    0.539142770
        O     0.062547750    0.000000000   -1.422632080
        H    -0.406965400   -0.760178410   -1.771744500
        H    -0.406965400    0.760178410   -1.771744500
    ''',
    basis='cc-pVDZ',
    unit='A',
    spin=0,
    verbose=0)
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
print(f"\n=== Ground state (SCF + OB-MP2) ===")

hf_gs = scf.UHF(mol)
hf_gs.scf()
mo0_gs = hf_gs.mo_coeff
occ_gs = hf_gs.mo_occ

# OB-MP2 GS
ob_gs = mp.uobmp2_mom_conv.UOBMP2(hf_gs)
ob_gs.second_order = True
ob_gs.mom_select = True
ob_gs.thresh = 1e-8

# print("> GS: Running pre-conditioning (css=0, cos=1.1)...")
# ob_gs.css = 0
# ob_gs.cos = 1.1
# ob_gs.kernel()

# print("> GS: Running pre-conditioning (css=0, cos=1.2)...")
# ob_gs.css = 0
# ob_gs.cos = 1.2
# ob_gs.kernel()

# print("> GS: Running pre-conditioning (css=0, cos=1.3)...")
# ob_gs.css = 0
# ob_gs.cos = 1.3
# ob_gs.kernel()

print("> GS: Running final calculation (css=1, cos=1.0)...")
ob_gs.css = 1
ob_gs.cos = 1
e_obmp2_gs = ob_gs.kernel()

# Lưu kết quả GS
calc_results["GS"] = ob_gs
state_energies["GS"] = e_obmp2_gs

# Check Info & Dipole
dip_gs = dipole_from_rdm(ob_gs, rx, "GS")
n_ao_gs = ob_gs.mo_coeff[0].shape[0]
n_mo_gs = ob_gs.mo_coeff[0].shape[1]
print(f"[GS] AO = {n_ao_gs}, MO = {n_mo_gs}")
dipole_orbitals(ob_gs, rx, label='GS')

# ======================================================================
# 2. EXCITED STATES CALCULATION (LOOP)
# ======================================================================
# --- CẤU HÌNH CÁC TRẠNG THÁI (Dynamic Injection) ---
transitions = [(9, 11, 'ES1'), (8, 10, 'ES2'), (9, 10, 'ES3'), (8, 12, 'ES4'), (7, 10, 'ES5'), (6, 10, 'ES6')]
es_summary = []
all_active_orbitals = []

for hole, electron, label in transitions:
    all_active_orbitals.extend([hole, electron])

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
    hf = mp.hf_mom_conv.UOBMP2(scf_mom).run()

    print(f"\n=== Excited states (MOM + OB-MP2) ===")
    print(f"\n--- Processing {label}: Hole {hole} -> Elec {electron} ---")

    # 3. OB-MP2
    ob_es = mp.uobmp2_mom_conv.UOBMP2(scf_mom)
    ob_es.second_order = True
    ob_es.mom_select = True
    ob_es.thresh = 1e-8
    ob_es.niter = 500

    # print(f"  -> {label}: Step 1 (css=0, cos=1.1)...")
    # ob_es.css = 0
    # ob_es.cos = 1.1
    # ob_es.kernel()

    # print(f"  -> {label}: Step 2 (css=0, cos=1.2)...")
    # ob_es.css = 0
    # ob_es.cos = 1.2
    # ob_es.kernel()

    # print(f"  -> {label}: Step 3 (css=0, cos=1.3)...")
    # ob_es.css = 0
    # ob_es.cos = 1.3
    # ob_es.kernel()

    print(f"  -> {label}: Final optimization (css=1, cos=1.0)...")
    ob_es.css = 1
    ob_es.cos = 1
    e_obmp2 = ob_es.kernel()    # Lấy năng lượng cuối cùng tại đây

    # 4. Lưu đối tượng tính toán
    calc_results[label] = ob_es

    # 5. Get Info for Summary
    dip_vals = dipole_from_rdm(ob_es, rx, label)
    n_ao = ob_es.mo_coeff[0].shape[0]
    n_mo = ob_es.mo_coeff[0].shape[1]

    es_summary.append({
        "label": label, "hole": hole, "elec": electron,
        "energy": e_obmp2, "dipole": dip_vals, "n_ao": n_ao, "n_mo": n_mo})



# ===================== EXPORTING MOLECULAR ORBITALS =====================
print("\n" + "="*20 + " EXPORTING MOLECULAR ORBITALS " + "="*20)
export_settings = {
    "export_all": False,
    "export_range": False,
    "export_homo_lumo": False,
    "export_specific": False}

# 1. Export GS (Vẽ tất cả các orbital có tham gia vào các quá trình kích thích)
unique_active_mos = sorted(list(set(all_active_orbitals)))
print(f"Exporting GS with active orbitals: {unique_active_mos}")
export_orbitals_to_cube(ob_gs, mol, Molecule, job_label="GS", target_indices=unique_active_mos, **export_settings)

# 2. Export Excited States (Loop dynamic)
for res in es_summary:
    lbl = res["label"]
    target_mos = [res["hole"], res["elec"]]
    print(f"Exporting {lbl} with orbitals: {target_mos}")
    current_ob = calc_results[lbl]
    export_orbitals_to_cube(current_ob, mol, Molecule, job_label=lbl, target_indices=target_mos, **export_settings)



# ===================== SUMMARY & COMPARISON =====================
print("\n" + "="*20 + " ENERGIES & DIPOLES " + "="*20)

# 1. Xử lý Ground State (GS)
gs_dip_total_mag = np.linalg.norm(dip_gs[2])
print(f"GS  Energy: {e_obmp2_gs:10.8f} a.u. | Dipole Total: {gs_dip_total_mag:.4f} a.u.")
print("-" * 60)

# 2. Xử lý Excited States (ES)
for res in es_summary:
    delta_ev = (res['energy'] - e_obmp2_gs) * har_to_eV
    es_dip_vector = res['dipole'][2]
    # Tính độ lớn
    es_dip_mag = np.linalg.norm(es_dip_vector)
    print(f"[{res['label']}] Hole {res['hole']} -> Elec {res['elec']}")
    print(f"      Energy  : {res['energy']:10.8f} a.u. (ΔE = {delta_ev:10.4f} eV)")
    print(f"      Dipole  : {es_dip_mag:.4f} a.u.") # In độ lớn thay vì in vector
    print("-" * 60)


# ======================================================================
# TRANSITION DIPOLE MOMENTS
# ======================================================================
# Cấu hình danh sách cặp trạng thái (GS -> ES)
state_objs = calc_results
for res in es_summary:
    state_energies[res['label']] = res['energy']

pairs = [("GS", item[2]) for item in transitions]

# Gán các giá trị tham số
SVD_THRESHOLD = 1e-5
GSC_THRESHOLD = 0.05

# 1. Chuẩn bị các tích phân cơ bản
# LƯU Ý: Không cần set common origin nữa vì ta sẽ cộng dipole hạt nhân vào
mol = ob_gs.mol
S = mol.intor('int1e_ovlp')
rx_tdm = mol.intor('int1e_r', comp=3)       # Tính từ gốc tọa độ mặc định (0,0,0)

# 2. Tính Vector Dipole Hạt nhân =  Sum(Z_i * R_i)
charges = mol.atom_charges()
coords_bohr = mol.atom_coords(unit='Bohr')
nucl_dip_vec_au = np.einsum('i,ix->x', charges, coords_bohr) # (Sửa tên biến cho khớp)

# 3. Lấy số lượng electron
nelec_a, nelec_b = mol.nelec

# 4. Vòng lặp tính toán TDM
for m, n in pairs:
    print("\n" + "="*70)
    print(f" TRANSITION: {m} <- {n}")
    print("="*70)

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
    det_a, p_trans_a, mode_a, sa_spectator = calc_transition_density_term(S, c_m_a_occ, c_n_a_occ, SVD_THRESHOLD, GSC_THRESHOLD)
    det_b, p_trans_b, mode_b, sb_spectator = calc_transition_density_term(S, c_m_b_occ, c_n_b_occ, SVD_THRESHOLD, GSC_THRESHOLD)

    # Tổng hợp các thông số chẩn đoán
    det_ovlpMO = det_a * det_b              # Đây chính là S_12 (Total Overlap)

    # In thông tin chẩn đoán
    print(f"Overlap Alpha (Det)          : {det_a: .8f} ({mode_a})")
    print(f"Overlap Beta (Det)           : {det_b: .8f} ({mode_b})")
    print(f"sa_spectator                 : {sa_spectator: .8f}")
    print(f"sb_spectator                 : {sb_spectator: .8f}")
    print(f"Total Wavefunction Overlap   : {det_ovlpMO: .8f}")
    print("Max values of Transition Density matrices:")
    print(f"  max|P_trans_a| = {np.max(np.abs(p_trans_a)):>12.8f}")
    print(f"  max|P_trans_b| = {np.max(np.abs(p_trans_b)):>12.8f}")

    # --- TÍNH TOÁN TRANSITION DIPOLE ---

    # A. Phần điện tử (Electronic Component)
    # Tính Trace(P * r). rx_tdm là vector vị trí r.
    term_a = contract_tdm_ao(p_trans_a, rx_tdm)
    term_b = contract_tdm_ao(p_trans_b, rx_tdm)
    mu_elec = -1.0 * ((det_b * term_a) + (det_a * term_b))

    # B. Phần hạt nhân (Nuclear Component)
    # Sửa tên biến nuc_dip_vec_au -> nucl_dip_vec_au
    mu_nucl = nucl_dip_vec_au * det_ovlpMO

    # C. Tổng Dipole Raw (Chưa scale spin)
    mu_total_raw = mu_elec + mu_nucl

    # D. Scale theo giả định Singlet
    # 1/sqrt(2) là chuẩn hóa hàm sóng Singlet, * 2 là tổng 2 spin
    mu_trans = 1/np.sqrt(2) * mu_total_raw * 2

    # Tính bình phương độ lớn (Dipole Strength)
    mu_sq = np.sum(mu_trans**2)
    mu_mag = np.sqrt(mu_sq)

    print("\nTransition dipole components (a.u. - unscaled):")
    print(f"  Electronic contribution: [{mu_elec[0]:.6f}  {mu_elec[1]:.6f}  {mu_elec[2]:.6f}]")
    print(f"  Nuclear contribution:    [{mu_nucl[0]:.6f}  {mu_nucl[1]:.6f}  {mu_nucl[2]:.6f}]")

    print("\nFinal Transition dipole (a.u.) [Scaled for Singlet]:")
    print(f"  μ_x,y,z = [{mu_trans[0].real:12.6f}  {mu_trans[1].real:12.6f}  {mu_trans[2].real:12.6f}]")
    print(f"  |μ|     = {mu_mag:.8f}")

    # Energy difference
    deltaE = E_n - E_m

    # Oscillator strength (length gauge)
    f_length = (2.0/3.0) * deltaE * mu_sq

    print("\nOscillator strength (length gauge):")
    print(f" ΔE ({n}-{m})  = {deltaE: .8f} a.u.   ({deltaE*har_to_eV:.8f} eV)")
    print(f" |μ|^2        = {mu_sq: .8f}  a.u.")
    print(f" f_length     = {f_length: .8f}")
