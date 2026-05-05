import numpy as np
from pyscf.pbc import gto, scf
import pywannier90
import copy
from pycmf.OBMP import kobmp2
from pycmf.OBDF import krobdf
from pyscf.pbc.lib import kpts_helper
from pyscf import fci
from pyscf.pbc import df

# ============================================================
# 1. Build cell
# ============================================================

cell = gto.Cell()
cell.atom = """
H 1.5 1.5 1
H 1.5 1.5 2
"""
cell.basis = '6-31g'
# cell.basis = 'gth-szv'
# cell.pseudo = 'gth-pade'
cell.a = np.eye(3) * 50
# cell.dimension = 0s
cell.gs = [100, 100, 100]
cell.build()


nk = [1, 1, 1]
kpts = cell.make_kpts(nk)

# ============================================================
# 2. KRHF
# ============================================================

# kmf = scf.KRHF(cell, kpts).run()

kmf = scf.KRHF(cell, kpts)
kmf.with_df = df.GDF(cell, kpts)  # Sử dụng Gaussian Density Fitting
kmf.run()

# ============================================================
# 3. kOBMP2
# ============================================================

# khf = copy.copy(kmf)
khf = kmf

krobmp = kobmp2.OBMP2(khf)
krobmp.second_order = True
krobmp.kernel()

# ============================================================
# 4. Downfold
# ============================================================

krobact = krobdf.OBMP2(khf)
krobact.nact = 2
krobact.nocc_act = 1
krobact.mo_coeff = krobmp.mo_coeff
krobact.mo_energy = krobmp.mo_energy
krobact.fock_hf = krobmp.fock_hf
krobact.second_order = True

import pycmf.OBDF.krobdf as krobdf_module

# Lưu lại hàm gốc để dùng cho các hệ có electron lõi
original_make_veff_core = krobdf_module.make_veff_core


def patched_make_veff_core(mp):
    # KROBDF thường lưu ncore dưới dạng list cho [alpha, beta]
    ncore_total = sum(mp.ncore) if hasattr(mp.ncore, '__iter__') else mp.ncore
    if ncore_total == 0:
        # Bỏ qua get_jk, trả về V_core = 0 cho cả 2 kênh spin
        return [0.0, 0.0]
    return original_make_veff_core(mp)


# Áp dụng bản vá
krobdf_module.make_veff_core = patched_make_veff_core

krobact.kernel()

h1_eff_k = krobact.h1mo_act_eff
h2_k = krobact.h2mo_act

Nk_total, Norb, _ = h1_eff_k.shape
Nkx, Nky, Nkz = nk
Nk = Nkx * Nky * Nkz

print('# ============================================================')
print('# 5. FCI (KHÔNG WANNIER) để kiểm chứng')
print('# ============================================================')
print('\n' + '=' * 40)
print('TÍNH TOÁN FCI TRÊN HAMILTONIAN DOWNFOLD')
print('=' * 40)

# 1. Trích xuất Hamiltonian tại điểm Gamma và CHUYỂN SANG SỐ THỰC
# (Bắt buộc phải dùng .real vì fci.direct_spin1 không nhận số phức)
h1_fci = h1_eff_k[0].real
h2_fci = h2_k[0, 0, 0, 0].real

# 2. Thiết lập không gian Active
norb_act = krobact.nact
nelecas = (krobact.nocc_act, krobact.nocc_act)  # (số e alpha, số e beta) = (1, 1) cho H2

# 3. Khởi tạo FCI solver
cis = fci.direct_spin1.FCI()
cis.nroots = 1  # Chỉ lấy trạng thái cơ bản

# 4. Giải FCI
e_fci_act, wfci = cis.kernel(h1_fci, h2_fci, norb_act, nelecas)

print(f'> E_FCI_active (Gamma-point PBC) = {e_fci_act:.10f} Hartree')

# ============================================================
# In ma trận để so sánh trực tiếp với bản Phân tử
# ============================================================
print('\n> So sánh Ma trận h1_eff (Gamma) với phân tử:')
print(h1_fci)

# Note: In ra phần tử đặc trưng của h2 để đối chiếu
print(f'\n> Kiểm tra h2_eff[0,0,0,0]: {h2_fci[0, 0, 0, 0]:.8f}')

# ============================================================
# 5. Wannier & Gauge Transformation
# ============================================================
print('\n# ============================================================')
print('# 5. Wannier90 và Kẹp U (Gauge Transformation)')
print('# ============================================================')

kmf.mo_coeff = krobmp.mo_coeff.copy()
kmf.mo_energy = krobmp.mo_energy.copy()

w90 = pywannier90.W90(
    kmf,
    cell,
    nk,
    2,
    other_keywords="""
exclude_bands : 3,4
begin projections
H:s
end projections
""",
)
w90.kernel()

U = w90.U_matrix

# ---- Biến đổi Gauge cho h1 và h2 tại k (Wannier Gauge in k-space)
h1_wann_k = np.zeros_like(h1_eff_k, dtype=complex)
h2_wann_k = np.zeros_like(h2_k, dtype=complex)
kconserv = kpts_helper.get_kconserv(cell, kpts)
N1, N2, N3, N4, n, _, _, _ = h2_k.shape

for k in range(Nk_total):
    h1_wann_k[k] = U[k].conj().T @ h1_eff_k[k] @ U[k]

for kp in range(N1):
    for kq in range(N2):
        for kr in range(N3):
            ks = kconserv[kp, kq, kr]
            Up, Uq, Ur, Us = U[kp], U[kq], U[kr], U[ks]
            eri = h2_k[kp, kq, kr, ks]
            h2_wann_k[kp, kq, kr, ks] = np.einsum(
                'ai,bj,abcd,ck,dl->ijkl', Up.conj(), Uq.conj(), eri, Ur, Us, optimize=True
            )

# ------------------------------------------------------------
# CHECKPOINT 1: TEST FCI SAU KHI KẸP U (TRONG KHÔNG GIAN K)
# ------------------------------------------------------------
print('\n--- CHECKPOINT 1: FCI SAU KHI KẸP U (Wannier Gauge, k-space) ---')
h1_chk1 = h1_wann_k[0].real
h2_chk1 = h2_wann_k[0, 0, 0, 0].real

cis_chk1 = fci.direct_spin1.FCI()
cis_chk1.nroots = 1
e_chk1, _ = cis_chk1.kernel(h1_chk1, h2_chk1, norb_act, nelecas)
print(f'> E_FCI_chk1 = {e_chk1:.10f} Hartree (Phải BẰNG E_FCI_active)')

# ============================================================
# 6. Biến đổi Fourier (k -> R)
# ============================================================
print('\n# ============================================================')
print('# 6. Fourier Transform (k -> R)')
print('# ============================================================')

# Xử lý h1
h1_wann_k_grid = np.zeros((Nkx, Nky, Nkz, Norb, Norb), dtype=complex)
iR = 0
for ix in range(Nkx):
    for iy in range(Nky):
        for iz in range(Nkz):
            h1_wann_k_grid[ix, iy, iz] = h1_wann_k[iR]
            iR += 1

h1k_shift = np.fft.ifftshift(h1_wann_k_grid, axes=(0, 1, 2))
h1_R = np.fft.fftn(h1k_shift, axes=(0, 1, 2)) / Nk
h1_gamma = h1_R[0, 0, 0]

# Xử lý h2
h2_R = np.fft.fftn(h2_wann_k, axes=(0, 1, 2)) / Nk
h2_gamma = h2_R[0, 0, 0, 0]

# ------------------------------------------------------------
# CHECKPOINT 2: TEST FCI SAU KHI FOURIER (TRONG KHÔNG GIAN R)
# ------------------------------------------------------------
print('\n--- CHECKPOINT 2: FCI SAU KHI FOURIER (Real Space R) ---')
# Đánh giá phản biện: Kiểm tra độ lớn của phần ảo trước khi ném đi
max_imag_h1 = np.max(np.abs(h1_gamma.imag))
max_imag_h2 = np.max(np.abs(h2_gamma.imag))
print(f'> Max phần ảo của h1(R): {max_imag_h1:.2e} (Cần rất gần 0)')
print(f'> Max phần ảo của h2(R): {max_imag_h2:.2e} (Cần rất gần 0)')

h1_chk2 = h1_gamma.real
h2_chk2 = h2_gamma.real

cis_chk2 = fci.direct_spin1.FCI()
cis_chk2.nroots = 1
e_chk2, _ = cis_chk2.kernel(h1_chk2, h2_chk2, norb_act, nelecas)
print(f'> E_FCI_chk2 = {e_chk2:.10f} Hartree (Phải BẰNG E_FCI_chk1)')

# ============================================================
# Bước 7 & 8: Tính tổng năng lượng cuối cùng
# ============================================================
print('\n# ============================================================')
print('# TỔNG KẾT NĂNG LƯỢNG HỆ THỐNG')
print('# ============================================================')

ene_inact = getattr(krobact, 'ene_inact', 0.0)
if isinstance(ene_inact, (list, np.ndarray)):
    ene_inact = ene_inact[0].real

e_tot_final = e_chk2 + ene_inact

print('\n' + '=' * 50)
print(f'Năng lượng Active Space (Bước 1 - Bloch)  : {e_fci_act:.8f}')
print(f'Năng lượng Active Space (Bước 2 - kẹp U)  : {e_chk1:.8f}')
print(f'Năng lượng Active Space (Bước 3 - Fourier): {e_chk2:.8f}')
print(f'Năng lượng Tổng toàn hệ (FCI final)       : {e_tot_final:.8f} Hartree')
print('=' * 50 + '\n')

# ============================================================
# ====== 9. Trích xuất ma trận để đối soát ======
# ============================================================
print('\n' + '=' * 50)
print('TRÍCH XUẤT MA TRẬN ĐỐI SOÁT VỚI CODE PHÂN TỬ')
print('=' * 50)

# Trích xuất Hamiltonian hiệu dụng tại R=0 (Gamma) đã được chuyển sang số thực
h1_pbc = h1_chk2
h2_pbc = h2_chk2

print('> Toàn bộ ma trận h1_eff (Hệ PBC - R=0):')
print(h1_pbc)

print('\n> Kiểm tra nhanh một vài phần tử đường chéo của h1_eff:')
for i in range(norb_act):
    print(f'  h1_eff[{i}, {i}] = {h1_pbc[i, i]:.8f}')

print('\n> Kiểm tra nhanh một vài phần tử đặc trưng của h2_eff:')
print(f'  h2_eff[0, 0, 0, 0] = {h2_pbc[0, 0, 0, 0]:.8f}')
if norb_act > 1:
    print(f'  h2_eff[0, 1, 0, 1] = {h2_pbc[0, 1, 0, 1]:.8f}')
    print(f'  h2_eff[1, 1, 1, 1] = {h2_pbc[1, 1, 1, 1]:.8f}')

# Lưu ma trận ra file nhị phân của Numpy để dùng script so sánh tự động
np.save('pbc_h1_eff.npy', h1_pbc)
np.save('pbc_h2_eff.npy', h2_pbc)
print("\n[+] Đã lưu toàn bộ ma trận ra file 'pbc_h1_eff.npy' và 'pbc_h2_eff.npy'")
