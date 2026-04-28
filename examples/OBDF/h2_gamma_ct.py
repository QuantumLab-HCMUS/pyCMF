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
# cell.dimension = 0
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
krobact.kernel()

h1_eff_k = krobact.h1mo_act_eff
h2_k = krobact.h2mo_act

Nk_total, Norb, _ = h1_eff_k.shape
Nkx, Nky, Nkz = nk
Nk = Nkx * Nky * Nkz

# ============================================================
# 5. Wannier
# ============================================================

kmf.mo_coeff = krobmp.mo_coeff.copy()
kmf.mo_energy = krobmp.mo_energy.copy()

# ============================================================
# 5. FCI (KHÔNG WANNIER) để kiểm chứng
# ============================================================
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


'''
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

# ---- h1(k) -> Wannier gauge
h1_wann_k = np.zeros_like(h1_eff_k, dtype=complex)
for k in range(Nk_total):
    h1_wann_k[k] = U[k].conj().T @ h1_eff_k[k] @ U[k]

# reshape sang grid
h1_wann_k_grid = np.zeros((Nkx, Nky, Nkz, Norb, Norb), dtype=complex)

iR = 0
for ix in range(Nkx):
    for iy in range(Nky):
        for iz in range(Nkz):
            h1_wann_k_grid[ix, iy, iz] = h1_wann_k[iR]
            iR += 1

# Fourier k -> R
h1k_shift = np.fft.ifftshift(h1_wann_k_grid, axes=(0, 1, 2))
h1_R = np.fft.fftn(h1k_shift, axes=(0, 1, 2)) / Nk

h1_gamma = h1_R[0, 0, 0]

print('\n===== h1(R=Gamma) =====')
print(h1_gamma)


# ============================================================
# 6. h2
# ============================================================

kconserv = kpts_helper.get_kconserv(cell, kpts)

N1, N2, N3, N4, n, _, _, _ = h2_k.shape
h2_wann_k = np.zeros_like(h2_k, dtype=complex)

for kp in range(N1):
    for kq in range(N2):
        for kr in range(N3):
            ks = kconserv[kp, kq, kr]

            Up = U[kp]
            Uq = U[kq]
            Ur = U[kr]
            Us = U[ks]

            eri = h2_k[kp, kq, kr, ks]

            h2_wann_k[kp, kq, kr, ks] = np.einsum(
                'ai,bj,abcd,ck,dl->ijkl', Up.conj(), Uq.conj(), eri, Ur, Us, optimize=True
            )

# Fourier theo 3 index
h2_R = np.fft.fftn(h2_wann_k, axes=(0, 1, 2)) / Nk

h2_gamma = h2_R[0, 0, 0, 0]

print('\n===== h2(R=Gamma) =====')
print(h2_gamma)
'''
