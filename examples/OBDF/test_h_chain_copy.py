import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
from pyscf.pbc import gto, df, scf
from pyscf.pbc.lib import kpts_helper
from pyscf import fci
import pywannier90
import copy

from pycmf.OBMP import kobmp2
from pycmf.OBDF import krobdf
import pycmf.OBDF.krobdf as krobdf_module

# ============================================================
# PATCH GLOBAL V_CORE (Chỉ cần định nghĩa 1 lần ngoài hàm)
# ============================================================
original_make_veff_core = krobdf_module.make_veff_core


def patched_make_veff_core(mp):
    ncore_total = sum(mp.ncore) if hasattr(mp.ncore, '__iter__') else mp.ncore
    if ncore_total == 0:
        return [0.0, 0.0]
    return original_make_veff_core(mp)


# krobdf_module.make_veff_core = patched_make_veff_core


def run_1d_hydrogen_point(R_dist, nk_x=4):
    """
    Chạy 1 điểm R_dist và trả về (Tổng năng lượng, Số hạt chiếm chỗ).
    """
    print(f'\n{"=" * 60}')
    print(f'ĐANG TÍNH TOÁN CHO R = {R_dist} a.u. VỚI K-POINTS = [{nk_x}, 1, 1]')
    print(f'{"=" * 60}')

    # 1. Khởi tạo Cell
    cell = gto.Cell()
    cell.unit = 'Bohr'
    cell.atom = f'H 0.0 0.0 0.0\nH {R_dist} 0.0 0.0'
    cell.basis = 'sto-6g'

    # [SỬA LỖI TẠI ĐÂY]: Thiết lập hệ 3 chiều để qua mặt lỗi lưới tích phân
    cell.dimension = 3

    # Giữ nguyên vector mạng với khoảng chân không (vacuum) 20 Bohr ở trục Y và Z.
    # Khoảng cách này đủ lớn để tương tác tĩnh điện giữa các chuỗi kề nhau triệt tiêu,
    # ép hệ thống cư xử hoàn toàn như một chuỗi 1 chiều (1D).
    cell.a = np.array([[2.0 * R_dist, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]])

    cell.verbose = 0  # Tắt log PySCF để tránh trôi màn hình
    cell.build()

    # Lưới K-points [nk_x, 1, 1] đảm bảo chỉ lấy mẫu trong không gian Fourier dọc theo trục X
    nk = [nk_x, 1, 1]
    kpts = cell.make_kpts(nk)
    Nk = nk_x * 1 * 1

    # 2. KRHF & kOBMP2
    kmf = scf.KRHF(cell, kpts)
    kmf.with_df = df.GDF(cell, kpts)
    kmf.run()

    khf = copy.copy(kmf)
    krobmp = kobmp2.OBMP2(khf)
    krobmp.second_order = True
    krobmp.kernel()

    # 3. Downfold
    krobact = krobdf.OBMP2(khf)
    krobact.nact = 2
    krobact.nocc_act = 1
    krobact.mo_coeff = krobmp.mo_coeff
    krobact.mo_energy = krobmp.mo_energy
    krobact.fock_hf = krobmp.fock_hf
    krobact.second_order = True
    krobdf_module.make_veff_core = patched_make_veff_core
    krobact.kernel()

    h1_eff_k = krobact.h1mo_act_eff
    h2_k = krobact.h2mo_act

    Nkx, Nky, Nkz = nk
    Nk_total, Norb, _ = h1_eff_k.shape

    # 4. CHUYỂN ĐỔI CƠ SỞ WANNIER THEO ĐIỀU KIỆN R
    kmf.mo_coeff = krobmp.mo_coeff.copy()
    kmf.mo_energy = krobmp.mo_energy.copy()

    # ====== ĐIỂM QUYẾT ĐỊNH VẬT LÝ ======
    if R_dist <= 2.0:
        wannier_iter = 0  # Natural Wannier
        print('-> Đang sử dụng: NATURAL Wannier Orbitals (num_iter = 0)')
    else:
        wannier_iter = 100  # Regular Wannier (Có thể điều chỉnh)
        print('-> Đang sử dụng: REGULAR Wannier Orbitals (num_iter = 100)')

    keywords = f"""
    num_iter = {wannier_iter}
    begin projections
    H:s
    end projections
    """

    # exclude_bands : 3,4

    w90 = pywannier90.W90(kmf, cell, nk, 2, other_keywords=keywords)
    w90.kernel()
    U = w90.U_matrix

    # 5. Biến đổi Gauge (k-space -> Wannier Gauge)
    h1_wann_k = np.zeros_like(h1_eff_k, dtype=complex)
    for ik in range(Nk_total):
        h1_wann_k[ik] = U[ik].conj().T @ h1_eff_k[ik] @ U[ik]

    h2_wann_k = np.zeros_like(h2_k, dtype=complex)
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    N1, N2, N3, _, _, _, _, _ = h2_k.shape

    for kp in range(N1):
        for kq in range(N2):
            for kr in range(N3):
                ks = kconserv[kp, kq, kr]
                Up, Uq, Ur, Us = U[kp], U[kq], U[kr], U[ks]
                eri = h2_k[kp, kq, kr, ks]
                h2_wann_k[kp, kq, kr, ks] = np.einsum(
                    'ai,bj,abcd,ck,dl->ijkl', Up.conj(), Uq.conj(), eri, Ur, Us, optimize=True
                )

    # 6. Fourier Transform (k -> R)
    h1_wann_k_grid = np.zeros((Nkx, Nky, Nkz, Norb, Norb), dtype=complex)
    iR = 0
    for ix in range(Nkx):
        for iy in range(Nky):
            for iz in range(Nkz):
                h1_wann_k_grid[ix, iy, iz] = h1_wann_k[iR]
                iR += 1

    h1k_shift = np.fft.ifftshift(h1_wann_k_grid, axes=(0, 1, 2))
    h1_R = np.fft.fftn(h1k_shift, axes=(0, 1, 2)) / Nk

    h2_R = np.fft.fftn(h2_wann_k[..., :, :, :, :], axes=(0, 1, 2)) / Nk

    h1_gamma = h1_R[0, 0, 0]
    h2_gamma = h2_R[0, 0, 0, 0]

    # 7. Giải Impurity bằng FCI (tại R = 0)
    h1_fci = h1_gamma.real
    h2_fci = h2_gamma.real
    nelecas = (krobact.nocc_act, krobact.nocc_act)

    cis = fci.direct_spin1.FCI()
    cis.nroots = 1
    e_fci_act, wfci = cis.kernel(h1_fci, h2_fci, krobact.nact, nelecas)

    # 8. Tính Năng lượng Tổng và Số hạt chiếm chỗ (Occupations)
    ene_inact = getattr(krobact, 'ene_inact', 0.0)
    if isinstance(ene_inact, (list, np.ndarray)):
        ene_inact = ene_inact[0].real

    e_tot_final = e_fci_act + ene_inact

    rdm1 = cis.make_rdm1(wfci, krobact.nact, nelecas)
    occupations = np.diag(rdm1)

    return e_tot_final, occupations


def run_scan_and_plot():
    # 1. Định nghĩa dải R và lưới k-points
    # Đã cập nhật lưới giảm 1/3 để chạy an toàn hơn
    production_params = {1.4: 77, 1.8: 63, 2.0: 57, 2.4: 43, 2.8: 37, 3.2: 33, 3.6: 31}

    R_list = np.array(list(production_params.keys()))

    energies = []
    occ_orb1 = []
    occ_orb2 = []

    print('\n' + '#' * 60)
    print('BẮT ĐẦU PRODUCTION-RUN CHO 1D HYDROGEN SOLID')
    print('#' * 60)

    start_time = time.time()

    # 2. Chạy vòng lặp tính toán
    for R in R_list:
        nk_x_prod = production_params[R]
        """
        # KDOWNFOLD KHUYẾN NGHỊ: LUÔN GIỮ TRY...EXCEPT
        try:
            E_tot, occupations = run_1d_hydrogen_point(R, nk_x=nk_x_prod)
            
            energies.append(E_tot)
            occ_sorted = np.sort(occupations) 
            occ_orb1.append(occ_sorted[0]) 
            occ_orb2.append(occ_sorted[1]) 
            
        except Exception as e:
            print(f"\n[!] LỖI NGHIÊM TRỌNG tại R = {R} (k-points = {nk_x_prod}): {e}")
            print("[!] Bỏ qua điểm này, tiếp tục chạy điểm tiếp theo để cứu dữ liệu...\n")
            energies.append(np.nan)
            occ_orb1.append(np.nan)
            occ_orb2.append(np.nan)
        """

        E_tot, occupations = run_1d_hydrogen_point(R, nk_x=nk_x_prod)

        energies.append(E_tot)
        occ_sorted = np.sort(occupations)
        occ_orb1.append(occ_sorted[0])
        occ_orb2.append(occ_sorted[1])

    print(f'\n[+] Đã quét xong! Tổng thời gian: {time.time() - start_time:.2f} giây.')

    # 3. Vẽ đồ thị so sánh
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # [ĐÃ SỬA LỖI NAME_ERROR Ở ĐÂY]
    ax1.plot(R_list, energies, marker='s', linestyle='-', color='teal', label='SEET(FCI) Downfold')
    ax1.set_xlabel('R (H-H) [a.u.]', fontsize=12)
    ax1.set_ylabel('Energy per -(H-H)- unit [a.u.]', fontsize=12)
    ax1.set_title('Figure 2 Reproduction: Total Energy', fontsize=14, pad=15)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11)

    ax2.plot(R_list, occ_orb2, marker='x', linestyle='-', color='seagreen', label='SEET occupied orbital')
    ax2.plot(R_list, occ_orb1, marker='.', linestyle='-', color='darkviolet', label='SEET unoccupied orbital')

    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Strong Correlation Limit (1.0)')

    ax2.set_xlabel('R (H-H) [a.u.]', fontsize=12)
    ax2.set_ylabel('Unit cell occupation numbers', fontsize=12)
    ax2.set_title('Figure 3 Reproduction: Occupation Numbers', fontsize=14, pad=15)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()

    plt.savefig('1D_Hydrogen_SEET_Reproduction.png', dpi=300)
    print("\n[+] Đã lưu đồ thị thành file '1D_Hydrogen_SEET_Reproduction.png'")
    plt.show()
