import numpy as np
import time
import warnings
from pyscf.pbc import gto, scf
from pyscf.data import nist
from pycmf.OBMP import kobmp2  # Custom module của bạn

# from pycmf.OBDF import krobdf # Uncomment nếu bạn cần dùng OBDF sau này
# import pywannier90           # Uncomment nếu bạn cần dùng wannier90 sau này
# Tắt các cảnh báo phụ để bảng kết quả dễ nhìn
warnings.filterwarnings('ignore')

# ============================================================
# CẤU HÌNH THÔNG SỐ TEST
# ============================================================
# Năng lượng RHF tham chiếu (eV) để tính độ lệch (Delta E)
E_RHF_REF_EV = -1.09480796286051 * nist.HARTREE2EV
E_OBMP_REF_EV = -1.09480796286051 * nist.HARTREE2EV

# Các giá trị kích thước hộp (Å) và lưới (Grid size) muốn khảo sát
list_a = [30, 40, 50, 60, 80, 90, 100]  # Thay đổi kích thước ô mạng liên tục ở đây
list_gs = [100, 120, 140, 160]  # Thay đổi mật độ lưới ở đây
list_nk = [[1, 1, 1]]
use_df = True  # Bạn tự chỉnh tay         # Gamma point cho phân tử cô lập

# ============================================================
# IN HEADER CỦA BẢNG
# ============================================================
print('=' * 110)
print(
    f'{"a (Å)":<8} | {"Grid (gs)":<16} | {"k-points":<10} | {"E_KRHF (eV)":<15} | {"ΔE_HF (eV)":<12} | {"E_kOBMP2 (eV)":<15} | {"Thời gian(s)"}'
)
print('-' * 110)

# ============================================================
# VÒNG LẶP BENCHMARK
# ============================================================
for a_val in list_a:
    for gs_val in list_gs:
        for nk_val in list_nk:
            start_time = time.time()

            # Khởi tạo biến lưu kết quả (đề phòng lỗi)
            e_krhf_ev = float('nan')
            delta_e = float('nan')
            e_kobmp2_ev = float('nan')

            try:
                # 1. Khởi tạo Cell
                cell = gto.Cell()
                cell.atom = """
                H 1.5 1.5 1.0
                H 1.5 1.5 2.0
                """
                cell.basis = '6-31g'
                cell.a = np.eye(3) * a_val
                cell.gs = [gs_val, gs_val, gs_val]
                cell.verbose = 0  # Tắt log quá trình chạy để bảng in gọn gàng
                cell.build()

                kpts = cell.make_kpts(nk_val)

                # 2. Chạy KRHF
                if use_df:
                    kmf = scf.KRHF(cell, kpts).density_fit().run()
                else:
                    kmf = scf.KRHF(cell, kpts).run()

                e_krhf_ev = kmf.e_tot * nist.HARTREE2EV
                delta_e = abs(e_krhf_ev - E_RHF_REF_EV)

                # 3. Chạy kOBMP2
                krobmp = kobmp2.OBMP2(kmf)
                krobmp.second_order = True
                krobmp.verbose = 0  # Tắt log của kOBMP2

                # Hàm kernel() thường trả về energy, mo_energy...
                # Tùy thuộc vào cấu trúc của kOBMP2, bạn lấy e_tot từ object
                krobmp.kernel()

                # Giả sử năng lượng tổng lưu trong krobmp.e_tot hoặc krobmp.ene_tot
                # Nếu code của bạn báo lỗi dòng này, hãy đổi thành krobmp.ene_tot
                if hasattr(krobmp, 'e_tot'):
                    e_kobmp2_ev = krobmp.e_tot * nist.HARTREE2EV
                elif hasattr(krobmp, 'ene_tot'):
                    e_kobmp2_ev = krobmp.ene_tot * nist.HARTREE2EV
                delta_e_obmp = abs(e_kobmp2_ev - E_OBMP_REF_EV)

            except Exception as e:
                # Nếu không đủ RAM hoặc ma trận lỗi, in NaN và bỏ qua để chạy tiếp
                pass

            # Tính thời gian
            run_time = time.time() - start_time

            # In kết quả của vòng lặp hiện tại
            gs_str = f'[{gs_val}, {gs_val}, {gs_val}]'
            nk_str = str(nk_val)
            print('Đang dùng Density Fitting:', 'Có' if hasattr(kmf, 'with_df') else 'Không')
            print(
                f'{a_val:<8} | {gs_str:<16} | {nk_str:<10} | {e_krhf_ev:<15.6f} | {delta_e:<12.6f} | {delta_e_obmp:<15.6f} | {run_time:.2f}'
            )

print('=' * 110)
print('Hoàn thành quá trình quét thông số!')
