import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Bổ sung thư viện pandas
from pyscf.pbc import df
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf

# =====================================================================
# 1. THIẾT LẬP THÔNG SỐ QUÉT
# =====================================================================
# Danh sách các khoảng cách R (đơn vị a.u.) theo đồ thị Hình 2 bài SEET
R_values = [1.4, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6]
energies_khf = []

print('BẮT ĐẦU QUÉT NĂNG LƯỢNG KHF THEO KHOẢNG CÁCH R...')
print('-' * 60)

# =====================================================================
# 2. VÒNG LẶP CHÍNH
# =====================================================================
for R in R_values:
    # Thiết lập Unit Cell
    cell = pbc_gto.Cell()
    cell.a = [[2 * R, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]
    cell.atom = f'H 0 0 0; H {R} 0 0'
    cell.basis = 'sto-6g'
    cell.unit = 'Bohr'
    cell.dimension = 1
    cell.low_dim_ft_type = 'inf_vacuum'
    cell.max_memory = 65536  # 64 GB
    # cell.verbose = 0  # Tắt log để màn hình sạch sẽ
    cell.build()

    # Xây dựng lưới k-points.
    kpts = cell.make_kpts([60, 1, 1])

    # Thiết lập và chạy KHF
    mf_hf = pbc_scf.KRHF(cell, kpts)
    mf_hf.with_df = df.AFTDF(cell, kpts)

    energy_hf_total_cell = mf_hf.kernel()

    energies_khf.append(energy_hf_total_cell)

    print(f'R = {R:.1f} a.u. | Năng lượng KHF (Unit Cell): {energy_hf_total_cell:.6f} Hartree')

print('-' * 60)

# =====================================================================
# 3. VẼ ĐỒ THỊ (TÁI LẬP HÌNH 2 - BÀI BÁO SEET)
# =====================================================================
plt.figure(figsize=(8, 6))

# Vẽ đường KHF (màu tím giống trong bài SEET)
plt.plot(
    R_values,
    energies_khf,
    marker='x',
    linestyle='-',
    color='purple',
    linewidth=2,
    markersize=8,
    label='HF (Mô phỏng lại)',
)

# Tinh chỉnh giao diện đồ thị bám sát phong cách bài báo
plt.xlabel('R (H-H) [a.u.]', fontsize=14)
plt.ylabel('Energy per -(H-H)- unit [a.u.]', fontsize=14)
plt.title('Phương trình trạng thái chuỗi 1D Hydrogen (Mạng tuần hoàn)', fontsize=14)

# Đặt giới hạn trục Y cho giống Hình 2 (từ -1.10 đến -0.80)
plt.ylim(-1.10, -0.80)
plt.xlim(1.3, 3.7)

# Cấu hình vạch chia (ticks)
plt.tick_params(direction='in', length=6, width=1, colors='black', labelsize=12, top=True, right=True)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=False, fontsize=12, loc='upper left')

plt.tight_layout()
plt.savefig('SEET_Fig2_KHF_Replication_60x1x1.png', dpi=300)
print('Đã xuất đồ thị thành công ra file: SEET_Fig2_KHF_Replication_60x1x1.png')

# Hiển thị đồ thị
# plt.show()
plt.close()

# =====================================================================
# 4. XUẤT DỮ LIỆU RA FILE EXCEL
# =====================================================================
# Tạo DataFrame từ 2 list R_values và energies_khf
df_results = pd.DataFrame({'Khoảng cách R (Bohr)': R_values, 'Năng lượng KHF (Hartree)': energies_khf})

# Tên file Excel
excel_filename = 'Nang_luong_H_chain_KHF_60x1x1.xlsx'

# Xuất ra Excel (bỏ cột index)
df_results.to_excel(excel_filename, index=False)
print(f'Đã xuất dữ liệu thành công ra file Excel: {excel_filename}')
