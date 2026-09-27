import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft

import matplotlib.pyplot as plt

from ase.build import bulk
from ase.dft.kpoints import ibz_points, get_bandpath

import sys  # Xuất file log
import time
import datetime
import os
import pandas as pd
import numpy as np


class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# Generate a unique filename using the current timestamp
output_filename = f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Initialize Tee to write to both terminal and file
sys.stdout = Tee(output_filename)

start_time = time.time()

c = bulk("Ge", "diamond", a=5.658)  # Tương tự
print(c.get_volume())

cell = pbcgto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(c)
cell.a = c.cell  # Tương tự

cell.basis = "gth-szv"
cell.pseudo = "gth-pade"
cell.verbose = 5  # Tương tự
cell.exp_to_discard = (
    0.1  # Loại bỏ các hàm cơ sở có exponent < 0.1, tránh tràn số (Chỗ đã chỉnh sửa)
)
cell.a = np.array(cell.a).tolist()  # Chuyển numpy array thành list (Chỗ đã chỉnh sửa)
cell.build()  # Hơi khác 1 chút

# Số điểm trên đường K-path

points = ibz_points["fcc"]  # Chỗ đã chỉnh sửa
G = points["Gamma"]  # Chỗ đã chỉnh sửa
X = points["X"]
W = points["W"]
K = points["K"]
L = points["L"]

# Số điểm trên đường K-path
npoints1 = 110

path = get_bandpath([L, G, X, W, K, G], c.cell, npoints=npoints1)
band_kpts = path.kpts
x_axis, sp_points, labels = path.get_linear_kpoint_axis()  # Use x_axis for plotting

mf = pbcdft.RKS(cell)
print("Energy from Gamma point sampling:", mf.kernel())

# Cấu hình KRKS với 222 k-point sampling và tính toán năng lượng
kmf = pbcdft.KRKS(cell, cell.make_kpts([2, 2, 2]))
print("Energy from 222 k-point sampling:", kmf.kernel())

e_kn_2 = kmf.get_bands(band_kpts)[0]
vbmax = -99
for en in e_kn_2:
    vb_k = en[cell.nelectron // 2 - 1]
    if vb_k > vbmax:
        vbmax = vb_k
e_kn_2 = [en - vbmax for en in e_kn_2]

# Trích xuất năng lượng tại các điểm k
vbm_index = cell.nelectron // 2 - 1  # Chỉ số VBM
cbm_index = cell.nelectron // 2  # Chỉ số CBM
print(f"DFT (2x2x2) completed. Elapsed time: {time.time() - start_time:.2f} seconds")

au2ev = 27.21139

emin = -1 * au2ev
emax = 1 * au2ev


# Create filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(os.getcwd(), f"Band-structure_{timestamp}_Ge_old_110.xlsx")


# DFT data: assuming e_kn_2 is a 2D array
dft_data = np.array(e_kn_2) * au2ev
n_bands = dft_data.shape[1]
dft_columns = {"x_axis": x_axis}
for i in range(n_bands):
    dft_columns[f"DFT_band_{i + 1}"] = dft_data[:, i]
df_dft = pd.DataFrame(dft_columns)

# Special points
# Sửa đoạn code tạo df_sp thành:
sp_data = list(zip(sp_points, labels))
df_sp = pd.DataFrame(sp_data, columns=["x_coordinate", "label"])

# Conversion factor
df_au = pd.DataFrame({"au2ev": [au2ev]})

# Write all data to Excel file with multiple sheets
with pd.ExcelWriter(filename) as writer:
    df_dft.to_excel(writer, sheet_name="DFT Bands", index=False)
    df_sp.to_excel(writer, sheet_name="Special Points", index=False)
    df_au.to_excel(writer, sheet_name="Conversion Factor", index=False)
