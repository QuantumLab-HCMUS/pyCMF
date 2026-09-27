import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto

# import pyscf.pbc.dft as pbcdft
from pyscf.pbc import scf, cc  # from pyscf.pbc import gto, scf
# import matplotlib.pyplot as plt

from ase.build import bulk  # Xây dựng phân tử
from ase.dft.kpoints import ibz_points, get_bandpath

import numpy as np
import sys  # Xuất file log
import time
import datetime
import os
import pandas as pd


from pyscf import lib

lib.num_threads(8)

# from functools import reduce

"""
Phương pháp KOBMP2
"""


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
output_filename = (
    f"output_Si_2X2X2_HF_CCSD_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

# Initialize Tee to write to both terminal and file
sys.stdout = Tee(output_filename)

start_time = time.time()

print("Starting program...")

# Step 1: Build cell
# Tạo cấu trúc bulk

"""
kpointx = int(os.getenv('kpointx'))
kpointy = int(os.getenv('kpointy'))
kpointz = int(os.getenv('kpointz'))

star = int(os.getenv('star'))
end = int(os.getenv('end'))

npoints1 = int(os.getenv('npoints')) # Tăng số điểm để đường cong mượt hơn
"""

kpointx = 2
kpointy = 2
kpointz = 2

star = 0
end = 2

npoints1 = 26

c = bulk("Si", "diamond", a=5.431)  # Tương tự
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

# band_kpts, kpath, sp_points = get_bandpath([L, G, X, W, K, G], c.cell, npoints=npoints1) # Y chang

path = get_bandpath([L, G, X, W, K, G], cell.a, npoints=npoints1)
band_kpts = path.kpts
x_axis, sp_points, labels = path.get_linear_kpoint_axis()  # Use x_axis for plotting

print("band_kpts: ")
print(band_kpts)
# band_kpts = cell.get_abs_kpts(band_kpts)
print("x_axis: ")
print(x_axis)
print("sp_points: ")
print(sp_points)

list_ip = np.zeros(npoints1)
list_ea = np.zeros(npoints1)

for i in range(star, end):
    "KRHF"
    print("center k", band_kpts[i])
    kpts = cell.make_kpts([kpointx, kpointy, kpointz], scaled_center=band_kpts[i])
    kmf = scf.KRHF(cell, kpts, exxdiv="none")
    kmf.kernel()
    print(
        f"KRHF ({kpointx}x{kpointy}x{kpointz}) completed. Elapsed time: {time.time() - start_time:.2f} seconds"
    )

    print("kmf.mo_energy = ", kmf.mo_energy)

    "KRCCSD"
    mycc = cc.KRCCSD(kmf)
    mycc.kernel()
    ea = mycc.eaccsd(nroots=1, kptlist=[0])
    ip = mycc.ipccsd(nroots=1, kptlist=[0])
    print(
        f"KRCCSD ({kpointx}x{kpointy}x{kpointz}) completed. Elapsed time: {time.time() - start_time:.2f} seconds"
    )

    list_ip[i] = -ip[0][0].item()  # Extract scalar value
    list_ea[i] = ea[0][0].item()  # Extract scalar value


"""
'KRHF'
max_hf = max(band_v1)
band_v1 = [x - max_hf for x in band_v1]
#band_v2 = [x - max_hf for x in band_v2]
band_c1 = [x - max_hf for x in band_c1]
#band_c2 = [x - max_hf for x in band_c2]

'KRCCSD'
band_v1_cc = [arr for arr in list_ip]
band_c1_cc = [arr for arr in list_ea]

max_cc = max(band_v1_cc)
band_v1_cc = [x - max_cc for x in band_v1_cc]
band_c1_cc = [x - max_cc for x in band_c1_cc]
"""

au2ev = 27.21139


# Create filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(
    os.getcwd(),
    f"Band-structure_Si_{kpointx}x{kpointy}x{kpointz}_HF_CCSD_{timestamp}.xlsx",
)

# Create DataFrames for each section
"""
df_hf = pd.DataFrame({
    'x_axis': x_axis,
    'hf_v1': np.array(band_v1) * au2ev,
    #'hf_v2': np.array(band_v2) * au2ev,
    'hf_c1': np.array(band_c1) * au2ev
    #'hf_c2': np.array(band_c2) * au2ev
})
"""

df_ccsd = pd.DataFrame(
    {
        "x_axis": x_axis,
        "ccsd_v1": np.array(list_ip) * au2ev,
        "ccsd_c1": np.array(list_ea) * au2ev,
    }
)

# Special points
# Sửa đoạn code tạo df_sp thành:
sp_data = list(zip(sp_points, labels))
df_sp = pd.DataFrame(sp_data, columns=["x_coordinate", "label"])  # Chỗ đã chỉnh sửa

# Conversion factor
df_au = pd.DataFrame({"au2ev": [au2ev]})

# Write all data to Excel file with multiple sheets
with pd.ExcelWriter(filename) as writer:
    # df_hf.to_excel(writer, sheet_name='HF Bands', index=False)
    df_ccsd.to_excel(writer, sheet_name="CCSD Bands", index=False)
    df_sp.to_excel(writer, sheet_name="Special Points", index=False)
    df_au.to_excel(writer, sheet_name="Conversion Factor", index=False)
