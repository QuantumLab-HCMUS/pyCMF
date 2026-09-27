import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto
from pyscf.pbc import scf  # from pyscf.pbc import gto, scf
import matplotlib.pyplot as plt

from ase.build import bulk  # Xây dựng phân tử
from ase.dft.kpoints import ibz_points, get_bandpath

from pyscf.pbc import scf as pbchf
from pyscf.pbc import df as pdf

import kobmp2_ct10_no_print as kobmp2  # tương tự với from pyscf.pbc import cc
import numpy as np
import sys  # Xuất file log
import time
import datetime
import os
import pandas as pd

# from functools import reduce

'''
Phương pháp KOBMP2
'''


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

'''
kpointx = int(os.getenv('kpointx'))
kpointy = int(os.getenv('kpointy'))
kpointz = int(os.getenv('kpointz'))

star = int(os.getenv('star'))
end = int(os.getenv('end'))

npoints = int(os.getenv('npoints'))
'''

kpointx = 3
kpointy = 3
kpointz = 3

star = 0
end = 2

npoints = 26

# Generate a unique filename using the current timestamp
output_filename = f"Output_band-structure_Si_3D_{kpointx}x{kpointy}x{kpointz}_KOBMP2_nguyenban_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Initialize Tee to write to both terminal and file
sys.stdout = Tee(output_filename)

start_time = time.time()

print("Starting program...")

# Step 1: Build cell
# Tạo cấu trúc bulk

c = bulk('Si', 'diamond', a=5.431) # Tương tự
print("Thể tích của cell:", c.get_volume())
print("Cấu trúc: ", c.cell)

cell = pbcgto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(c)
cell.a = c.cell  # Tương tự
print("Cấu trúc 2: ", c.cell)


cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.verbose = 5  # Tương tự
cell.exp_to_discard = 0.1  # Loại bỏ các hàm cơ sở có exponent < 0.1, tránh tràn số (Chỗ đã chỉnh sửa)
cell.a = np.array(cell.a).tolist()  # Chuyển numpy array thành list (Chỗ đã chỉnh sửa)
cell.build()  # Hơi khác 1 chút

# Số điểm trên đường K-path

points = ibz_points['fcc']  # Chỗ đã chỉnh sửa
G = points['Gamma']  # Chỗ đã chỉnh sửa
X = points['X']
W = points['W']
K = points['K']
L = points['L']

# band_kpts, kpath, sp_points = get_bandpath([L, G, X, W, K, G], c.cell, npoints=npoints1) # Y chang

path = get_bandpath([L, G, X, W, K, G], cell.a , npoints=npoints)
band_kpts = path.kpts
x_axis, sp_points, labels = path.get_linear_kpoint_axis()  # Use x_axis for plotting

print("band_kpts: ")
print(band_kpts)
#band_kpts = cell.get_abs_kpts(band_kpts)
print("x_axis: ")
print(x_axis)
print("sp_points: ")
print(sp_points)

#band_v1 = []
#band_c1 = []

HOMO = np.zeros(npoints)
LUMO = np.zeros(npoints)

HOMO_1 = np.zeros(npoints)
LUMO_1 = np.zeros(npoints)

HOMO_2 = np.zeros(npoints)
LUMO_2 = np.zeros(npoints)

IP = np.zeros(npoints)
EA = np.zeros(npoints)

for i in range(star, end):

    
    print("center k", band_kpts[i])
    kpts = cell.make_kpts([kpointx, kpointy, kpointz], scaled_center=band_kpts[i])
    kmf = scf.KRHF(cell, kpts, exxdiv='none')
    kmf.kernel()
    print(f"KRHF ({kpointx}x{kpointy}x{kpointz}) completed. Elapsed time: {time.time() - start_time:.2f} seconds")
    
    '''
    print("center k", band_kpts[i])
    kmf = pbchf.KRHF(cell)
    kmf.with_df = pdf.AFTDF(cell)
    kmf.kpts = cell.make_kpts([5, 5, 1], scaled_center=band_kpts[i])
    kmf.kernel()
    '''

    mypt = kobmp2.OBMP2(kmf)
    mypt.kernel()
    #print("KOBMP2 energy (per unit cell) =", mypt.e_tot)
    print(f"MP2 ({kpointx}x{kpointy}x{kpointz}) completed. Elapsed time: {time.time() - start_time:.2f} seconds")

    HOMO[i] = mypt.mo_energy[0][mypt.nocc - 1].item()
    LUMO[i] = mypt.mo_energy[0][mypt.nocc].item()

    HOMO_1[i] = mypt.mo_energy[0][mypt.nocc - 2].item()
    LUMO_1[i] = mypt.mo_energy[0][mypt.nocc + 1].item()

    HOMO_2[i] = mypt.mo_energy[0][mypt.nocc - 3].item()
    LUMO_2[i] = mypt.mo_energy[0][mypt.nocc + 2].item()

    IP[i] = mypt.IP
    EA[i] = mypt.EA

print(f"Band_v1: {HOMO}")
print(f"Band_c1: {LUMO}")
print(f"Band_v2: {HOMO_1}")
print(f"Band_c2: {LUMO_1}")
print(f"Band_v2: {HOMO_2}")
print(f"Band_c2: {LUMO_2}")
print(f"IP: {IP}")
print(f"EA: {EA}")

au2ev = 27.21139

# Create filename with timestamp
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
filename = os.path.join(os.getcwd(), f"Band-structure_Si_{kpointx}x{kpointy}x{kpointz}_KOBMP2_nguyenban_{timestamp}.xlsx")

# Create DataFrames for each section
df_kobmp2 = pd.DataFrame({
    'x_axis': x_axis,
    'hf_HOMO': np.array(HOMO) * au2ev,
    'hf_v2': np.array(HOMO_1) * au2ev,
    'hf_v3': np.array(HOMO_2) * au2ev,
    'hf_LUMO': np.array(LUMO) * au2ev,
    'hf_c2': np.array(LUMO_1) * au2ev,
    'hf_c3': np.array(LUMO_2) * au2ev,
    'hf_IP': np.array(IP) * au2ev,
    'hf_EA': np.array(EA) * au2ev
})



# Special points
# Sửa đoạn code tạo df_sp thành:
sp_data = list(zip(sp_points, labels))
df_sp = pd.DataFrame(sp_data, columns=['x_coordinate', 'label'])  # Chỗ đã chỉnh sửa

# Conversion factor
df_au = pd.DataFrame({'au2ev': [au2ev]})

# Write all data to Excel file with multiple sheets
with pd.ExcelWriter(filename) as writer:
    df_kobmp2.to_excel(writer, sheet_name='KOBMP2_NB Bands', index=False)
    df_sp.to_excel(writer, sheet_name='Special Points', index=False)
    df_au.to_excel(writer, sheet_name='Conversion Factor', index=False)