import numpy as np
from pyscf.pbc import gto, scf, mp
from pyscf.pbc.mp import kobmp2_ksymm, kmp2, kobmp2, kobmp2_ksymm_4index

# 1. Định nghĩa ô cơ sở (Cell)
cell = gto.Cell()

a = 2  # Angstrom

# lattice vector (1D theo trục x)
cell.a = np.array(
    [
        [a, 0, 0],
        [0, 10.0, 0],  # vacuum theo y
        [0, 0, 10.0],  # vacuum theo z
    ]
)

# 1 nguyên tử H trong unit cell
cell.atom = """
H 0.0 0.0 0.0
H 1 0.0 0.0
"""
cell.basis = 'gth-dzvp'  # Basis set tối giản cho tính toán nhanh
cell.pseudo = 'gth-pade'  # Pseudopotential GTH
cell.verbose = 5
cell.build()

# 2. Chạy Hartree-Fock tại điểm Gamma (RHF)
k_point_scaled = np.array([0.5, 0, 0])
# Nếu muốn tính với lưới k-points (ví dụ 3x3x1):
nk = [12, 1, 1]
kpts = cell.make_kpts(nk, scaled_center=k_point_scaled)
kmf = scf.KRHF(cell, kpts, exxdiv='none')
kmf.kernel()
