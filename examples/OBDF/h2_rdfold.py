from pyscf import gto, scf, mcscf
#from mp import UOBMP2_faster, UOBMP2_downfold, OBMP2_faster, obmp2_faster, ROBMP2_downfold
from pycmf.OBMP import OBMP2
from pycmf.OBDF import OBMP2_downfold
from pyscf import fci

from pyscf.fci import direct_uhf, direct_spin1
import numpy as np


# ===== 1. Khai báo input: H2 CAS(2,2) =====
basis = 'ccpvdz'
nocc_inact = [0, 0]      # không có orbital core đóng băng
nact = [4, 4]            # 2 orbital active (σ, σ*)
num_particles = [1, 1]   # 2 electron: 1 alpha, 1 beta
caslist_a = [1, 2, 3, 4]       # active orbital index (0-based)
caslist_b = caslist_a
caslist = [caslist_a, caslist_b]

# Tạo phân tử H2
mol = gto.Mole()
mol.atom = [
    ['H', (0.0, 0.0, 0.0)],
    ['H', (0.0, 0.0, 0.74)]   # khoảng cách cân bằng ~0.74 Å
]


'''
# ===== 1. Khai báo input =====
basis = 'ccpvdz'
nocc_inact = [2, 2] #number of orb and e inactive
nact = [8, 8] #so orb alpha và beta active
num_particles = [5, 5]   # so e alpha, beta
caslist_a = [3, 4, 5, 6, 7, 8, 9, 10]  # tinh tu 1!
caslist_b = caslist_a
caslist = [caslist_a, caslist_b]

# Tạo phân tử N₂
mol = gto.Mole()
mol.atom = [
    ['N', (0.0, 0.0, 0.0)],
    ['N', (0.0, 0.0, 1.1)] #1.1
]
'''

mol.basis = basis
mol.spin = 0
mol.build()

# ===== 2. Chạy UHF =====
myrhf = scf.RHF(mol)
myrhf.kernel()

myuhf = scf.UHF(mol).run()

# ===== 3. OBMP2 full-space =====
robmp = OBMP2(myrhf)
robmp.second_order = True
robmp.kernel()

# ===== 4. Sort MO theo caslist (restricted) =====
mycas = mcscf.CASCI(myrhf, ncas=nact[0], nelecas=sum(num_particles))
mo_sorted = mcscf.sort_mo(mycas, robmp.mo_coeff, caslist_a)

# ===== 5. Tạo đối tượng downfold =====
robact = OBMP2_downfold(myrhf, nact=nact[0], nocc_act=num_particles[0])
robact.mo_coeff = mo_sorted
robact.mo_energy = robmp.mo_energy
robact.c0_tot = getattr(robmp, "c0_tot", None)
robact.ene_tot = getattr(robmp, "ene_tot", None)
robact.fock_hf = getattr(robmp, "fock_hf", None)
robact.c1 = getattr(robmp, "c1", None)
robact.second_order = True
#robact.restricted = True

#print("robact.c0_tot, robact.ene_tot:", robact.c0_tot, robact.ene_tot)
#_, robmp.tmp1_bar = obmp2_faster.make_amp(robmp)

# Sắp lại tmp1/tmp1_bar cho đúng thứ tự MO
#robact.tmp1 = robact.sort_tmp1(robmp.tmp1, caslist)
#robact.tmp1_bar = robact.sort_tmp1(robmp.tmp1_bar, caslist)

robact.tmp1 = robmp.tmp1
robact.tmp1_bar = robmp.tmp1_bar


# ===== 6. Chạy kernel downfold =====
robact.kernel()

# ===== 7. Lấy kết quả downfold =====
h1mo_act_eff = robact.h1mo_act_eff   # ma trận 1-electron active-space
h2mo_act = robact.h2mo_act   # tensor 2-electron active-space
ene_inact = robact.ene_inact # năng lượng inactive-space

'''
# ===== 8. In/Lưu kết quả =====
print("E_inact =", ene_inact)
print("h1mo_act_eff:\n", h1mo_act_eff)
print("h2mo_act:\n", h2mo_act)
'''

# ====== FCI cho Hamiltonian đã downfold ======
h1 = h1mo_act_eff
h2 = h2mo_act

# Số electron alpha và beta trong active space
nalpha, nbeta = num_particles

# Số orbital trong active space
norb = nact[0]  # hoặc nact[1], 2 giá trị này phải bằng nhau

# Khởi tạo FCI solver dạng UHF
cis = direct_spin1.FCI()

cis.nroots = 5   # Lấy 4 trạng thái đầu tiên (ground + 3 excited)

e_fci_act, wfci = cis.kernel(h1, h2, norb, (nalpha, nbeta))

print("e_fci_act:", e_fci_act)
print("e_fci_tot:", e_fci_act + [ene_inact]*cis.nroots)