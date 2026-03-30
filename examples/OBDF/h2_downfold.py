from pyscf import gto, scf, mcscf, tools
from pycmf.OBMP import UOBMP2
from pycmf.OBDF import UOBMP2_downfold

from pyscf import fci

from pyscf.fci import direct_uhf, direct_spin1
import numpy as np


# ===== 0. Khai báo các hàm =====



# ===== 1. Khai báo input: =====
basis = 'ccpvdz'


#==========H6(active space)==========
nocc_inact = [0, 0]      
nact = [12, 12]            
num_particles = [3, 3]   
caslist_a = list(range(nocc_inact[0]+1,nact[0]+nocc_inact[0]+1))        # active orbital index (0-based)
caslist_b =  caslist_a
caslist   = [caslist_a, caslist_b]
#==========H6(active space)==========
mol = gto.Mole()
mol.atom = [
    ['H', (0.0, 0.0, 0.0)],
    ['H', (0.0, 0.0, 0.74)],
    ['H', (0.0, 0.0, 1.48)],
    ['H', (0.0, 0.0, 2.22)],
    ['H', (0.0, 0.0, 2.96)],
    ['H', (0.0, 0.0, 3.70)],
    ]


'''
nocc_inact = [0, 0]      
nact = [4, 4]            
num_particles = [1, 1]   
caslist_a = [1, 2, 3, 4]       
caslist_b = caslist_a
caslist = [caslist_a, caslist_b]

# Tạo phân tử H2
mol = gto.Mole()
mol.atom = [
    ['H', (0.0, 0.0, 0.0)],
    ['H', (0.0, 0.0, 0.74)]   # khoảng cách cân bằng ~0.74 Å
]
'''
mol.basis = basis
mol.spin = 0
mol.build()


# ===== 2. Chạy UHF =====
myuhf = scf.UHF(mol)
myuhf.kernel()

'''
#=====2.1 chạy CCSD đê so sánh======================
from pyscf import cc
from copy import copy

mf_ref = copy(myuhf)
ucc = cc.UCCSD(mf_ref)
ucc.conv_tol = 1e-8
e_ccsd, t1, t2 = ucc.kernel()
E_tot_UCCSD = myuhf.e_tot + e_ccsd
print("E[UCCSD] total =", E_tot_UCCSD)

e_t = ucc.ccsd_t()   # (T) cho UCCSD
E_tot_UCCSDT = E_tot_UCCSD + e_t
print("E[UCCSD(T)] total =", E_tot_UCCSDT)
#===================================================

# ===== CASCI trực tiếp từ UHF =====
# tạo đối tượng CASCI: cần số orbital active và số electron trong active
mycasci = mcscf.CASCI(myuhf, ncas=nact[0], nelecas=num_particles)

# sắp xếp MO theo caslist
#mo_sorted_uhf = mcscf.sort_mo(mycasci, myuhf.mo_coeff, caslist)

# chạy CASCI
casci = mycasci.kernel() #mo_sorted_uhf

'''

# ===== 3. OBMP2 full-space =====
uobmp = UOBMP2(myuhf)
uobmp.second_order = True
uobmp.kernel()

# ===== 4. Sort MO theo caslist =====
mycas = mcscf.UCASCI(myuhf, ncas=nact[0], nelecas=sum(num_particles))
mo_sorted = mcscf.sort_mo(mycas, uobmp.mo_coeff, caslist)

'''
# ===== 4b. FCI full-space =====

cis_full = fci.FCI(myuhf, mo_sorted)   # dùng MO từ UOB-MP2 cũng ok
cis_full.nroots = 3
e_fci_full, wfci_full = cis_full.kernel()

print("\n=== FCI Full-space ===")
for i, e in enumerate(e_fci_full):
    if i == 0:
        label = "Ground state"
    else:
        label = f"Excited state {i}"
    print(f"{label}: {e:.8f} Ha")
'''

# ===== 5. Tạo đối tượng downfold =====
uobact = UOBMP2_downfold(myuhf, nact=nact, nocc_act=num_particles)
uobact.mo_coeff = mo_sorted
uobact.mo_energy = uobmp.mo_energy
uobact.c0_tot = getattr(uobmp, "c0_tot", None)
uobact.ene_tot = getattr(uobmp, "ene_tot", None)
uobact.fock_hf = getattr(uobmp, "fock_hf", None)
uobact.c1 = getattr(uobmp, "c1", None)
uobact.second_order = True

#print("robact.c0_tot, robact.ene_tot, robact.c1:", uobact.c0_tot, uobact.ene_tot, uobact.c1)


# Sắp lại tmp1/tmp1_bar cho đúng thứ tự MO
uobact.tmp1 = uobact.sort_tmp1(uobmp.tmp1, caslist)
uobact.tmp1_bar = uobact.sort_tmp1(uobmp.tmp1_bar, caslist)


# ===== 6. Chạy kernel downfold =====
uobact.kernel()


# ===== 7. Lấy kết quả downfold =====
h1mo_act_eff = uobact.h1mo_act_eff   # ma trận 1-electron active-space
h2mo_act = uobact.h2mo_act   # tensor 2-electron active-space
ene_inact = uobact.ene_inact # năng lượng inactive-space

# ===== 8. In/Lưu kết quả =====
'''
print("E_inact =", ene_inact)
print("h1mo_act_eff:\n", h1mo_act_eff)
print("h2mo_act:\n", h2mo_act)
'''

# ====== FCI cho Hamiltonian đã downfold ======
# h1mo_act_eff là tuple (h1a, h1b)
h1a, h1b = h1mo_act_eff

# h2mo_act là tuple chứa tích phân 2-e, theo UHF
# Thông thường: (h2aa, h2ab, h2bb, h2ba)
# Nhưng với pyscf.direct_uhf.FCI, chỉ cần (h2aa, h2ab, h2bb)
h2aa = h2mo_act[0]
h2ab = h2mo_act[1]
h2bb = h2mo_act[3]

# Số electron alpha và beta trong active space
nalpha, nbeta = num_particles

# Số orbital trong active space
norb = nact[0]  # hoặc nact[1], 2 giá trị này phải bằng nhau

# Khởi tạo FCI solver dạng UHF
cis = direct_uhf.FCI()

cis.nroots = 5  # Lấy 4 trạng thái đầu tiên (ground + 3 excited)

# Chạy FCI với Hamiltonian downfold
# Hàm kernel trả về (năng lượng, vector sóng)
# Năng lượng này chỉ là phần active-space (đã bao gồm hiệu ứng outer qua downfold)
e_fci_act, wfci = cis.kernel((h1a, h1b), (h2aa, h2ab, h2bb), norb, (nalpha, nbeta))

#print("Năng lượng FCI cho active space:", e_fci_act)

print("e_fci_act:", e_fci_act)
print("e_fci_tot:", e_fci_act + [ene_inact]*cis.nroots)

#print("h2aa:\n", h2aa)

'''
# ====== FCI cho Hamiltonian đã downfold ======
# h1mo_act_eff là tuple (h1a, h1b)
h1a, h1b = h1mo_act_eff

# h2mo_act là tuple chứa tích phân 2-e, theo UHF
# Thông thường: (h2aa, h2ab, h2bb, h2ba)
# Nhưng với pyscf.direct_uhf.FCI, chỉ cần (h2aa, h2ab, h2bb)
h2aa = h2mo_act[0]
h2ab = h2mo_act[1]
h2bb = h2mo_act[3]

# Số electron alpha và beta trong active space
nalpha, nbeta = num_particles

# Số orbital trong active space
norb = nact[0]  # hoặc nact[1], 2 giá trị này phải bằng nhau

# Khởi tạo FCI solver dạng UHF
cis = direct_spin1.FCI()

cis.nroots = 5   # Lấy 4 trạng thái đầu tiên (ground + 3 excited)

e_fci_act, wfci = cis.kernel(h1a, h2bb, norb, (nalpha, nbeta))

print("e_fci_act:", e_fci_act)
print("e_fci_tot:", e_fci_act + [ene_inact]*cis.nroots)
'''
