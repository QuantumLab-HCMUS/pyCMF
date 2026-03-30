from pyscf import gto, scf, mcscf
from pyscf.mp import UOBMP2_faster, UOBMP2_downfold, OBMP2_faster, obmp2_faster, ROBMP2_downfold

from pyscf import fci

from pyscf.fci import direct_uhf, direct_spin1, spin_op
import numpy as np

# ===== 0. Khai báo các hàm =====
def occstr(bin_str, norb): #Chuyen binary string thành occupation string chuan
    return format(int(bin_str, 2), f"0{norb}b")

def print_ci(ci, norbcas, neleccas, nocc_inact, tol=0.05):
    from pyscf.fci import addons
    
    norb = norbcas + nocc_inact[0] 
    nelec = np.array(neleccas) + nocc_inact
    
    terms = addons.large_ci(ci, norbcas, sum(neleccas), tol=tol)
    a_0 = [1]*nelec[0] + [0]*(norb-nelec[0])
    b_0 = [1]*nelec[1] + [0]*(norb-nelec[1])

    for coeff, a, b in terms:
    
        a_str = occstr(a, norbcas) + '1'*nocc_inact[1]
        b_str = occstr(b, norbcas) + '1'*nocc_inact[1]
      
        a_occ = np.array(list(map(int, str(a_str))))
        b_occ = np.array(list(map(int, str(b_str))))
        a_occ = a_occ[::-1]
        b_occ = b_occ[::-1]
        
        a_tran = a_occ - a_0
        b_tran = b_occ - b_0
        
        a_add  = np.where(a_tran ==  1)[0] + 1
        a_lost = np.where(a_tran == -1)[0] + 1

        b_add  = np.where(b_tran ==  1)[0] + 1
        b_lost = np.where(b_tran == -1)[0] + 1

        print(f"{coeff:+.6f}  α:{a_str}  β:{b_str}", "α:", a_lost, "->", a_add, "β:", b_lost, "->", b_add)

def print_ref(ci_vecs, norbcas, neleccas, nocc_inact, ene):
    print("Ref:")    
    for i, ci in enumerate(ci_vecs):
        print("ci_vecs:\n", ci.shape)
        s2, mult = spin_op.spin_square(ci, norb, sum(neleccas))
        print("state", i, "E:", e_df_fci[i], "E_exci:", ene[i] - ene[0])
        print(f"<S^2> = {s2:.0f}, multiplicity = {mult:.0f}")
        print_ci(ci, norbcas, neleccas, nocc_inact)

np.set_printoptions(precision=8, suppress=True)  # suppress=True bỏ 'e-xx'

# ===== 1. Khai báo input: =====

basis = 'ccpvdz'
nocc_inact = [2, 2]     
nact = [6, 6]            
num_particles = [3, 3]
nelec = [5,5]   
caslist_a = list(range(nocc_inact[0]+1,nact[0]+nocc_inact[0]+1))       # active orbital index (0-based)
caslist_b = caslist_a
caslist = [caslist_a, caslist_b]

n=15
# Tạo phân tử H2O
mol = gto.Mole()
mol.atom = [
    ['O', (0.0, 0.0,       -0.06990256)],
    ['H', (0.0, 0.75753241, 0.51843495)],   
    ['H', (0.0,-0.75753241, 0.51843495)]
]

'''
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

'''
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

# ===== 2.1 Casci
mycasci = mcscf.CASCI(myrhf, ncas=nact[0]+nocc_inact[0], nelecas=nelec)
mycasci.fcisolver.nroots = n
e_casci, e_casci_act, ci_vecs_casci, _, _ = mycasci.kernel()

# ===== 3. OBMP2 full-space =====
robmp = OBMP2_faster(myrhf)
robmp.second_order = True
robmp.kernel()

# ===== 4. Sort MO theo caslist (restricted) =====
mycas = mcscf.CASCI(myrhf, ncas=nact[0], nelecas=sum(num_particles))
mo_sorted = mcscf.sort_mo(mycas, robmp.mo_coeff, caslist_a)

# ===== 5. Tạo đối tượng downfold =====
robact = ROBMP2_downfold(myrhf, nact=nact[0], nocc_act=num_particles[0])
robact.mo_coeff = mo_sorted
robact.mo_energy = robmp.mo_energy
robact.c0_tot = getattr(robmp, "c0_tot", None)
robact.ene_tot = getattr(robmp, "ene_tot", None)
robact.fock_hf = getattr(robmp, "fock_hf", None)
robact.c1 = getattr(robmp, "c1", None)
robact.second_order = True

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

cis.nroots = n   

e_df_fci_act, ci_vecs = cis.kernel(h1, h2, norb, (nalpha, nbeta))
e_df_fci = e_df_fci_act + [ene_inact]*cis.nroots

e_df_fci_act *= 27.21138629
e_df_fci *= 27.21138629
e_casci *= 27.21138629

print("\nDownfold + FCI")
print_ref(ci_vecs, norb, num_particles, nocc_inact, e_df_fci)

#print("\nCASCI")
#print_ref(ci_vecs_casci, nact[0]+nocc_inact[0], nelec, [0,0], e_casci)
