from pyscf import gto, scf, mcscf, tools
from pycmf.OBMP import UOBMP2
from pycmf.OBDF import UOBMP2_downfold

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

def print_ref(ci_vecs, norbcas, neleccas, nocc_inact, ene, s2_casci, m_casci, ene_casci):
    print("Ref:")    
    for i, ci in enumerate(ci_vecs):
        s2, mult = spin_op.spin_square(ci, norb, sum(neleccas))
        print("state", i, "E:", e_df_fci[i])
        print(f"<S^2>_dfold = {s2:.0f}, multiplicity_dfold = {mult:.0f}", "E_excited_dfold:", ene[i] - ene[0])
        print(f"<S^2>_casci = {s2_casci[i]:.0f}, "
              f"multiplicity_casci = {m_casci[i]:.0f}",
              "E_excited_casci:", ene_casci[i] - ene_casci[0])        
        print_ci(ci, norbcas, neleccas, nocc_inact)

np.set_printoptions(precision=8, suppress=True)  # suppress=True bỏ 'e-xx'

# ===== 1. Khai báo input: C3H4 =====
#22 e 11 11 3 3
nocc_inact = [3, 3]     
nact = [11, 11]            
num_particles = [8, 8]
nelec = [11,11]   
caslist_a = list(range(nocc_inact[0]+1,nact[0]+nocc_inact[0]+1))     
caslist_b = caslist_a
caslist = [caslist_a, caslist_b]

n=15
# Tạo phân tử C3H4
mol = gto.Mole()

mol.atom = [

    ['C',( 0.00000000, 0.00000000,-0.88277808)],
    ['C',( 0.00000000, 0.64836859, 0.47986540)],
    ['C',( 0.00000000,-0.64836859, 0.47986540)],
    ['H',( 0.91153656, 0.00000000,-1.47048371)],
    ['H',(-0.91153656, 0.00000000,-1.47048371)],
    ['H',( 0.00000000, 1.57612532, 1.01642650)],
    ['H',( 0.00000000,-1.57612532, 1.01642650)]
]

'''
7
Cyclopropene,^1A_1,CC3,aug-cc-pVTZ
C    0.00000000            0.00000000           -0.88277808
C    0.00000000            0.64836859            0.47986540
C    0.00000000           -0.64836859            0.47986540
H    0.91153656            0.00000000           -1.47048371
H   -0.91153656            0.00000000           -1.47048371
H    0.00000000            1.57612532            1.01642650
H    0.00000000           -1.57612532            1.01642650
'''

mol.basis = "cc-pVDZ"
#mol.basis = "aug-cc-pVTZ"
mol.spin = 0
mol.build()

# ===== 2. Chạy UHF =====
myuhf = scf.UHF(mol)
myuhf.kernel()

#for i in range(15):
#    tools.cubegen.orbital(mol, f"mo_{i}.cube", myuhf.mo_coeff[0][:, i])

'''
# Xuất file molden
# Alpha spin orbitals
tools.molden.from_mo(
    mol,
    'water_alpha.molden',
    myuhf.mo_coeff[0],
    ene=myuhf.mo_energy[0],
    occ=myuhf.mo_occ[0],
    spin='Alpha'
)

# Beta spin orbitals
tools.molden.from_mo(
    mol,
    'water_beta.molden',
    myuhf.mo_coeff[1],
    ene=myuhf.mo_energy[1],
    occ=myuhf.mo_occ[1],
    spin='Beta'
)
'''

# ===== 2.1 Casci
mycasci = mcscf.CASCI(myuhf, ncas=nact[0]+nocc_inact[0], nelecas=nelec)
mycasci.fcisolver.nroots = n
e_casci, e_casci_act, ci_vecs_casci, _, _ = mycasci.kernel()

from pyscf.fci.spin_op import spin_square

ncas = mycasci.ncas
nelecas = mycasci.nelecas

ss_casci = []
m_casci = []

for i, ci in enumerate(mycasci.ci):
    ss, mult = mycasci.fcisolver.spin_square(ci, ncas, nelecas)
    print(f"State {i:2d}:  <S^2> = {ss:.6f}   Multiplicity = {mult:.3f}")
    
    ss_casci.append(ss)
    m_casci.append(mult)

# ===== 3. OBMP2 full-space =====
uobmp = UOBMP2(myuhf)
uobmp.second_order = True
uobmp.kernel()

# ===== 4. Sort MO theo caslist =====
mycas = mcscf.UCASCI(myuhf, ncas=nact[0], nelecas=sum(num_particles))
mo_sorted = mcscf.sort_mo(mycas, uobmp.mo_coeff, caslist)

'''
# Xuất file molden
# Alpha spin orbitals
tools.molden.from_mo(
    mol,
    'water_alpha_ob.molden',
    mo_sorted[0],
    ene=uobmp.mo_energy[0],
    occ=uobmp.mo_occ[0],
    spin='Alpha'
)

# Beta spin orbitals
tools.molden.from_mo(
    mol,
    'water_beta_ob.molden',
    mo_sorted[1],
    ene=uobmp.mo_energy[1],
    occ=uobmp.mo_occ[1],
    spin='Beta'
)
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

# Sắp lại tmp1/tmp1_bar cho đúng thứ tự MO
uobact.tmp1 = uobact.sort_tmp1(uobmp.tmp1, caslist)
uobact.tmp1_bar = uobact.sort_tmp1(uobmp.tmp1_bar, caslist)

# ===== 6. Chạy kernel downfold =====
uobact.kernel()

# ===== 7. Lấy kết quả downfold =====
h1mo_act_eff = uobact.h1mo_act_eff   # ma trận 1-electron active-space
h2mo_act = uobact.h2mo_act   # tensor 2-electron active-space
ene_inact = uobact.ene_inact # năng lượng inactive-space

'''
print("h1mo_act_eff:\n", h1mo_act_eff)
print("h2mo_act:\n", h2mo_act)
print("E_inact =", ene_inact)
'''

# ====== 9. FCI cho Hamiltonian đã downfold ======
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

cis.nroots = n  

# Chạy FCI với Hamiltonian downfold
# Hàm kernel trả về (năng lượng, vector sóng)
# Năng lượng này chỉ là phần active-space (đã bao gồm hiệu ứng outer qua downfold)
e_df_fci_act, ci_vecs = cis.kernel((h1a, h1b), (h2aa, h2ab, h2bb), norb, (nalpha, nbeta))
e_df_fci = e_df_fci_act + [ene_inact]*cis.nroots

print("e_df_fci_act:\n", e_df_fci_act)

e_df_fci_act *= 27.21138629
e_df_fci *= 27.21138629
e_casci *= 27.21138629



print("\nDownfold + FCI")
print_ref(ci_vecs, norb, num_particles, nocc_inact, e_df_fci, ss_casci, m_casci, e_casci)
