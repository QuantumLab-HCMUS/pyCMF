from pyscf import gto, scf, mcscf, tools
from pycmf.OBMP import UOBMP2
from pycmf.OBDF import UOBMP2_downfold

from pyscf import fci
from pyscf.fci import direct_uhf, direct_spin1, spin_op
import numpy as np


# ===== 0. Khai báo các hàm =====
def occstr(bin_str, norb):  # Chuyen binary string thành occupation string chuan
    return format(int(bin_str, 2), f'0{norb}b')


def print_ci(ci, norbcas, neleccas, nocc_inact, tol=0.05):
    from pyscf.fci import addons

    norb = norbcas + nocc_inact[0]
    nelec = np.array(neleccas) + nocc_inact[1]

    terms = addons.large_ci(ci, norbcas, sum(neleccas), tol=tol)
    a_0 = [1] * nelec[0] + [0] * (norb - nelec[0])
    b_0 = [1] * nelec[1] + [0] * (norb - nelec[1])

    for coeff, a, b in terms:
        a_str = occstr(a, norbcas) + '1' * nocc_inact[1]
        b_str = occstr(b, norbcas) + '1' * nocc_inact[1]

        a_occ = np.array(list(map(int, str(a_str))))
        b_occ = np.array(list(map(int, str(b_str))))
        a_occ = a_occ[::-1]
        b_occ = b_occ[::-1]

        a_tran = a_occ - a_0
        b_tran = b_occ - b_0

        a_add = np.where(a_tran == 1)[0] + 1
        a_lost = np.where(a_tran == -1)[0] + 1

        b_add = np.where(b_tran == 1)[0] + 1
        b_lost = np.where(b_tran == -1)[0] + 1

        print(f'{coeff:+.6f}  α:{a_str}  β:{b_str}', 'α:', a_lost, '->', a_add, 'β:', b_lost, '->', b_add)


def print_ref(ci_vecs, norbcas, neleccas, nocc_inact, ene):
    print('Ref:\n')
    for i, ci in enumerate(ci_vecs):
        s2, mult = spin_op.spin_square(ci, norb, sum(neleccas))
        print('\nstate', i, 'E:', e_df_fci[i], 'E_exci:', ene[i] - ene[0])
        print(f'<S^2> = {s2:.0f}, multiplicity = {mult:.0f}')
        print_ci(ci, norbcas, neleccas, nocc_inact)


np.set_printoptions(precision=8, suppress=True)  # suppress=True bỏ 'e-xx'

# ===== 1. Khai báo input: H2O =====
"""
4
Ammonia,^1A_1,CC3,aug-cc-pVTZ
N    0.06775910            0.00000000            0.00000000
H   -0.31382291            0.46874559           -0.81189118
H   -0.31382291           -0.93749118            0.00000000
H   -0.31382291            0.46874559            0.81189118
"""

nocc_inact = [2, 2]
nact = [10, 10]
nelec = [5, 5]
num_particles = [3, 3]
caslist_a = list(range(nocc_inact[0] + 1, nact[0] + nocc_inact[0] + 1))  # active orbital index (0-based)
caslist_b = caslist_a
caslist = [caslist_a, caslist_b]


n = 17

# Tạo phân tử
mol = gto.Mole()

mol.atom = [
    ['N', (0.06775910, 0.00000000, 0.00000000)],
    ['H', (-0.31382291, 0.46874559, -0.81189118)],
    ['H', (-0.31382291, -0.93749118, 0.00000000)],
    ['H', (-0.31382291, 0.46874559, 0.81189118)],
]

# mol.basis = "aug-cc-pVTZ"
mol.basis = 'cc-pVDZ'
mol.spin = 0
mol.build()

# ===== 2. Chạy UHF =====
myuhf = scf.UHF(mol)
myuhf.kernel()

# ===== 2.1 Casci
mycasci = mcscf.CASCI(myuhf, ncas=nact[0] + nocc_inact[0], nelecas=num_particles)
mycasci.fcisolver.nroots = n
e_casci, _, ci_vecs_casci, _, _ = mycasci.kernel()

# ===== 3. OBMP2 full-space =====
uobmp = UOBMP2(myuhf)
uobmp.second_order = True
uobmp.kernel()


# ===== 4. Sort MO theo caslist =====
mycas = mcscf.UCASCI(myuhf, ncas=nact[0], nelecas=sum(num_particles))
mo_sorted = mcscf.sort_mo(mycas, uobmp.mo_coeff, caslist)

# ===== 5. Tạo đối tượng downfold =====
uobact = UOBMP2_downfold(myuhf, nact=nact, nocc_act=num_particles)
uobact.mo_coeff = mo_sorted
uobact.mo_energy = uobmp.mo_energy
uobact.c0_tot = getattr(uobmp, 'c0_tot', None)
uobact.ene_tot = getattr(uobmp, 'ene_tot', None)
uobact.fock_hf = getattr(uobmp, 'fock_hf', None)
uobact.c1 = getattr(uobmp, 'c1', None)
uobact.second_order = True

# Sắp lại tmp1/tmp1_bar cho đúng thứ tự MO
uobact.tmp1 = uobact.sort_tmp1(uobmp.tmp1, caslist)
uobact.tmp1_bar = uobact.sort_tmp1(uobmp.tmp1_bar, caslist)

# ===== 6. Chạy kernel downfold =====
uobact.kernel()

# ===== 7. Lấy kết quả downfold =====
h1mo_act_eff = uobact.h1mo_act_eff  # ma trận 1-electron active-space
h2mo_act = uobact.h2mo_act  # tensor 2-electron active-space
ene_inact = uobact.ene_inact  # năng lượng inactive-space

"""
print("h1mo_act_eff:\n", h1mo_act_eff)
print("h2mo_act:\n", h2mo_act)
print("E_inact =", ene_inact)
"""

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
e_df_fci = e_df_fci_act + [ene_inact] * cis.nroots

e_df_fci_act *= 27.21138629
e_df_fci *= 27.21138629
e_casci *= 27.21138629

print('Ammonia ', nocc_inact[0], 'freeze', nact[0], 'active')

print('\nCasci')
print_ref(ci_vecs_casci, norb, num_particles, nocc_inact, e_casci)

print('\nDownfold + FCI')
print_ref(ci_vecs, norb, num_particles, nocc_inact, e_df_fci)


"""    
print("e_fci_act:", e_df_fci_act)
print("e_fci_tot:", e_df_fci)

e_exci_df_fci= e_df_fci - [e_df_fci[0]]*len(e_df_fci)
e_exci_df_fci= np.delete(e_exci_df_fci,0)
print("e_exci_fci_tot:", e_exci_df_fci)
"""
