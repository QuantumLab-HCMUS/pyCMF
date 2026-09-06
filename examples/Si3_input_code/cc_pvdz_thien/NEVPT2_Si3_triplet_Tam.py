import numpy as np
import pyscf
from pyscf import gto, scf, mcscf, lib, mrpt
import sys
sys.path.append('/Users/phungngocduy/Documents/Luan_van/Github/pyCMF/src')  # Thêm đường dẫn đến thư mục src để import OBMP và OBDF
import basis_set_exchange as bse
import psutil


lib.param.MAX_MEMORY = 10240
print("lib.param.MAX_MEMORY = ", lib.param.MAX_MEMORY)
print("available memory = ", psutil.virtual_memory().available / 1024**3)


#BASIS    = "cc-pVDZ"
BASIS    = "cc-pVTZ"

# Basis: aug-cc-pV(T+d)Z for Si triplet
#si_basis = gto.basis.parse(bse.get_basis(f"{BASIS}", elements=["Si"], fmt="nwchem"))

# Active space for Si3 triplet
nocc_inact    = [15, 15]                 # inactive occupied (frozen) per spin
num_particles = [7, 5]                   # active electrons (alpha, beta): 22-15=7, 20-15=5
nalpha, nbeta = num_particles

#caslist_a    = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29]   
#caslist_b    = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29]   
caslist_a    = [16,17,18,19,20,21,22,23,24,25,26,33]   
caslist_b    = [16,17,18,19,20,21,22,23,24,25,26,33]
#caslist_a    = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 34]   
#caslist_b    = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 34]   

caslist      = [caslist_a, caslist_b]
active_space = [(np.array(caslist_a) - 1).tolist(),
                (np.array(caslist_b) - 1).tolist()]              
nact         = [len(caslist_a), len(caslist_b)]
num_orbitals = len(caslist_a)
norb         = num_orbitals

# Build molecule

#SIDE = 2.307
#R    = SIDE / np.sqrt(3.0)                       # circumradius
R = 2.295097609
angle = 140.0 
for angle in [90, 100, 110]:    

    triplet_atoms = [
            ['Si', (0.0, 0.0, 0.0)],
            ['Si', (0.0, -R * np.sin(angle * np.pi / 180.0), R * np.cos(angle * np.pi / 180.0))],
            ['Si', (0.0, 0.0, R)]
        ]

    mol = gto.Mole()
    mol.atom    = triplet_atoms
    #mol.basis   = {"Si": si_basis}
    mol.basis   = BASIS
    mol.unit    = 'A'
    mol.charge  = 0
    mol.spin    = 2            # 2S = 2 -> triplet (open shell, 2 unpaired e-)
    mol.verbose = 5
    mol.build()

    print(f'active_space alpha: {caslist_a}')
    print(f'active_space beta : {caslist_b}')

    # ROHF (Restricted Open-Shell Hartree-Fock)
    myrohf = pyscf.scf.ROHF(mol)
    e_rohf = myrohf.kernel()
    ss_rohf, mult_rohf = myrohf.spin_square()

    # ROCASSCF
    mycas = mcscf.CASSCF(myrohf, ncas=num_orbitals, nelecas=(nalpha, nbeta))
    mo_sorted = mcscf.sort_mo(casscf=mycas, mo_coeff=myrohf.mo_coeff, caslst=caslist_a, base=1)
    e_cas = mycas.kernel(mo_coeff=mo_sorted)[0]

    # NEVPT2
    nev = mrpt.NEVPT(mycas)
    nev.max_memory = 7000
    e_corr_pt2 = nev.kernel()
    e_nevpt2_tot = e_cas + e_corr_pt2

    print("\n" + "=" * 64)
    print(f"Góc = {angle} độ")
    print(f"basis                    : {BASIS}")
    print(f"active space             : ({sum(num_particles)}e, {norb}o)  alpha {caslist_a}")
    print(f"                                       beta  {caslist_b}")
    print(f"E(ROHF)                  : {e_rohf:.13f}   <S^2> = {ss_rohf:.4f}")
    print(f"E(CASSCF)                : {e_cas:.13f}")
    print(f"E_corr(NEVPT2)           : {e_corr_pt2:.13f}")
    print(f"E(NEVPT2_Total)          : {e_nevpt2_tot:.13f} Ha")
    print("=" * 64)