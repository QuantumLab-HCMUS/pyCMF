import numpy as np
import pyscf
from pyscf import gto, mcscf, lib, mrpt
from pycmf.OBMP import OBMP2
import basis_set_exchange as bse
import psutil

lib.param.MAX_MEMORY = 10240
print("lib.param.MAX_MEMORY = ", lib.param.MAX_MEMORY)
print("available memory = ", psutil.virtual_memory().available / 1024**3)

BASIS    = "cc-pVTZ"

nocc_inact    = [15, 15]                 
num_particles = [6, 6]
nalpha, nbeta = num_particles

caslist_a    = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 32]   # 1-based
caslist_b    = caslist_a
caslist      = [caslist_a, caslist_b]
active_space = (np.array(caslist_a) - 1).tolist()                 # 0-based
nact         = [len(caslist_a), len(caslist_a)]
num_orbitals = len(active_space)
norb         = num_orbitals

R = 2.185333880
for angle in [160, 170]:    

    print(f"\nangle = {angle} degrees")
    singlet_atoms= [
            ['Si', (0.0, 0.0, 0.0)],
            ['Si', (0.0, -R * np.sin(angle * np.pi / 180.0), R * np.cos(angle * np.pi / 180.0))],
            ['Si', (0.0, 0.0, R)]
        ]

    mol = gto.Mole()
    mol.atom    = singlet_atoms
    mol.basis   = BASIS
    mol.unit    = 'A'
    mol.charge  = 0
    mol.spin    = 0            
    mol.verbose = 5
    mol.build()

    # RHF
    myrhf = pyscf.scf.RHF(mol)
    e_rhf = myrhf.kernel()

    # CASSCF 
    mycas = mcscf.CASSCF(myrhf, ncas=num_orbitals, nelecas=sum(num_particles))
    mo = mcscf.sort_mo(casscf=mycas, mo_coeff=myrhf.mo_coeff, caslst=caslist_a, base=1)
    e_cas = mycas.kernel(mo)[0]

    # NEVPT2
    nev           = mrpt.NEVPT(mycas)
    nev.max_memory = 7000
    e_corr_pt2    = nev.kernel()
    e_nevpt2_tot  = e_cas + e_corr_pt2

    print("\n" + "=" * 64)
    print(f"Góc = {angle} độ")
    print(f"basis                    : {BASIS}")
    print(f"active space             : ({sum(num_particles)}e, {norb}o)  orbitals {caslist_a}")
    print(f"E(RHF)                   : {e_rhf:.13f}")
    print(f"E(CASSCF)                : {e_cas:.13f}")
    print(f"E_corr(NEVPT2)           : {e_corr_pt2:.13f}")
    print(f"E(NEVPT2_Total)          : {e_nevpt2_tot:.13f}")
    print("=" * 64)