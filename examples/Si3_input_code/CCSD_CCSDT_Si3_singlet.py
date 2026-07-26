import numpy as np
import basis_set_exchange as bse
from pyscf import gto, scf, cc

XYZ_FILE = "Si3-neutral-singlet_CT.xyz"

# Basis: aug-cc-pV(D+d)Z for Si singlet
si_basis = gto.basis.parse(bse.get_basis("aug-cc-pV(D+d)Z", elements=["Si"], fmt="nwchem"))

# Read geometry from file
lines = open(XYZ_FILE).read().splitlines()
natom = int(lines[0])
atoms = []
for line in lines[2:2 + natom]:
    s = line.split()
    atoms.append([s[0], (float(s[1]), float(s[2]), float(s[3]))])

# Build molecule
mol = gto.Mole()
mol.atom       = atoms
mol.basis      = {"Si": si_basis}
mol.charge     = 0
mol.spin       = 0            # 2S = 0 -> singlet (closed shell)
mol.symmetry   = False
mol.verbose    = 4
mol.max_memory = 7000
mol.build()

# RHF (restricted)
mf = scf.RHF(mol)
mf.max_memory = 7000
mf.kernel()

# RCCSD(T)
mycc = cc.CCSD(mf)
mycc.frozen          = 15
mycc.incore_complete = True
mycc.max_memory      = 7000
mycc.kernel()
et = mycc.ccsd_t()
E_singlet_CCSD = mf.e_tot + mycc.e_corr
E_singlet = mf.e_tot + mycc.e_corr + et

print("\n" + "=" * 64)
print(f"geometry file            : {XYZ_FILE}")
print(f"basis                    : aug-cc-pV(D+d)Z   frozen core = 15")
print(f"E(RHF)                   : {mf.e_tot:.10f}")
print(f"E_corr(CCSD)             : {mycc.e_corr:.10f}")
print(f"E(T)                     : {et:.10f}")
print(f"T1 diagnostic            : {mycc.get_t1_diagnostic():.4f}")
print(f"E(Si3 SINGLET, CCSD(T))  : {E_singlet_CCSD:.13f} Ha")
print(f"E(Si3 SINGLET, CCSD(T))  : {E_singlet:.13f} Ha")
print("=" * 64)
