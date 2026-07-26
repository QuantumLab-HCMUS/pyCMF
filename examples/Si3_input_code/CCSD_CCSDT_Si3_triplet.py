import numpy as np
import basis_set_exchange as bse
from pyscf import gto, scf, cc

XYZ_FILE = "Si3-neutral-triplet_CT.xyz"

# Basis: aug-cc-pV(D+d)Z for Si triplet
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
mol.spin       = 2            # 2S = 2 -> triplet (open shell)
mol.symmetry   = False
mol.verbose    = 4
mol.max_memory = 7000
mol.build()

# UHF (unrestricted)
mf = scf.UHF(mol)
mf.max_memory = 7000
mf.kernel()

# UCCSD(T)
mycc = cc.UCCSD(mf)
mycc.frozen          = 15
mycc.incore_complete = True
mycc.max_memory      = 7000
mycc.kernel()
et = mycc.ccsd_t()
E_triplet_CCSD = mf.e_tot + mycc.e_corr
E_triplet = mf.e_tot + mycc.e_corr + et

print("\n" + "=" * 64)
print(f"geometry file            : {XYZ_FILE}")
print(f"basis                    : aug-cc-pV(D+d)Z   frozen core = 15")
print(f"UHF <S^2>                : {mf.spin_square()[0]:.4f}")
print(f"E(UHF)                   : {mf.e_tot:.10f}")
print(f"E_corr(CCSD)             : {mycc.e_corr:.10f}")
print(f"E(T)                     : {et:.10f}")
print(f"E(Si3 TRIPLET, CCSD(T))  : {E_triplet_CCSD:.13f} Ha")
print(f"E(Si3 TRIPLET, CCSD(T))  : {E_triplet:.13f} Ha")
print("=" * 64)
