import numpy
from pyscf import gto, scf, mp
import basis_set_exchange as bse

XYZ_FILE   = "Si3-neutral-singlet_CT.xyz"
FROZEN     = 15        # Si [Ne] core = 5 orb/atom x 3 atoms.

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

# RMP2
mymp = mp.MP2(mf)
mymp.frozen     = FROZEN
mymp.max_memory = 7000
mymp.kernel()

E_singlet = mf.e_tot + mymp.e_corr

print("\n" + "=" * 64)
print(f"geometry file            : {XYZ_FILE}")
print("basis                    : aug-cc-pV(D+d)Z")
print(f"frozen core orbitals     : {FROZEN}")
print(f"E(RHF)                   : {mf.e_tot:.13f}")
print(f"E_corr(MP2)              : {mymp.e_corr:.13f}")
print(f"E(Si3 SINGLET, MP2)      : {E_singlet:.13f} Ha")
print("=" * 64)
