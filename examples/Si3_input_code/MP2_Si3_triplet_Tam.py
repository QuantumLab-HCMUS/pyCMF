import numpy
from pyscf import gto, scf, mp
import basis_set_exchange as bse
import numpy as np

HARTREE2KJ = 2625.499639
FROZEN     = 15                    # Si[Ne] core x 3 atoms (valence-only)
BASIS      = "aug-cc-pV(D+d)Z"
#BASIS      = "aug-cc-pV(T+d)Z"
si_basis   = gto.basis.parse(bse.get_basis(BASIS, elements=["Si"], fmt="nwchem"))

# Build molecule
# --- TRIPLET geometry: equilateral triangle, side = 2.307 A ---------------
SIDE = 2.307
R    = SIDE / np.sqrt(3.0)                       # circumradius
triplet_atoms = [["Si",(R*np.cos(a), R*np.sin(a), 0.0)]
                 for a in np.radians([90.0, 210.0, 330.0])]
mol = gto.Mole()
mol.atom       = triplet_atoms
mol.basis      = {"Si": si_basis}
mol.charge     = 0
mol.spin       = 2            # 2S = 2 -> triplet (open shell)
mol.symmetry   = False
mol.verbose    = 4
mol.max_memory = 7000
mol.build()

# UHF (unrestricted) -- built ONCE
mf = scf.UHF(mol)
mf.max_memory = 7000
mf.kernel()

# UMP2
mymp = mp.UMP2(mf)
mymp.frozen     = FROZEN
mymp.max_memory = 7000
mymp.kernel()

E_triplet = mf.e_tot + mymp.e_corr

print("\n" + "=" * 64)
print("basis                    : aug-cc-pV(D+d)Z")
print(f"frozen core orbitals     : {FROZEN}")
print(f"UHF <S^2>                : {mf.spin_square()[0]:.13f}")
print(f"E(UHF)                   : {mf.e_tot:.13f}")
print(f"E_corr(UMP2)             : {mymp.e_corr:.13f}")
print(f"E(Si3 TRIPLET, UMP2)     : {E_triplet:.13f} Ha")
print("=" * 64)
