import numpy as np
import basis_set_exchange as bse
from pyscf import gto, scf, cc

HARTREE2KJ = 2625.499639
FROZEN     = 15                    # Si[Ne] core x 3 atoms (valence-only)
BASIS      = "aug-cc-pV(D+d)Z"
#BASIS      = "aug-cc-pV(T+d)Z"
si_basis   = gto.basis.parse(bse.get_basis(BASIS, elements=["Si"], fmt="nwchem"))

# --- TRIPLET geometry: equilateral triangle, side = 2.307 A ---------------
SIDE = 2.307
R    = SIDE / np.sqrt(3.0)                       # circumradius
triplet_atoms = [["Si",(R*np.cos(a), R*np.sin(a), 0.0)]
                 for a in np.radians([90.0, 210.0, 330.0])]
# Build molecule
mol = gto.Mole()
mol.atom       = triplet_atoms
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
mycc.frozen          = FROZEN
mycc.incore_complete = True
mycc.max_memory      = 7000
mycc.kernel()
et = mycc.ccsd_t()
E_triplet_CCSD = mf.e_tot + mycc.e_corr
E_triplet = mf.e_tot + mycc.e_corr + et

print("\n" + "=" * 64)
#print(f"geometry file            : {XYZ_FILE}")
print(f"basis                    : aug-cc-pV(D+d)Z   frozen core = 15")
print(f"UHF <S^2>                : {mf.spin_square()[0]:.4f}")
print(f"E(UHF)                   : {mf.e_tot:.10f}")
print(f"E_corr(CCSD)             : {mycc.e_corr:.10f}")
print(f"E(T)                     : {et:.10f}")
print(f"E(Si3 TRIPLET, CCSD)  : {E_triplet_CCSD:.13f} Ha")
print(f"E(Si3 TRIPLET, CCSD(T))  : {E_triplet:.13f} Ha")
print("=" * 64)
