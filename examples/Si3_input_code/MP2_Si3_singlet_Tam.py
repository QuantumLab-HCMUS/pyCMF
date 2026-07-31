import numpy
from pyscf import gto, scf, mp
import basis_set_exchange as bse

HARTREE2KJ = 2625.499639
FROZEN     = 15                    # Si[Ne] core x 3 atoms (valence-only)
BASIS      = "aug-cc-pV(D+d)Z"
si_basis   = gto.basis.parse(bse.get_basis(BASIS, elements=["Si"], fmt="nwchem"))

# --- SINGLET geometry (Angstrom): the CT .xyz ---aA--------------------------
singlet_atoms = [["Si",(0.0, 1.438106,-0.548486)],
                 ["Si",(0.0,-1.438106,-0.548486)],
                 ["Si",(0.0, 0.0,      1.096973)]]

# Build molecule
mol = gto.Mole()
mol.atom       = singlet_atoms
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
print("basis                    : aug-cc-pV(D+d)Z")
print(f"frozen core orbitals     : {FROZEN}")
print(f"E(RHF)                   : {mf.e_tot:.13f}")
print(f"E_corr(MP2)              : {mymp.e_corr:.13f}")
print(f"E(Si3 SINGLET, MP2)      : {E_singlet:.13f} Ha")
print("=" * 64)
