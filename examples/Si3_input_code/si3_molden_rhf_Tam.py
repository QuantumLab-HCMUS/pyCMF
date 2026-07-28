import os
import numpy
from pyscf import gto, scf
from pyscf.tools import molden
import basis_set_exchange as bse
from pycmf.OBMP import OBMP2

FROZEN     = 0         # OBMP2 restricted of pycmf does NOT support frozen core.
MOLE       = "Si3"
ARRANGE    = "isosceles"
LABEL      = "singlet"       # tag appended to the molden file name.
BASIS_NAME = "aug-cc-pV(T+d)Z"
MOLDEN_DIR = "/Users/phungngocduy/Downloads/Nguyen_Tri_Vy/Molden"

# Basis: aug-cc-pV(T+d)Z for Si singlet
si_basis = gto.basis.parse(bse.get_basis(BASIS_NAME, elements=["Si"], fmt="nwchem"))


# Output folder for molden files
folder = os.path.join(MOLDEN_DIR, MOLE, ARRANGE, BASIS_NAME)
os.makedirs(folder, exist_ok=True)

# Build molecule
singlet_atoms = [["Si",(0.0, 1.438106,-0.548486)],
                 ["Si",(0.0,-1.438106,-0.548486)],
                 ["Si",(0.0, 0.0,      1.096973)]]

mol = gto.Mole()
mol.atom       = singlet_atoms
mol.basis      = {"Si": si_basis}
mol.charge     = 0
mol.spin       = 0            # 2S = 0 -> singlet (closed shell)
mol.symmetry   = False
mol.verbose    = 4
#mol.max_memory = 7000
mol.build()

# RHF (restricted)
mf = scf.RHF(mol)
mf.max_memory = 7000
mf.kernel()

# OBMP2 (orbital-optimized MP2)
obmp = OBMP2(mf, frozen=FROZEN, mo_coeff=mf.mo_coeff.copy(), mo_occ=mf.mo_occ.copy())
obmp.mo_energy    = mf.mo_energy.copy()
obmp.second_order = True
obmp.max_memory   = 7000
obmp.kernel()

E_singlet = obmp.ene_tot

# Write molden AFTER OBMP2 (restricted: 1 set of MOs, occ = 0/2)
stem       = f"{MOLE}_{ARRANGE}_{BASIS_NAME}_{LABEL}"
molden_rhf = os.path.join(folder, f"{stem}.molden")
molden.from_mo(mol, molden_rhf, obmp.mo_coeff, ene=obmp.mo_energy, occ=obmp.mo_occ)

print("\n" + "=" * 64)
print("basis                    : aug-cc-pV(T+d)Z")
print(f"frozen core orbitals     : {FROZEN}")
print(f"E(RHF)                   : {mf.e_tot:.13f}")
print(f"E_corr(OBMP2)            : {obmp.e_corr:.13f}")
print(f"E(Si3 SINGLET, OBMP2)    : {E_singlet:.13f} Ha")
print(f"molden file              : {molden_rhf}")
print("=" * 64)
