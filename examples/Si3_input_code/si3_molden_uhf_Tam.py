import os
import numpy
from pyscf import gto, scf
from pyscf.tools import molden
import basis_set_exchange as bse
from pycmf.OBMP import UOBMP2
import numpy as np

FROZEN     = 0         # UOBMP2 of pycmf does NOT support frozen core.
MOLE       = "Si3"
ARRANGE    = "isosceles"
LABEL      = "triplet"       # tag appended to the molden file name.
BASIS_NAME = "aug-cc-pV(T+d)Z"
MOLDEN_DIR = "/Users/phungngocduy/Downloads/Nguyen_Tri_Vy/Molden"

# Basis: aug-cc-pV(T+d)Z for Si triplet
si_basis = gto.basis.parse(bse.get_basis(BASIS_NAME, elements=["Si"], fmt="nwchem"))

# Output folder for molden files
folder = os.path.join(MOLDEN_DIR, MOLE, ARRANGE, BASIS_NAME)
os.makedirs(folder, exist_ok=True)

# Build molecule
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
mol.verbose    = 5
mol.max_memory = 7000
mol.build()

# UHF (unrestricted) -- built ONCE
mf = scf.UHF(mol)
mf.max_memory = 7000
mf.kernel()

# UOBMP2 (orbital-optimized UMP2)
uobmp = UOBMP2(mf, frozen=FROZEN,
               mo_coeff=numpy.array(mf.mo_coeff, copy=True),
               mo_occ=numpy.array(mf.mo_occ, copy=True))
uobmp.mo_energy    = numpy.array(mf.mo_energy, copy=True)
uobmp.second_order = True
uobmp.max_memory   = 7000
uobmp.kernel()

E_triplet = uobmp.ene_tot

# Write molden AFTER UOBMP2 (unrestricted: 2 sets of MOs, alpha and beta)
stem         = f"{MOLE}_{ARRANGE}_{BASIS_NAME}_{LABEL}"
molden_alpha = os.path.join(folder, f"{stem}_alpha.molden")
molden_beta  = os.path.join(folder, f"{stem}_beta.molden")
molden.from_mo(mol, molden_alpha, uobmp.mo_coeff[0],
               ene=uobmp.mo_energy[0], occ=mf.mo_occ[0], spin='Alpha')
molden.from_mo(mol, molden_beta,  uobmp.mo_coeff[1],
               ene=uobmp.mo_energy[1], occ=mf.mo_occ[1], spin='Beta')

print("\n" + "=" * 64)
print("basis                    : aug-cc-pV(T+d)Z")
print(f"frozen core orbitals     : {FROZEN}")
print(f"UHF <S^2>                : {mf.spin_square()[0]:.13f}")
print(f"E(UHF)                   : {mf.e_tot:.13f}")
print(f"E_corr(UOBMP2)           : {E_triplet - mf.e_tot:.13f}")
print(f"E(Si3 TRIPLET, UOBMP2)   : {E_triplet:.13f} Ha")
print(f"molden file (alpha)      : {molden_alpha}")
print(f"molden file (beta)       : {molden_beta}")
print("=" * 64)
