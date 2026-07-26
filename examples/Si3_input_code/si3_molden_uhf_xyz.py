import os
import numpy
from pyscf import gto, scf
from pyscf.tools import molden
import basis_set_exchange as bse
from pycmf.OBMP import UOBMP2

XYZ_FILE   = "Si3-neutral-triplet_CT.xyz"
FROZEN     = 0         # UOBMP2 of pycmf does NOT support frozen core.
MOLE       = "Si3"
ARRANGE    = "isosceles"
LABEL      = "triplet"       # tag appended to the molden file name.
BASIS_NAME = "aug-cc-pV(D+d)Z"
MOLDEN_DIR = "/Users/phungngocduy/Downloads/Nguyen_Tri_Vy/Molden"

# Basis: aug-cc-pV(D+d)Z for Si triplet
si_basis = gto.basis.parse(bse.get_basis(BASIS_NAME, elements=["Si"], fmt="nwchem"))

# Read geometry from file
lines = open(XYZ_FILE).read().splitlines()
natom = int(lines[0])
atoms = []
for line in lines[2:2 + natom]:
    s = line.split()
    atoms.append([s[0], (float(s[1]), float(s[2]), float(s[3]))])

# Output folder for molden files
folder = os.path.join(MOLDEN_DIR, MOLE, ARRANGE, BASIS_NAME)
os.makedirs(folder, exist_ok=True)

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
print(f"geometry file            : {XYZ_FILE}")
print("basis                    : aug-cc-pV(D+d)Z")
print(f"frozen core orbitals     : {FROZEN}")
print(f"UHF <S^2>                : {mf.spin_square()[0]:.13f}")
print(f"E(UHF)                   : {mf.e_tot:.13f}")
print(f"E_corr(UOBMP2)           : {E_triplet - mf.e_tot:.13f}")
print(f"E(Si3 TRIPLET, UOBMP2)   : {E_triplet:.13f} Ha")
print(f"molden file (alpha)      : {molden_alpha}")
print(f"molden file (beta)       : {molden_beta}")
print("=" * 64)
