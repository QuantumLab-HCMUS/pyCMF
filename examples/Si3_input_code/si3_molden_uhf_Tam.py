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
BASIS_NAME = "cc-pVTZ"
MOLDEN_DIR = "/Users/phungngocduy/Downloads/Nguyen_Tri_Vy/Molden"
cutoff      = 1e-2

# Basis: cc-pVTZ for Si triplet
#si_basis = gto.basis.parse(bse.get_basis(BASIS_NAME, elements=["Si"], fmt="nwchem"))

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
#mol.basis      = {"Si": si_basis}
mol.basis      = BASIS_NAME
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
#uobmp = UOBMP2(mf, frozen=FROZEN,
#               mo_coeff=numpy.array(mf.mo_coeff, copy=True),
#               mo_occ=numpy.array(mf.mo_occ, copy=True))
#uobmp.mo_energy    = numpy.array(mf.mo_energy, copy=True)
uobmp = UOBMP2(mf)
uobmp.second_order = True
uobmp.max_memory   = 7000
uobmp.kernel()

e_uobmp2 = getattr(uobmp, "ene_tot", None)
import numpy as np 
S = mol.intor("int1e_ovlp")
eigval, eigvec = np.linalg.eigh(S)

S12 = eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T


"""Cho Unrestricted
for j in range(mol.nao_nr()):
print(f"\n===== MO {j} =====  energy: {robmp.mo_energy[0][j]:12.6f} ===== "
      f"normalization: {robmp.mo_coeff[0][:,j].T @ S @ robmp.mo_coeff[0][:,j]}")
"""
for s in [0, 1]:
    print(f"\n===== SPIN {s} =====" )
    print("#==================================================")
    print("\n")
    print("AO contribution")
    for j in range(mol.nao_nr()):
        print(f"\n===== MO {j} =====  energy: {uobmp.mo_energy[s][j]:12.6f} ===== normalization: {uobmp.mo_coeff[s][:,j].T @ S @ uobmp.mo_coeff[s][:,j]}" )
        prob = (S12 @ uobmp.mo_coeff[s][:,j])**2
        prob /= prob.sum()
        print(f"orth: {sum(prob)}")
        C = uobmp.mo_coeff[s][:,j]
        contrbS = C*(S@C)
        print(f"norm: {sum(contrbS)}")
        for ao, coeff in enumerate(uobmp.mo_coeff[s][:,j]):
            if abs(coeff) < 1e-15:
                coeff = 0
            else: coeff
            if prob[ao] > cutoff:
                print(f"{ao:3d}: {mol.ao_labels()[ao]:15s}: {coeff:12.6f};  prob: {prob[ao]:12.6f}; contrb: {contrbS[ao]:12.6f}")
    print("#==================================================")
    print("\n")
    #==============================OBMP2 full-space==============================

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
print(f"basis                    : {BASIS_NAME}")
#print(f"frozen core orbitals     : {FROZEN}")
print(f"UHF <S^2>                : {mf.spin_square()[0]:.13f}")
print(f"E(UHF)                   : {mf.e_tot:.13f}")
print(f"E_corr(UOBMP2)           : {E_triplet - mf.e_tot:.13f}")
print(f"E(Si3 TRIPLET, UOBMP2)   : {E_triplet:.13f} Ha")
print(f"molden file (alpha)      : {molden_alpha}")
print(f"molden file (beta)       : {molden_beta}")
print("=" * 64)
