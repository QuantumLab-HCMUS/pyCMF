import numpy as np
import pyscf
from pyscf import gto, mcscf, lib
from pyscf.fci import direct_spin1
from pycmf.OBMP import OBMP2
from pycmf.OBDF import OBMP2_downfold
import basis_set_exchange as bse
import psutil

lib.param.MAX_MEMORY = 10240
print("lib.param.MAX_MEMORY = ", lib.param.MAX_MEMORY)
print("available memory = ", psutil.virtual_memory().available / 1024**3)


BASIS    = "aug-cc-pV(T+d)Z"

# Basis: aug-cc-pV(T+d)Z for Si singlet
si_basis = gto.basis.parse(bse.get_basis(f"{BASIS}", elements=["Si"], fmt="nwchem"))

# Active space for Si3 singlet
nocc_inact    = [15, 15]                 # inactive occupied (frozen) pairs
num_particles = [6, 6]                   # active electrons (alpha, beta)
nalpha, nbeta = num_particles

caslist_a    = [16, 17, 18, 19, 20, 21, 22, 23, 24, 30, 32, 33]   # 1-based
caslist_b    = caslist_a
caslist      = [caslist_a, caslist_b]
active_space = (np.array(caslist_a) - 1).tolist()                 # 0-based
nact         = [len(caslist_a), len(caslist_a)]
num_orbitals = len(active_space)
norb         = num_orbitals

# Build molecule
singlet_atoms = [["Si",(0.0, 1.438106,-0.548486)],
                 ["Si",(0.0,-1.438106,-0.548486)],
                 ["Si",(0.0, 0.0,      1.096973)]]

mol = gto.Mole()
mol.atom    = singlet_atoms
mol.basis   = {"Si": si_basis}
mol.unit    = 'A'
mol.charge  = 0
mol.spin    = 0            # 2S = 0 -> singlet (closed shell)
mol.verbose = 4
# mol.max_memory = 10000 # Không cần vì phía trên đã set lib.param.MAX_MEMORY = 10240
mol.build()

print(f'active_space: {caslist_a}')

# RHF (restricted)
myrhf = pyscf.scf.RHF(mol)
e_rhf = myrhf.kernel()

# CASCI(12,12)
mycas = mcscf.CASCI(myrhf, ncas=num_orbitals, nelecas=sum(num_particles))
mo = mycas.sort_mo(active_space, base=0)
e_casci = mycas.run().e_tot

hf_mo_sorted = mcscf.sort_mo(mycas, myrhf.mo_coeff, caslist_a)

hcore, nuclear_repulsion_energy = mycas.get_h1cas(hf_mo_sorted)
eri = pyscf.ao2mo.restore(1, mycas.get_h2cas(hf_mo_sorted), num_orbitals)

# OBMP2 (full space)
robmp = OBMP2(myrhf)
robmp.second_order = True
robmp.kernel()
e_obmp2 = getattr(robmp, "ene_tot", None)

# OBMP2 DOWNFOLDING
omp2_mo_sorted = mcscf.sort_mo(mycas, robmp.mo_coeff, caslist_a)

robact = OBMP2_downfold(myrhf, nact=nact[0], nocc_act=num_particles[0])
robact.mo_coeff     = omp2_mo_sorted
robact.mo_energy    = robmp.mo_energy
robact.c0_tot       = getattr(robmp, "c0_tot", None)
robact.ene_tot      = getattr(robmp, "ene_tot", None)
robact.c1           = getattr(robmp, "c1", None)
robact.second_order = True

# re-sort tmp1/tmp1_bar and fock_hf to the sorted MO ordering
fock_temp       = mcscf.sort_mo(mycas, robmp.fock_hf, caslist_a)
robact.fock_hf  = mcscf.sort_mo(mycas, fock_temp.T, caslist_a)
robact.tmp1     = robact.sort_tmp1(robmp.tmp1, caslist_a)
robact.tmp1_bar = robact.sort_tmp1(robmp.tmp1_bar, caslist_a)

robact.kernel()

h1        = robact.h1mo_act_eff      # effective 1-body in active space
h2        = robact.h2mo_act          # 2-body in active space
ene_inact = robact.ene_inact         # inactive (downfolded) energy

# DOWNFOLDING FCI
cis = direct_spin1.FCI()
cis.nroots = 1
e_dfold_fci, _ = cis.kernel(h1, h2, norb, (nalpha, nbeta))

E_singlet = e_dfold_fci + ene_inact

print("\n" + "=" * 64)
print(f"basis                    : {BASIS}")
print(f"active space             : ({sum(num_particles)}e, {norb}o)  orbitals {caslist_a}")
print(f"E(RHF)                   : {e_rhf:.13f}")
print(f"E(CASCI)                 : {e_casci:.13f}")
print(f"E(OBMP2)                 : {e_obmp2:.13f}")
print(f"E_inactive (downfold)    : {ene_inact:.13f}")
print(f"E_FCI(active)            : {e_dfold_fci:.13f}")
print(f"E(Si3 SINGLET, DfoldFCI) : {E_singlet:.13f} Ha")
print("=" * 64)
