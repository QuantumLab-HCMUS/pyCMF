import numpy as np
import pyscf
from pyscf import gto, scf, mcscf, lib
from pyscf.fci import direct_uhf
from pycmf.OBMP import UOBMP2
from pycmf.OBDF import UOBMP2_downfold
import basis_set_exchange as bse
import psutil

lib.param.MAX_MEMORY = 10240
print("lib.param.MAX_MEMORY = ", lib.param.MAX_MEMORY)
print("available memory = ", psutil.virtual_memory().available / 1024**3)

BASIS    = "aug-cc-pV(D+d)Z"

# Basis: aug-cc-pV(D+d)Z for Si triplet
si_basis = gto.basis.parse(bse.get_basis(f"{BASIS}", elements=["Si"], fmt="nwchem"))

# Active space for Si3 triplet
nocc_inact    = [15, 15]                 # inactive occupied (frozen) per spin
num_particles = [7, 5]                   # active electrons (alpha, beta): 22-15=7, 20-15=5
nalpha, nbeta = num_particles

caslist_a    = [16, 17, 18, 19, 20, 21, 22, 23, 24, 30, 32, 33]   
caslist_b    = [16, 17, 18, 19, 20, 21, 22, 23, 24, 30, 32, 33]   
caslist      = [caslist_a, caslist_b]
active_space = [(np.array(caslist_a) - 1).tolist(),
                (np.array(caslist_b) - 1).tolist()]              
nact         = [len(caslist_a), len(caslist_b)]
num_orbitals = len(caslist_a)
norb         = num_orbitals

# Build molecule

SIDE = 2.307
R    = SIDE / np.sqrt(3.0)                       # circumradius
triplet_atoms = [["Si",(R*np.cos(a), R*np.sin(a), 0.0)]
                 for a in np.radians([90.0, 210.0, 330.0])]

mol = gto.Mole()
mol.atom    = triplet_atoms
mol.basis   = {"Si": si_basis}
mol.unit    = 'A'
mol.charge  = 0
mol.spin    = 2            # 2S = 2 -> triplet (open shell, 2 unpaired e-)
mol.verbose = 5
mol.build()

print(f'active_space alpha: {caslist_a}')
print(f'active_space beta : {caslist_b}')

# UHF (unrestricted)
myuhf = pyscf.scf.UHF(mol)
e_uhf = myuhf.kernel()
ss_uhf, mult_uhf = myuhf.spin_square()

# UCASCI(12e in 12o, 7a/5b)
mycas = mcscf.UCASCI(myuhf, ncas=num_orbitals, nelecas=(nalpha, nbeta))
mo_sorted = mcscf.sort_mo(mycas, myuhf.mo_coeff, caslist, base=1)
e_casci, _, _ = mycas.kernel(mo_coeff=mo_sorted)

# UOBMP2 (full space)
uobmp = UOBMP2(myuhf)
uobmp.niter = 1 
uobmp.second_order = True
uobmp.kernel()
e_uobmp2 = getattr(uobmp, "ene_tot", None)
print(f"E(UOBMP2)                : {e_uobmp2:.13f}")
exit()

# UOBMP2 DOWNFOLDING
omp2_mo_sorted = mcscf.sort_mo(mycas, uobmp.mo_coeff, caslist)

uobact = UOBMP2_downfold(myuhf, nact=nact, nocc_act=num_particles)
uobact.mo_coeff     = omp2_mo_sorted
uobact.mo_energy    = uobmp.mo_energy
uobact.c0_tot       = getattr(uobmp, "c0_tot", None)
uobact.ene_tot      = getattr(uobmp, "ene_tot", None)
uobact.c1           = getattr(uobmp, "c1", None)
uobact.second_order = True

# re-sort tmp1/tmp1_bar and fock_hf to the sorted MO ordering
fock_temp       = mcscf.sort_mo(mycas, [uobmp.fock_hf[0], uobmp.fock_hf[1]], caslist)
uobact.fock_hf  = mcscf.sort_mo(mycas, [fock_temp[0].T, fock_temp[1].T], caslist)
uobact.tmp1     = uobact.sort_tmp1(uobmp.tmp1, caslist)
uobact.tmp1_bar = uobact.sort_tmp1(uobmp.tmp1_bar, caslist)

uobact.kernel()

h1mo_act_eff = uobact.h1mo_act_eff          
h2mo_act = uobact.h2mo_act                  
ene_inact = uobact.ene_inact                
     
h1a, h1b = h1mo_act_eff
h2aa = h2mo_act[0]
h2ab = h2mo_act[1]
h2bb = h2mo_act[3]

cis = direct_uhf.FCI()
cis.nroots = 1
e_dfold_fci, _ = cis.kernel((h1a,h1b), (h2aa,h2ab,h2bb), norb, (nalpha,nbeta))

E_triplet = e_dfold_fci + ene_inact

print("\n" + "=" * 64)
print(f"basis                    : {BASIS}")
print(f"active space             : ({sum(num_particles)}e, {norb}o)  alpha {caslist_a}")
print(f"                                       beta  {caslist_b}")
print(f"E(UHF)                   : {e_uhf:.13f}   <S^2> = {ss_uhf:.4f}")
print(f"E(UCASCI)                : {e_casci:.13f}")
print(f"E(UOBMP2)                : {e_uobmp2:.13f}")
print(f"E_inactive (downfold)    : {ene_inact:.13f}")
print(f"E_FCI(active)            : {e_dfold_fci:.13f}")
print(f"E(Si3 TRIPLET, DfoldFCI) : {E_triplet:.13f} Ha")
print("=" * 64)
