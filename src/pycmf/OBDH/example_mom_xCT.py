import time
from pyscf import scf, gto

# Import the refactored module containing both OBDH and OBMP2
# Adjust the import path if your file name is different (e.g., from pycmf.OBDH.main import OBDH, OBMP2)
from pycmf.OBDH import OBDH_CL, OBMP2_CL

mol = gto.Mole()
mol.atom = '''
O    0.000000   0.000000   0.000000
H    0.000000   0.000000   0.962000
H    1.790000   1.026000   0.000000
'''
mol.spin = 0
mol.verbose = 4
mol.basis = 'ccpvdz'
mol.build()

# Run UHF once to serve as the reference for all 4 cases
print("Running initial Full System UHF...")
a = scf.UHF(mol).density_fit().run()
print(a.energy_elec())

mppp_obdh_std = OBDH_CL(a)
mppp_obdh_std.alphaa = (0.53, 0.39)
mppp_obdh_std.thresh = 1e-06
mppp_obdh_std.second_order = True
mppp_obdh_std.mom_select = False
mppp_obdh_std.is_hybrid =  True
#mppp_obdh_std.is_hybrid =  False
mppp_obdh_std.use_embed = False  # Disable Embedding
mppp_obdh_std.use_cl = False     # Ignored when use_embed is False

mppp_obdh_std.run()

mo0 = a.mo_coeff
occ = a.mo_occ


# Assign initial occupation pattern
occ[0][4]=0      # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
occ[0][6]=1     # is still a singlet state

# New SCF caculation 
b = scf.UHF(mol).density_fit()
# Construct new dnesity matrix with new occpuation pattern
dm_u = b.make_rdm1(mo0, occ)
# Apply mom occupation principle
b = scf.addons.mom_occ(b, mo0, occ)
# Start new SCF with new density matrix
b.scf(dm_u)
print(b.mo_energy)

print("UHF GS energy:  ", a.e_tot)
print("UHF xCT energy: ", b.e_tot)
print("Delta E (eV):   ", (b.e_tot - a.e_tot) * 27.2114)
print("Alpha occ:", b.mo_occ[0])
print("Beta occ: ", b.mo_occ[1])

# ==============================================================================
# CASE 1: STANDARD OBDH (HYBRID) - NO EMBEDDING
# ==============================================================================
print("\n" + ">"*10 + " CASE 1: STANDARD OBDH (NO EMBEDDING) " + "<"*10)
mppp_obdh_std = OBDH_CL(b)
mppp_obdh_std.alphaa = (0.53, 0.39)
mppp_obdh_std.thresh = 1e-08
mppp_obdh_std.second_order = True
mppp_obdh_std.mom_select = True
mppp_obdh_std.mom_start_cycle = 0
mppp_obdh_std.is_hybrid =  True
#mppp_obdh_std.is_hybrid =  False
mppp_obdh_std.use_embed = False  # Disable Embedding
mppp_obdh_std.use_cl = False     # Ignored when use_embed is False

start1 = time.time()
mppp_obdh_std.run()
print('=> Runtime (OBDH Standard): {:.4f} seconds'.format(time.time() - start1))
