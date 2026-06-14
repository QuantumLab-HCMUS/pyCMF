import time
from pyscf import scf, gto

# Import the refactored module containing both OBDH and OBMP2
# Adjust the import path if your file name is different (e.g., from pycmf.OBDH.main import OBDH, OBMP2)
from pycmf.OBDH import OBDH_CL, OBMP2_CL

mol = gto.Mole()
mol.atom = '''
   O     -2.833000   -0.564000    0.366000
   H     -2.950000   -0.421000    1.305000
   H     -3.721000   -0.681000    0.029000
'''
mol.spin = 0
mol.verbose = 0
mol.basis = 'cc-pvdz'
mol.build()

# Run UHF once to serve as the reference for all 4 cases
print("Running initial Full System UHF...")
mf = scf.UHF(mol).density_fit().run()


# ==============================================================================
# CASE 1: STANDARD OBDH (HYBRID) - NO EMBEDDING
# ==============================================================================
print("\n" + ">"*10 + " CASE 1: STANDARD OBDH (NO EMBEDDING) " + "<"*10)
mppp_obdh_std = OBDH_CL(mf)
mppp_obdh_std.alphaa = (0.53, 0.39)
mppp_obdh_std.thresh = 1e-06
mppp_obdh_std.second_order = True
mppp_obdh_std.mom_select= False
mppp_obdh_std.mom_start_cycle = 0
mppp_obdh_std.use_embed = False  # Disable Embedding
mppp_obdh_std.use_cl = False     # Ignored when use_embed is False

start1 = time.time()
mppp_obdh_std.run()
print(mppp_obdh_std.dip_mom)
print(mppp_obdh_std.mulliken_charges)
print('=> Runtime (OBDH Standard): {:.4f} seconds'.format(time.time() - start1))


# # ==============================================================================
# # CASE 2: EMBEDDED OBDH (HYBRID) + CONCENTRIC LOCALIZATION (CL)
# # ==============================================================================
# print("\n" + ">"*10 + " CASE 2: EMBEDDED OBDH + CL TRUNCATION " + "<"*10)
# mppp_obdh_emb = OBDH_CL(mf)
# mppp_obdh_emb.alphaa = (0.53, 0.39)
# mppp_obdh_emb.thresh = 1e-06
# mppp_obdh_emb.second_order = True

# mppp_obdh_emb.use_embed = True   # Enable Embedding
# mppp_obdh_emb.active_atoms = [0] # Set active system to Oxygen atom
# mppp_obdh_emb.mu = 1e6
# mppp_obdh_emb.use_cl = True      # Enable CL Truncation
# mppp_obdh_emb.n_shells = 1

# start2 = time.time()
# mppp_obdh_emb.run()
# print('=> Runtime (OBDH Embed + CL): {:.4f} seconds'.format(time.time() - start2))


# # ==============================================================================
# # CASE 3: STANDARD OBMP2 (PURE) - NO EMBEDDING
# # ==============================================================================
# print("\n" + ">"*10 + " CASE 3: STANDARD OBMP2 (NO EMBEDDING) " + "<"*10)
# mppp_obmp2_std = OBMP2_CL(mf)
# mppp_obmp2_std.thresh = 1e-08
# mppp_obmp2_std.second_order = True
# mppp_obmp2_std.mom_select = True
# mppp_obmp2_std.use_embed = False  # Disable Embedding
# mppp_obmp2_std.use_cl = False     # Ignored when use_embed is False

# start3 = time.time()
# mppp_obmp2_std.run()
# print('=> Runtime (OBMP2 Standard): {:.4f} seconds'.format(time.time() - start3))


# # ==============================================================================
# # CASE 4: EMBEDDED OBMP2 (PURE) + CONCENTRIC LOCALIZATION (CL)
# # ==============================================================================
# print("\n" + ">"*10 + " CASE 4: EMBEDDED OBMP2 + CL TRUNCATION " + "<"*10)
# mppp_obmp2_emb = OBMP2_CL(mf)
# mppp_obmp2_emb.thresh = 1e-06
# mppp_obmp2_emb.second_order = True

# mppp_obmp2_emb.use_embed = True   # Enable Embedding
# mppp_obmp2_emb.active_atoms = [0] # Set active system to Oxygen atom
# mppp_obmp2_emb.mu = 1e6
# mppp_obmp2_emb.use_cl = True      # Enable CL Truncation
# mppp_obmp2_emb.n_shells = 1

# start4 = time.time()
# mppp_obmp2_emb.run()
# print('=> Runtime (OBMP2 Embed + CL): {:.4f} seconds'.format(time.time() - start4))

# print("\n" + "="*60)
# print("ALL 4 CASES COMPLETED SUCCESSFULLY!")
# print("="*60)

