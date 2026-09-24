import time
from pyscf import scf, gto
import numpy as np
# Import the refactored module containing both OBDH and OBMP2
# Adjust the import path if your file name is different (e.g., from pycmf.OBDH.main import OBDH, OBMP2)
from pycmf.OBDH import OBDH_CL, OBMP2_CL
from pycmf.OBDH.stability import stabilize_scf

mol = gto.Mole()
mol.atom = '''
C   -1.061709204   1.297140572   0.292060003
O   -0.358161116   2.270458613   0.531812668
O   -0.589303516   0.094917758   0.003788813
H    0.404435659   0.127722621   0.018411838
C   -2.558427798   1.342549823   0.296257320
H   -2.895997978   2.347464002   0.518316340
H   -2.932889278   1.022390451  -0.672995551
H   -2.937211960   0.644910433   1.039557084
C    2.630513912   1.107716378   0.269968222
O    1.926895547   0.134401890   0.030180620
O    2.158087578   2.310151766   0.557746693
H    1.164299038   2.277253532   0.543346189
C    4.127226360   1.061813632   0.268003827
H    4.464805924   0.060494439   0.030478332
H    4.508724905   1.772166571  -0.461465071
H    4.498742671   1.364508149   1.244059188
'''

mol.charge = 0
mol.spin = 0
mol.verbose = 0
mol.basis = 'sto-3g'
mol.build()

# Run UHF once to serve as the reference for all 4 cases
print("Running initial Full System UHF...")
mf = scf.UHF(mol).density_fit()
mf.kernel()

mf = stabilize_scf(mf, max_macro_cycles=10, verbose=True)

# # ==============================================================================
# # CASE 1: STANDARD OBDH (HYBRID) - NO EMBEDDING
# # ==============================================================================
# print("\n" + ">"*10 + " CASE 1: STANDARD OBDH (NO EMBEDDING) " + "<"*10)
# mppp_obdh_std = OBMP2_CL(mf)
# mppp_obdh_std.alphaa = (0.53, 0.39)
# mppp_obdh_std.thresh = 1e-08
# mppp_obdh_std.second_order = True
# mppp_obdh_std.mom_select= False
# mppp_obdh_std.mom_start_cycle = 0
# mppp_obdh_std.use_embed = False  # Disable Embedding
# mppp_obdh_std.use_cl = False     # Ignored when use_embed is False

# start1 = time.time()
# mppp_obdh_std.run()
# print(mppp_obdh_std.converged)
# print(mppp_obdh_std.dip_mom)
# #print(mppp_obdh_std.mulliken_charges)
# print('=> Runtime (OBDH Standard): {:.4f} seconds'.format(time.time() - start1))


# ==============================================================================
# CASE 2: EMBEDDED OBDH (HYBRID) + CONCENTRIC LOCALIZATION (CL)
# ==============================================================================
print("\n" + ">"*10 + " CASE 2: EMBEDDED OBDH + CL TRUNCATION " + "<"*10)
mppp_obdh_emb = OBMP2_CL(mf)
mppp_obdh_emb.alphaa = (0.53, 0.39)
mppp_obdh_emb.thresh = 1e-06
mppp_obdh_emb.second_order = True
mppp_obdh_emb.use_embed = True   # Enable Embedding
mppp_obdh_emb.active_atoms = [0, 1, 2, 3, 8, 9, 10, 11] # Set active system to Oxygen atom
mppp_obdh_emb.mu = 1e6
mppp_obdh_emb.use_cl = False      # Enable CL Truncation
mppp_obdh_emb.n_shells = 1
mppp_obdh_emb.xc_env = 'pbe'

start2 = time.time()
mppp_obdh_emb.run()
print('=> Runtime (OBDH Embed + CL): {:.4f} seconds'.format(time.time() - start2))


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

