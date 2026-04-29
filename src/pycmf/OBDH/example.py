import time
from pyscf import scf, gto
from pyCMF.src.pycmf.OBDH.main import UB2PLYPDFUOBMP2

mol = gto.Mole()
mol.atom = '''
O    0.000000   0.000000   0.000000
H    0.000000   0.000000   0.962000
C    1.430000   0.000000   0.000000
H    1.790000   1.026000   0.000000
H    1.790000  -0.513000   0.889000
C    1.943000  -0.513000  -1.334000
H    1.583000   0.051000  -2.200000
H    1.583000  -1.540000  -1.334000
C    3.463000  -0.513000  -1.334000
H    3.823000   0.051000  -0.468000
H    3.823000  -1.540000  -1.334000
C    3.976000  -0.000000  -2.668000
H    3.616000   1.027000  -2.668000
H    3.616000  -0.513000  -3.534000
C    5.496000  -0.000000  -2.668000
H    5.856000  -1.027000  -2.668000
H    5.856000   0.513000  -1.802000
C    6.009000   0.513000  -4.002000
H    5.649000   1.540000  -4.002000
H    5.649000   0.000000  -4.868000
H    7.099000   0.513000  -4.002000
'''
mol.spin = 0
mol.spin = 0
mol.verbose = 0
mol.basis = 'sto-3g'
mol.build()

print("\n>>>>>>>> CHẠY CHẾ ĐỘ: KHÔNG CÓ EMBEDDING (OBDH CHUẨN) <<<<<<<<")
mf_std = scf.UHF(mol).density_fit().run()
mppp_std = UB2PLYPDFUOBMP2(mf_std)
mppp_std.alphaa = (0.53, 0.39)
mppp_std.thresh = 1e-06
mppp_std.use_embed = False  # Tắt Embedding
# mppp_std.use_cl = True    # Dù bật cũng sẽ bị bỏ qua

start = time.time()
mppp_std.run()
print('=> Thời gian chạy (No Embed): ', time.time() - start)

print("\n\n>>>>>>>> CHẠY CHẾ ĐỘ: CÓ EMBEDDING & TRUNCATION <<<<<<<<")
mf_emb = scf.UHF(mol).density_fit().run()
mppp_emb = UB2PLYPDFUOBMP2(mf_emb)
mppp_emb.alphaa = (0.53, 0.39)
mppp_emb.thresh = 1e-06

mppp_emb.use_embed = True   # Bật Embedding
mppp_emb.active_atoms = [0, 1]
mppp_emb.mu = 1e6
mppp_emb.use_cl = True      # Bật CL Truncation (chỉ có ý nghĩa khi use_embed=True)
mppp_emb.n_shells = 1

start2 = time.time()
mppp_emb.run()
print('=> Thời gian chạy (Embed + CL): ', time.time() - start2)