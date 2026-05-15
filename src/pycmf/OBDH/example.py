import time
from pyscf import scf, gto
from pycmf.OBDH import DFTUOBMP2

mol = gto.Mole()
mol.atom = '''
O    0.000000   0.000000   0.000000
H    0.000000   0.000000   0.962000
H    1.790000   1.026000   0.000000
'''
mol.spin = 0
mol.spin = 0
mol.verbose = 0
mol.basis = 'sto-3g'
mol.build()

# print("\n>>>>>>>> CHẠY CHẾ ĐỘ: KHÔNG CÓ EMBEDDING (OBDH CHUẨN) <<<<<<<<")
# mf_std = scf.UHF(mol).density_fit().run()
# mppp_std = DFTUOBMP2(mf_std)
# mppp_std.alphaa = (0.53, 0.39)
# mppp_std.thresh = 1e-06
# mppp_std.use_embed = False  # Tắt Embedding
# # mppp_std.use_cl = True    # Dù bật cũng sẽ bị bỏ qua

# start = time.time()
# mppp_std.run()
# print('=> Thời gian chạy (No Embed): ', time.time() - start)

print("\n\n>>>>>>>> CHẠY CHẾ ĐỘ: CÓ EMBEDDING & TRUNCATION <<<<<<<<")
mf_emb = scf.UHF(mol).density_fit().run()
mppp_emb = DFTUOBMP2(mf_emb)
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