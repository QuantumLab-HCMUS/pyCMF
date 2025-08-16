from pycmf.uobmp2 import UOBMP2
from pyscf import scf
from pyscf import gto
mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]

mol.basis = 'cc-pvdz'
mol.build()
mf = scf.RHF(mol).run()
mp = UOBMP2(mf)
mp.verbose = 5