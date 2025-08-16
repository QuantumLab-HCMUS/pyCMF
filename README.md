# pyCMF

pyCMF is a Python package for performing Orbital-Optimized Møller-Plesset Perturbation Theory (OB-MP2) calculations. It is built on top of the [PySCF](https://github.com/pyscf/pyscf) library.

## Installation

1.  Clone the repository:

```bash
git clone https://github.com/Quantum-Lab-HCMUS/pyCMF.git
cd pyCMF
```

2.  Install the required dependencies:
Create conda enviroment:
```bash
conda create -n pycmf python=3.11 anaconda
conda activate pycmf
```
Now we install the requirements:
```bash
pip install -r requirements.txt
```

## Usage
Run `main.py` to see a simple example of how to use pyCMF to perform a UOBMP2

```python
# Found in main.py
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
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
