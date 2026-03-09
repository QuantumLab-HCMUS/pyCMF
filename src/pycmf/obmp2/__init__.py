# File: src/pycmf/obmp2/__init__.py

# 1. Import class cơ bản
from .obmp2 import OBMP2, _ChemistsERIs

# 2. Import và đặt bí danh (alias) cho các class trùng tên OBMP2
from .obmp2_faster import OBMP2 as OBMP2_faster
from .obmp2_active import OBMP2 as OBMP2_active
from .obmp2mod import OBMP2 as OBMP2_mod
from .kobmp2 import OBMP2 as KOBMP2

# 3. Import và đặt bí danh cho các class Density Fitting (DF)
from .dfobmp2_faster_ram import DFOBMP2 as DFOBMP2_faster_ram
from .dfobmp2_slower import DFOBMP2 as DFOBMP2_slower

# 4. Import class DFT
from .dftobmp2 import B2PLYPDFOBMP2 as DFTOBMP2

# 5. Khai báo danh sách các class được phép gọi từ bên ngoài
__all__ = [
    'OBMP2',
    '_ChemistsERIs',
    'OBMP2_faster',
    'OBMP2_active',
    'OBMP2_mod',
    'KOBMP2',
    'DFOBMP2_faster_ram',
    'DFOBMP2_slower',
    'DFTOBMP2'
]
