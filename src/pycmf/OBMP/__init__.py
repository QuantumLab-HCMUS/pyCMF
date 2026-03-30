# =========================================================================
# File: src/pycmf/OBMP/__init__.py
# Chức năng: Quản lý các thuật toán Orbital-Optimized MP2 (Incore & Density Fitting)
# =========================================================================

# 1. Nhánh Restricted (Hệ đóng)
from .obmp2 import OBMP2, _ChemistsERIs
from .obmp2_slow import OBMP2 as OBMP2_slow
from .obmp2_einsum import OBMP2 as OBMP2_einsum
from .obmp2_cas import OBMP2 as OBMP2_cas
from .dfobmp2 import DFOBMP2
from .dfobmp2_slow import DFOBMP2 as DFOBMP2_slow

# 2. Nhánh Unrestricted (Hệ mở)
from .uobmp2 import UOBMP2
from .uobmp2_slow import UOBMP2 as UOBMP2_slow
from .uobmp2_scs import UOBMP2_SCS
from .uobmp2_mom import UOBMP2 as UOBMP2_mom
from .uobmp2_mom_diis import UOBMP2 as UOBMP2_mom_diis
from .uhf_mom_diis import UOBMP2 as UHF_mom_diis
from .uobmp2_cas import UOBMP2 as UOBMP2_cas
from .uobmp2_cas_scf import UOBMP2 as UOBMP2_cas_scf
from .dfuobmp2 import DFUOBMP2
from .dfuobmp2_einsum import DFUOBMP2 as DFUOBMP2_einsum
from .dfuobmp2_mom import DFUOBMP2 as DFUOBMP2_mom
from .dfuobmp2_mom_diis import DFUOBMP2 as DFUOBMP2_mom_diis

from .kobmp2 import OBMP2 as KOBMP2
from .kuobmp2 import OBMP2 as KUOBMP2

__all__ = [
    'KOBMP2',
    'OBMP2', '_ChemistsERIs', 'OBMP2_slow', 'OBMP2_einsum', 'OBMP2_cas',
    'DFOBMP2', 'DFOBMP2_slow',
    'UOBMP2', 'UOBMP2_slow', 'UOBMP2_SCS', 'UOBMP2_mom', 'UOBMP2_mom_diis', 'UHF_mom_diis',
    'UOBMP2_cas', 'UOBMP2_cas_scf',
    'DFUOBMP2', 'DFUOBMP2_einsum', 'DFUOBMP2_mom', 'DFUOBMP2_mom_diis', 'KUOBMP2'
]