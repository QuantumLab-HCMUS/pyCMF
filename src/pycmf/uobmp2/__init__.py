# =========================================================================
# File: src/pycmf/uobmp2/__init__.py
# Chức năng: Quản lý và public các thuật toán thuộc hệ mở (Unrestricted)
# =========================================================================

# 1. Import các class cơ bản
from .uobmp2 import UOBMP2, _ChemistsERIs

# 2. Import và đặt bí danh (alias) cho các biến thể UOBMP2
from .uobmp2_faster import UOBMP2 as UOBMP2_faster
from .uobmp2_scs import UOBMP2_SCS  # Class này đã có tên riêng
from .uobmp2_mom import UOBMP2 as UOBMP2_MOM
from .uobmp2_mom_conv import UOBMP2 as UOBMP2_mom_conv
from .uobmp2_dfold import UOBMP2 as UOBMP2_dfold
from .uobmp2_active import UOBMP2 as UOBMP2_active
from .uobmp2_active_scf import UOBMP2 as UOBMP2_active_scf

# 3. Import và đặt bí danh cho các class Density Fitting (DFUOBMP2)
from .dfuobmp2_ram_reduced import DFUOBMP2 as DFUOBMP2_ram_reduced
from .dfuobmp2_ram_reduced_new import DFUOBMP2 as DFUOBMP2_ram_reduced_new
from .dfuobmp2_mom_conv import DFUOBMP2 as DFUOBMP2_mom_conv
from .dfuobmp2_faster_ram import DFUOBMP2 as DFUOBMP2_faster_ram

# 4. Import class DFT (Double Hybrid)
from .dftuobmp2 import UB2PLYPDFUOBMP2 as DFTUOBMP2

# 5. Khai báo danh sách các class chính thức được phép gọi từ bên ngoài
__all__ = [
    'UOBMP2',
    '_ChemistsERIs',
    'UOBMP2_faster',
    'UOBMP2_SCS',
    'UOBMP2_MOM',
    'UOBMP2_mom_conv',
    'UOBMP2_dfold',
    'UOBMP2_active',
    'UOBMP2_active_scf',
    'DFUOBMP2_ram_reduced',
    'DFUOBMP2_ram_reduced_new',
    'DFUOBMP2_mom_conv',
    'DFUOBMP2_faster_ram',
    'DFTUOBMP2'
]
