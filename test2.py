import os

BASE_DIR = "src/pycmf"

# =========================================================
# Nội dung cho thư mục OBMP (Chứa các file MP2 tiêu chuẩn)
# =========================================================
init_obmp = """# =========================================================================
# File: src/pycmf/OBMP/__init__.py
# Chức năng: Quản lý các thuật toán Orbital-Optimized MP2 (Incore/Cơ bản)
# =========================================================================

# 1. Nhánh Restricted (Hệ đóng)
from .obmp2 import OBMP2, _ChemistsERIs
from .obmp2_slow import OBMP2 as OBMP2_slow
from .obmp2_einsum import OBMP2 as OBMP2_einsum
from .obmp2_cas import OBMP2 as OBMP2_cas

# 2. Nhánh Unrestricted (Hệ mở)
from .uobmp2 import UOBMP2
from .uobmp2_slow import UOBMP2 as UOBMP2_slow
from .uobmp2_scs import UOBMP2_SCS
from .uobmp2_mom import UOBMP2 as UOBMP2_mom
from .uobmp2_mom_diis import UOBMP2 as UOBMP2_mom_diis
from .uobmp2_cas import UOBMP2 as UOBMP2_cas
from .uobmp2_cas_scf import UOBMP2 as UOBMP2_cas_scf
from .uobmp2_downfold import UOBMP2 as UOBMP2_downfold

__all__ = [
    'OBMP2', '_ChemistsERIs', 'OBMP2_slow', 'OBMP2_einsum', 'OBMP2_cas',
    'UOBMP2', 'UOBMP2_slow', 'UOBMP2_SCS', 'UOBMP2_mom', 'UOBMP2_mom_diis',
    'UOBMP2_cas', 'UOBMP2_cas_scf', 'UOBMP2_downfold'
]
"""

# =========================================================
# Nội dung cho thư mục OBDF (Chứa các file Density Fitting)
# =========================================================
init_obdf = """# =========================================================================
# File: src/pycmf/OBDF/__init__.py
# Chức năng: Quản lý các thuật toán OBMP2 dùng Density Fitting giảm RAM
# =========================================================================

# 1. Nhánh Restricted
from .dfobmp2 import DFOBMP2
from .dfobmp2_slow import DFOBMP2 as DFOBMP2_slow

# 2. Nhánh Unrestricted
from .dfuobmp2 import DFUOBMP2
from .dfuobmp2_einsum import DFUOBMP2 as DFUOBMP2_einsum
from .dfuobmp2_mom import DFUOBMP2 as DFUOBMP2_mom
from .dfuobmp2_mom_diis import DFUOBMP2 as DFUOBMP2_mom_diis
from .dfuobmp2_old import DFUOBMP2 as DFUOBMP2_old

__all__ = [
    'DFOBMP2', 'DFOBMP2_slow',
    'DFUOBMP2', 'DFUOBMP2_einsum', 'DFUOBMP2_mom', 'DFUOBMP2_mom_diis', 'DFUOBMP2_old'
]
"""

# =========================================================
# Nội dung cho thư mục OBDH (Chứa các file Double Hybrid DFT)
# =========================================================
init_obdh = """# =========================================================================
# File: src/pycmf/OBDH/__init__.py
# Chức năng: Quản lý các thuật toán Double Hybrid DFT (B2PLYP)
# =========================================================================

from .dft_obmp2 import B2PLYPDFOBMP2 as DFTOBMP2
from .dft_uobmp2 import UB2PLYPDFUOBMP2 as DFTUOBMP2

__all__ = [
    'DFTOBMP2', 'DFTUOBMP2'
]
"""

# =========================================================
# Nội dung cho thư mục KOBMP (Chứa các file hệ tuần hoàn)
# =========================================================
init_kobmp = """# =========================================================================
# File: src/pycmf/KOBMP/__init__.py
# Chức năng: Quản lý các thuật toán OBMP2 cho hệ tuần hoàn (k-points)
# =========================================================================

from .kobmp2 import OBMP2 as KOBMP2

__all__ = [
    'KOBMP2'
]
"""

# Thực thi ghi file
files_to_write = {
    os.path.join(BASE_DIR, "OBMP", "__init__.py"): init_obmp,
    os.path.join(BASE_DIR, "OBDF", "__init__.py"): init_obdf,
    os.path.join(BASE_DIR, "OBDH", "__init__.py"): init_obdh,
    os.path.join(BASE_DIR, "KOBMP", "__init__.py"): init_kobmp,
}

print("Bắt đầu khởi tạo nội dung cho các file __init__.py...")
for filepath, content in files_to_write.items():
    if os.path.exists(os.path.dirname(filepath)):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Đã cập nhật thành công: {filepath}")
    else:
        print(f"❌ Không tìm thấy thư mục cho: {filepath}. Bạn đã chạy script chuyển file chưa?")

print("\n🎉 Hoàn tất! Bạn đã có một bộ Public API cực kỳ chuyên nghiệp!")