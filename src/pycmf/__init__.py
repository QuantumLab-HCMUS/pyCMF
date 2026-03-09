# =========================================================================
# File: src/pycmf/__init__.py
# Chức năng: Mặt tiền (Facade) Tổng của toàn bộ thư viện pyCMF
# =========================================================================

from pyscf import scf

# --- IMPORT TỪ HỆ KÍN (Thư mục obmp2) ---
from .obmp2 import (
    OBMP2 as _OBMP2_class,
    OBMP2_faster as _OBMP2_faster_class,
    OBMP2_active as _OBMP2_active_class,
    OBMP2_mod as _OBMP2_mod_class,
    KOBMP2 as _KOBMP2_class,
    DFOBMP2_faster_ram as _DFOBMP2_faster_ram_class,
    DFOBMP2_slower as _DFOBMP2_slower_class,
    DFTOBMP2 as _DFTOBMP2_class
)

# --- IMPORT TỪ HỆ MỞ (Thư mục uobmp2) ---
from .uobmp2 import (
    UOBMP2 as _UOBMP2_class,
    UOBMP2_faster as _UOBMP2_faster_class,
    UOBMP2_SCS as _UOBMP2_SCS_class,
    UOBMP2_MOM as _UOBMP2_MOM_class,
    UOBMP2_mom_conv as _UOBMP2_mom_conv_class,
    UOBMP2_dfold as _UOBMP2_dfold_class,
    UOBMP2_active as _UOBMP2_active_class,
    UOBMP2_active_scf as _UOBMP2_active_scf_class,
    DFUOBMP2_ram_reduced as _DFUOBMP2_ram_reduced_class,
    DFUOBMP2_ram_reduced_new as _DFUOBMP2_ram_reduced_new_class,
    DFUOBMP2_mom_conv as _DFUOBMP2_mom_conv_class,
    DFUOBMP2_faster_ram as _DFUOBMP2_faster_ram_class,
    DFTUOBMP2 as _DFTUOBMP2_class
)

# =========================================================================
# CÁC HÀM BỌC (WRAPPER FUNCTIONS) CHO HỆ KÍN (RHF)
# =========================================================================

def OBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _OBMP2_class(mf, frozen, mo_coeff, mo_occ)

def OBMP2_faster(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _OBMP2_faster_class(mf, frozen, mo_coeff, mo_occ)

def OBMP2_active(mf, nact, nocc_act, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _OBMP2_active_class(mf, nact, nocc_act, frozen, mo_coeff, mo_occ)

def OBMP2_mod(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _OBMP2_mod_class(mf, frozen, mo_coeff, mo_occ)

def KOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _KOBMP2_class(mf, frozen, mo_coeff, mo_occ)

def DFOBMP2_faster_ram(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _DFOBMP2_faster_ram_class(mf, frozen, mo_coeff, mo_occ)

def DFOBMP2_slower(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _DFOBMP2_slower_class(mf, frozen, mo_coeff, mo_occ)

def DFTOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _DFTOBMP2_class(mf, frozen, mo_coeff, mo_occ)


# =========================================================================
# CÁC HÀM BỌC (WRAPPER FUNCTIONS) CHO HỆ MỞ (UHF)
# =========================================================================

def UOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_class(mf, frozen, mo_coeff, mo_occ)

def UOBMP2_faster(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_faster_class(mf, frozen, mo_coeff, mo_occ)

def UOBMP2_SCS(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_SCS_class(mf, frozen, mo_coeff, mo_occ)

def UOBMP2_MOM(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_MOM_class(mf, frozen, mo_coeff, mo_occ)

def UOBMP2_mom_conv(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_mom_conv_class(mf, frozen, mo_coeff, mo_occ)

def UOBMP2_dfold(mf, nact, nocc_act, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_dfold_class(mf, nact, nocc_act, frozen, mo_coeff, mo_occ)

def UOBMP2_active(mf, nact, nocc_act, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_active_class(mf, nact, nocc_act, frozen, mo_coeff, mo_occ)

def UOBMP2_active_scf(mf, nact, nocc_act, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_active_scf_class(mf, nact, nocc_act, frozen, mo_coeff, mo_occ)

def DFUOBMP2_ram_reduced(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _DFUOBMP2_ram_reduced_class(mf, frozen, mo_coeff, mo_occ)

def DFUOBMP2_ram_reduced_new(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _DFUOBMP2_ram_reduced_new_class(mf, frozen, mo_coeff, mo_occ)

def DFUOBMP2_mom_conv(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _DFUOBMP2_mom_conv_class(mf, frozen, mo_coeff, mo_occ)

def DFUOBMP2_faster_ram(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _DFUOBMP2_faster_ram_class(mf, frozen, mo_coeff, mo_occ)

def DFTUOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _DFTUOBMP2_class(mf, frozen, mo_coeff, mo_occ)
