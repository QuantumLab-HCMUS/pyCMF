# =========================================================================
# File: src/pycmf/__init__.py
# Chức năng: Mặt tiền (Facade) Tổng của toàn bộ thư viện pyCMF
# =========================================================================

from pyscf import scf

# --- IMPORT TỪ OBMP (Incore & DF) ---
from .OBMP import (
    OBMP2 as _OBMP2_class,
    OBMP2_slow as _OBMP2_slow_class,
    OBMP2_einsum as _OBMP2_einsum_class,
    OBMP2_cas as _OBMP2_cas_class,
    DFOBMP2 as _DFOBMP2_class,
    DFOBMP2_slow as _DFOBMP2_slow_class,
    UOBMP2 as _UOBMP2_class,
    UOBMP2_slow as _UOBMP2_slow_class,
    UOBMP2_SCS as _UOBMP2_SCS_class,
    UOBMP2_mom as _UOBMP2_mom_class,
    UOBMP2_mom_diis as _UOBMP2_mom_diis_class,
    UOBMP2_cas as _UOBMP2_cas_class,
    UOBMP2_cas_scf as _UOBMP2_cas_scf_class,
    DFUOBMP2 as _DFUOBMP2_class,
    DFUOBMP2_einsum as _DFUOBMP2_einsum_class,
    DFUOBMP2_mom as _DFUOBMP2_mom_class,
    DFUOBMP2_mom_diis as _DFUOBMP2_mom_diis_class
)

# --- IMPORT TỪ OBDF (Quantum Downfolding) ---
from .OBDF import (
    OBMP2_downfold as _OBMP2_downfold_class,
    UOBMP2_downfold as _UOBMP2_downfold_class
)

# --- IMPORT TỪ DOUBLE HYBRID DFT VÀ K-POINTS ---
from .OBDH import (
    DFTOBMP2 as _DFTOBMP2_class,
    DFTUOBMP2 as _DFTUOBMP2_class,
    DFTUOBMP2_CL as _DFTUOBMP2_CL_class
)
from .OBMP import KOBMP2 as _KOBMP2_class


# =========================================================================
# CÁC HÀM BỌC (WRAPPER FUNCTIONS) CHO HỆ KÍN (RHF)
# =========================================================================
def OBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _OBMP2_class(mf, frozen, mo_coeff, mo_occ)

def OBMP2_slow(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _OBMP2_slow_class(mf, frozen, mo_coeff, mo_occ)

def OBMP2_einsum(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _OBMP2_einsum_class(mf, frozen, mo_coeff, mo_occ)

def OBMP2_cas(mf, nact, nocc_act, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _OBMP2_cas_class(mf, nact, nocc_act, frozen, mo_coeff, mo_occ)

def OBMP2_downfold(mf, nact, nocc_act, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _OBMP2_downfold_class(mf, nact, nocc_act, frozen, mo_coeff, mo_occ)

def DFOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _DFOBMP2_class(mf, frozen, mo_coeff, mo_occ)

def DFOBMP2_slow(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _DFOBMP2_slow_class(mf, frozen, mo_coeff, mo_occ)


# =========================================================================
# CÁC HÀM BỌC (WRAPPER FUNCTIONS) CHO HỆ MỞ (UHF)
# =========================================================================
def UOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_class(mf, frozen, mo_coeff, mo_occ)

def UOBMP2_slow(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_slow_class(mf, frozen, mo_coeff, mo_occ)

def UOBMP2_SCS(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_SCS_class(mf, frozen, mo_coeff, mo_occ)

def UOBMP2_mom(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_mom_class(mf, frozen, mo_coeff, mo_occ)

def UOBMP2_mom_diis(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_mom_diis_class(mf, frozen, mo_coeff, mo_occ)

def UOBMP2_cas(mf, nact, nocc_act, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_cas_class(mf, nact, nocc_act, frozen, mo_coeff, mo_occ)

def UOBMP2_cas_scf(mf, nact, nocc_act, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_cas_scf_class(mf, nact, nocc_act, frozen, mo_coeff, mo_occ)

def UOBMP2_downfold(mf, nact, nocc_act, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _UOBMP2_downfold_class(mf, nact, nocc_act, frozen, mo_coeff, mo_occ)

def DFUOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _DFUOBMP2_class(mf, frozen, mo_coeff, mo_occ)

def DFUOBMP2_einsum(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _DFUOBMP2_einsum_class(mf, frozen, mo_coeff, mo_occ)

def DFUOBMP2_mom(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _DFUOBMP2_mom_class(mf, frozen, mo_coeff, mo_occ)

def DFUOBMP2_mom_diis(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _DFUOBMP2_mom_diis_class(mf, frozen, mo_coeff, mo_occ)

# =========================================================================
# CÁC HÀM BỌC (WRAPPER FUNCTIONS) CHO DFT & PERIODIC
# =========================================================================
def DFTOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return _DFTOBMP2_class(mf, frozen, mo_coeff, mo_occ)

def DFTUOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _DFTUOBMP2_class(mf, frozen, mo_coeff, mo_occ)

def DFTUOBMP2_CL(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return _DFTUOBMP2_CL_class(mf, frozen, mo_coeff, mo_occ)

def KOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    return _KOBMP2_class(mf, frozen, mo_coeff, mo_occ)