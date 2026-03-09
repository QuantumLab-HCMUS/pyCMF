# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Moller-Plesset perturbation theory
'''

from pyscf import scf
from pyscf.mp import mp2
from pyscf.mp import dfmp2
from pyscf.mp import ump2
from pyscf.mp import gmp2
from pyscf.mp import dfgmp2
from pyscf.mp import dfump2

from pyscf.mp import obmp2
from pyscf.mp import obmp2_faster
from pyscf.mp import obmp2_active
from pyscf.mp import obmp2mod # Newly added


from pyscf.mp import uobmp2
from pyscf.mp import uobmp2_faster
from pyscf.mp import uobmp2_scs
from pyscf.mp import uobmp2_mom
from pyscf.mp import uobmp2_mom_conv # Newly added
from pyscf.mp import uobmp2_dfold # Newly added
from pyscf.mp import uobmp2_active # Newly added
from pyscf.mp import uobmp2_active_scf # Newly added

from pyscf.mp import kobmp2 # Newly added

from pyscf.mp import dfuobmp2_ram_reduced # Newly added
from pyscf.mp import dfuobmp2_ram_reduced_new # Newly added
from pyscf.mp import dfuobmp2_mom_conv # Newly added
from pyscf.mp import dfuobmp2_faster_ram # Newly added

from pyscf.mp import dfobmp2_slower # Newly added
from pyscf.mp import dfobmp2_faster_ram # Newly added

from pyscf.mp import dftuobmp2 # Newly added
from pyscf.mp import dftobmp2 # Newly added



def MP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if mf.istype('UHF'):
        return UMP2(mf, frozen, mo_coeff, mo_occ)
    elif mf.istype('GHF'):
        return GMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return RMP2(mf, frozen, mo_coeff, mo_occ)
MP2.__doc__ = mp2.MP2.__doc__

def RMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf import lib

    if mf.istype('UHF'):
        raise RuntimeError('RMP2 cannot be used with UHF method.')
    elif mf.istype('ROHF'):
        lib.logger.warn(mf, 'RMP2 method does not support ROHF method. ROHF object '
                        'is converted to UHF object and UMP2 method is called.')
        return UMP2(mf, frozen, mo_coeff, mo_occ)

    mf = mf.remove_soscf()
    if not mf.istype('RHF'):
        mf = mf.to_rhf()

    if getattr(mf, 'with_df', None):
        return dfmp2.DFMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return mp2.RMP2(mf, frozen, mo_coeff, mo_occ)
RMP2.__doc__ = mp2.RMP2.__doc__

def UMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = mf.remove_soscf()
    if not mf.istype('UHF'):
        mf = mf.to_uhf()

    if getattr(mf, 'with_df', None):
        return dfump2.DFUMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return ump2.UMP2(mf, frozen, mo_coeff, mo_occ)
UMP2.__doc__ = ump2.UMP2.__doc__

def GMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = mf.remove_soscf()
    if not mf.istype('GHF'):
        mf = mf.to_ghf()

    if getattr(mf, 'with_df', None):
        return dfgmp2.DFGMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return gmp2.GMP2(mf, frozen, mo_coeff, mo_occ)
GMP2.__doc__ = gmp2.GMP2.__doc__

def OBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return obmp2.OBMP2(mf, frozen, mo_coeff, mo_occ)
    #elif isinstance(mf, scf.ghf.GHF):
    #    return GMP2(mf, frozen, mo_coeff, mo_occ)
    #else:
    #    return RMP2(mf, frozen, mo_coeff, mo_occ)
OBMP2.__doc__ = obmp2.OBMP2.__doc__

def OBMP2_active(mf, nact, nocc_act, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return obmp2_active.OBMP2(mf, nact, nocc_act, frozen, mo_coeff, mo_occ)
OBMP2_active.__doc__ = obmp2_active.OBMP2.__doc__

def OBMP2_faster(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return obmp2_faster.OBMP2(mf, frozen, mo_coeff, mo_occ)
OBMP2_faster.__doc__ = obmp2_faster.OBMP2.__doc__


def UOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return uobmp2.UOBMP2(mf, frozen, mo_coeff, mo_occ)
UOBMP2.__doc__ = uobmp2.UOBMP2.__doc__

#def UOBMP2_faster(mf, frozen=0, mo_coeff=None, mo_occ=None):
#    __doc__ = uobmp2_faster.UOBMP2.__doc__
#    if isinstance(mf, scf.uhf.UHF):
#        return uobmp2_faster.UOBMP2(mf, frozen, mo_coeff, mo_occ)

def UOBMP2_faster(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return uobmp2_faster.UOBMP2(mf, frozen, mo_coeff, mo_occ)
UOBMP2_faster.__doc__ = uobmp2_faster.UOBMP2.__doc__

def UOBMP2_SCS(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return uobmp2_scs.UOBMP2_SCS(mf, frozen, mo_coeff, mo_occ)
UOBMP2_SCS.__doc__ = uobmp2_scs.UOBMP2_SCS.__doc__


def UOBMP2_MOM(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return uobmp2_mom.UOBMP2(mf, frozen, mo_coeff, mo_occ)
UOBMP2_MOM.__doc__ = uobmp2_mom.UOBMP2.__doc__

############################## Newly added ##############################

def OBMP2_mod(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return obmp2mod.OBMP2(mf, frozen, mo_coeff, mo_occ)
OBMP2_mod.__doc__ = obmp2mod.OBMP2.__doc__

def UOBMP2_mom_conv(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return uobmp2_mom_conv.UOBMP2(mf, frozen, mo_coeff, mo_occ)
UOBMP2_mom_conv.__doc__ = uobmp2_mom_conv.UOBMP2.__doc__

def UOBMP2_dfold(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return uobmp2_dfold.UOBMP2(mf, frozen, mo_coeff, mo_occ)
UOBMP2_dfold.__doc__ = uobmp2_dfold.UOBMP2.__doc__

def UOBMP2_active(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return uobmp2_active.UOBMP2(mf, frozen, mo_coeff, mo_occ)
UOBMP2_active.__doc__ = uobmp2_active.UOBMP2.__doc__

def UOBMP2_active_scf(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return uobmp2_active_scf.UOBMP2(mf, frozen, mo_coeff, mo_occ)
UOBMP2_active_scf.__doc__ = uobmp2_active_scf.UOBMP2.__doc__

def KOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return kobmp2.OBMP2(mf, frozen, mo_coeff, mo_occ)
KOBMP2.__doc__ = kobmp2.OBMP2.__doc__

def UOBMP2_mom_conv(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return uobmp2_mom_conv.UOBMP2(mf, frozen, mo_coeff, mo_occ)
UOBMP2_mom_conv.__doc__ = uobmp2_mom_conv.UOBMP2.__doc__

def DFUOBMP2_ram_reduced(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return dfuobmp2_ram_reduced.DFUOBMP2(mf, frozen, mo_coeff, mo_occ)
DFUOBMP2_ram_reduced.__doc__ = dfuobmp2_ram_reduced.DFUOBMP2.__doc__

def DFUOBMP2_ram_reduced_new(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return dfuobmp2_ram_reduced_new.DFUOBMP2(mf, frozen, mo_coeff, mo_occ)
DFUOBMP2_ram_reduced_new.__doc__ = dfuobmp2_ram_reduced_new.DFUOBMP2.__doc__

def DFUOBMP2_mom_conv(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return dfuobmp2_mom_conv.DFUOBMP2(mf, frozen, mo_coeff, mo_occ)
DFUOBMP2_mom_conv.__doc__ = dfuobmp2_mom_conv.DFUOBMP2.__doc__

def DFUOBMP2_faster_ram(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return dfuobmp2_faster_ram.DFUOBMP2(mf, frozen, mo_coeff, mo_occ)
DFUOBMP2_faster_ram.__doc__ = dfuobmp2_faster_ram.DFUOBMP2.__doc__

def DFOBMP2_slower(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return dfobmp2_slower.DFOBMP2(mf, frozen, mo_coeff, mo_occ)
DFOBMP2_slower.__doc__ = dfobmp2_slower.DFOBMP2.__doc__

def DFOBMP2_faster_ram(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return dfobmp2_faster_ram.DFOBMP2(mf, frozen, mo_coeff, mo_occ)
DFOBMP2_faster_ram.__doc__ = dfobmp2_faster_ram.DFOBMP2.__doc__

def DFTOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.rhf.RHF):
        return dftobmp2.B2PLYPDFOBMP2(mf, frozen, mo_coeff, mo_occ)
DFTOBMP2.__doc__ = dftobmp2.B2PLYPDFOBMP2.__doc__

def DFTUOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return dftuobmp2.UB2PLYPDFUOBMP2(mf, frozen, mo_coeff, mo_occ)
DFTUOBMP2.__doc__ = dftuobmp2.UB2PLYPDFUOBMP2.__doc__

