#!/usr/bin/env python
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


"""
Periodic OB-MP2
"""

import time, logging, tracemalloc
from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.lib import kpts_helper
from pyscf import __config__
from pyscf.pbc.mp import kmp2

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)
LARGE_DENOM = getattr(__config__, 'LARGE_DENOM', 1e14)


def kernel(mp, mo_energy, mo_coeff, mo_occ, with_t2=WITH_T2, verbose=logger.NOTE):

    kpts = mp.kpts
    nkpts = numpy.shape(mo_energy)[0]

    nocc = mp.nocc
    nact = mp.nact
    nocc_act = mp.nocc_act

    # Core band
    mp.ncore = nocc - nocc_act
    ncore = mp.ncore

    # Virtual band (Vẫn gán vào mp để đảm bảo tính nhất quán của object)
    mp.nvir = mp.nmo - nact
    mp.nvir_act = nact - nocc_act

    print()
    print('**********************************')
    print('************** KROBDF *************')

    print('nmo, ncore, nact, nvir, nocc', mp.nmo, ncore, nact, mp.nvir, mp.nocc)
    sort_idx = numpy.argsort(mo_energy)

    # ============================================================
    # H1 (k)
    # ============================================================
    h1ao = mp._scf.get_hcore()
    C = mp.mo_coeff

    # h1 in MO basis
    h1mo = [reduce(numpy.dot, (mo.conj().T, h1ao[k], mo)) for k, mo in enumerate(C)]
    veff_core = make_veff_core(mp)

    # Build effective h1 in active space trực tiếp vào Numpy Array
    h1mo_act_eff = numpy.zeros((nkpts, nact, nact), dtype=complex)
    for k in range(nkpts):
        h1mo_act_k = h1mo[k][ncore : ncore + nact, ncore : ncore + nact]
        h1mo_act_eff[k] = h1mo_act_k + veff_core[k]

    make_veff(mp)

    # ============================================================
    # Fock
    # ============================================================
    fock_hf_act = []
    for k in range(nkpts):
        fock_hf_k = mp.fock_hf[k][ncore : ncore + nact, ncore : ncore + nact]
        fock_hf_act.append(fock_hf_k.copy())

    if mp.second_order:
        mp.ampf = 1.0

    # ============================================================
    # BCH
    # ============================================================
    # 1. Active BCH
    c0, c1 = active_BCH(mp, mp.mo_energy, mp.mo_coeff, fock_hf_act)  # Chạy được

    # 2. Internal BCH (Loại bỏ các hàm _loop dư thừa)
    c1_inter = inter_BCH(mp, mp.mo_energy, mp.mo_coeff, mp.fock_hf)

    # 3. External BCH
    c1_exter = c1_inter - c1

    # 4. Áp dụng External BCH vào Hamiltonian hiệu dụng
    for k in range(nkpts):
        h1mo_act_eff[k] += c1_exter[k] + c1_exter[k].T.conj()

    # ============================================================
    # H2
    # ============================================================
    kconserv = kpts_helper.get_kconserv(mp._scf.cell, kpts)
    fao2mo = mp._scf.with_df.ao2mo

    ic = ncore
    ia = ncore + nact

    h2mo_act = numpy.zeros((nkpts, nkpts, nkpts, nkpts, nact, nact, nact, nact), dtype=complex)

    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):
                ks = kconserv[kp, kq, kr]

                Cp = mo_coeff[kp][:, ic:ia]
                Cq = mo_coeff[kq][:, ic:ia]
                Cr = mo_coeff[kr][:, ic:ia]
                Cs = mo_coeff[ks][:, ic:ia]

                h2mo_act[kp, kq, kr, ks] = (
                    fao2mo((Cp, Cq, Cr, Cs), (kpts[kp], kpts[kq], kpts[kr], kpts[ks]), compact=False).reshape(
                        nact, nact, nact, nact
                    )
                    / nkpts
                )

    return (h1mo_act_eff, h2mo_act)


#################################################################################################################


def make_veff(mp):  # , mo_coeff, mo_energy

    mo_coeff = mp.mo_coeff
    mo_energy = mp.mo_energy

    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = numpy.shape(mo_energy)[0]

    kpts = mp.kpts
    dm = mp._scf.make_rdm1()
    veff_ao = mp._scf.get_veff(mp._scf.cell, dm)

    veff = numpy.zeros((nkpts, nmo, nmo), dtype=complex)

    for kp in range(nkpts):
        veff[kp] = numpy.matmul(mo_coeff[kp].T.conj(), numpy.matmul(veff_ao[kp], mo_coeff[kp]))

    c0_hf = 0
    for kp in range(nkpts):
        for i in range(nocc):
            c0_hf -= veff[kp][i, i].real
    c0_hf /= nkpts

    return veff_ao, veff, c0_hf


def make_veff_core(mp):
    """
    Frozen-core effective potential in active MO space (PBC, k-resolved)

    Returns
    -------
    veff_core : list of ndarray
        veff_core[k] shape = (nact, nact)
    """
    mf = mp._scf
    cell = mf.cell
    kpts = mf.kpts
    nkpts = len(kpts)

    nocc = mp.nocc
    nocc_act = mp.nocc_act
    nact = mp.nact
    nocc_inact = nocc - nocc_act

    # --- 1. build core density matrix (AO, per k) ---
    dm_core = []
    for k in range(nkpts):
        mo = mf.mo_coeff[k][:, :nocc_inact]
        dm = mo @ mo.conj().T
        dm_core.append(dm)

    # --- 2. AO-level J/K from core density ---
    vj, vk = mf.get_jk(cell, dm_core, kpts=kpts)

    # --- 3. AO → active MO projection ---
    veff_core = []
    for k in range(nkpts):
        v_core_ao = vj[k] - 0.5 * vk[k]

        mo = mf.mo_coeff[k]
        cg = mo[:, nocc_inact : nocc_inact + nact]

        v_core_mo = cg.conj().T @ v_core_ao @ cg
        veff_core.append(v_core_mo)

    return veff_core


def ene_denom(mp, mo_energy, ki, ka, kj, kb):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = numpy.shape(mo_energy)[0]

    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind='split')
    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    eia = LARGE_DENOM * numpy.ones((nocc, nvir), dtype=mo_energy[0].dtype)
    n0_ovp_ia = numpy.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
    eia[n0_ovp_ia] = (mo_e_o[ki][:, None] - mo_e_v[ka])[n0_ovp_ia]

    ejb = LARGE_DENOM * numpy.ones((nocc, nvir), dtype=mo_energy[0].dtype)
    n0_ovp_jb = numpy.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
    ejb[n0_ovp_jb] = (mo_e_o[kj][:, None] - mo_e_v[kb])[n0_ovp_jb]

    ejh = LARGE_DENOM * numpy.ones((nocc, nocc), dtype=mo_energy[0].dtype)
    n0_ovp_jh = numpy.ix_(nonzero_opadding[kj], nonzero_opadding[kb])
    ejh[n0_ovp_jh] = (mo_e_o[kj][:, None] - mo_e_o[kb])[n0_ovp_jh]

    eah = LARGE_DENOM * numpy.ones((nvir, nocc), dtype=mo_energy[0].dtype)
    n0_ovp_ah = numpy.ix_(nonzero_opadding[ka], nonzero_opadding[kb])
    eah[n0_ovp_ah] = (mo_e_v[ka][:, None] - mo_e_o[kb])[n0_ovp_ah]

    eij = LARGE_DENOM * numpy.ones((nocc, nocc), dtype=mo_energy[0].dtype)
    n0_ovp_ij = numpy.ix_(nonzero_opadding[ki], nonzero_opadding[kj])
    eij[n0_ovp_ij] = (mo_e_o[ki][:, None] - mo_e_o[kj])[n0_ovp_ij]

    elb = LARGE_DENOM * numpy.ones((nvir, nvir), dtype=mo_energy[0].dtype)
    n0_ovp_lb = numpy.ix_(nonzero_vpadding[kj], nonzero_vpadding[kb])
    elb[n0_ovp_lb] = (mo_e_v[kj][:, None] - mo_e_v[kb])[n0_ovp_lb]

    e_iajb = lib.direct_sum('ia,jb -> iajb', eia, ejb)
    e_ialb = lib.direct_sum('ia,lb -> ialb', eia, elb)
    e_iajh = lib.direct_sum('ia,jh -> iajh', eia, ejh)
    e_ahij = lib.direct_sum('ah,ij -> ahij', eah, eij)

    return e_iajb, e_ialb, e_iajh, e_ahij


def active_BCH(mp, mo_energy, mo_coeff, fock_hf_act):
    import numpy
    from pyscf.pbc.lib import kpts_helper

    nkpts = numpy.shape(mo_energy)[0]
    kpts = mp.kpts

    ncore = mp.ncore
    nact = mp.nact
    nocc_act = mp.nocc_act
    nvir_act = mp.nvir_act

    occ_act = slice(ncore, ncore + nocc_act)
    vir_act = slice(0, nvir_act)

    kconserv = kpts_helper.get_kconserv(mp._scf.cell, kpts)
    fao2mo = mp._scf.with_df.ao2mo

    fock_hf_act = numpy.array(fock_hf_act)

    # Khởi tạo mảng (5D logic)
    tmp1_act = numpy.zeros((nkpts, nocc_act, nvir_act, nocc_act, nvir_act), dtype=complex)
    tmp1_bar_act = numpy.zeros((nkpts, nocc_act, nvir_act, nocc_act, nvir_act), dtype=complex)
    tmp_bar = numpy.zeros((nkpts, nkpts, nocc_act, nvir_act, nocc_act, nvir_act), dtype=complex)

    h2mo_ovgg = numpy.zeros((nkpts, nocc_act, nvir_act, nact, nact), dtype=complex)
    h2mo_ovov = numpy.zeros((nkpts, nocc_act, nvir_act, nocc_act, nvir_act), dtype=complex)
    h2mo_ovog = numpy.zeros((nkpts, nocc_act, nvir_act, nocc_act, nact), dtype=complex)
    h2mo_ovgv = numpy.zeros((nkpts, nocc_act, nvir_act, nact, nvir_act), dtype=complex)

    c1 = numpy.zeros((nkpts, nact, nact), dtype=complex)
    c2 = numpy.zeros((nkpts, nact, nact), dtype=complex)

    y1 = numpy.zeros((nkpts, nvir_act, nocc_act), dtype=complex)
    y2 = numpy.zeros((nkpts, nocc_act, nvir_act, nocc_act, nvir_act), dtype=complex)
    y3 = numpy.zeros((nkpts, nocc_act, nvir_act, nocc_act, nvir_act), dtype=complex)
    y4 = numpy.zeros((nkpts, nocc_act, nocc_act), dtype=complex)
    y5 = numpy.zeros((nkpts, nvir_act, nvir_act), dtype=complex)

    c0 = 0.0
    print('mo energy', mo_energy)

    for ki in range(nkpts):
        for kj in range(nkpts):
            # --- TÍNH TÍCH PHÂN ---
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]

                o_i = mo_coeff[ki][:, ncore : ncore + nocc_act]
                o_a = mo_coeff[ka][:, ncore + nocc_act : ncore + nocc_act + nvir_act]
                o_j = mo_coeff[kj][:, ncore : ncore + nact]
                o_b = mo_coeff[kb][:, ncore : ncore + nact]

                h2mo_ovgg[ka] = (
                    fao2mo(
                        (o_i, o_a, o_j, o_b), (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]), compact=False
                    ).reshape(nocc_act, nvir_act, nact, nact)
                    / nkpts
                )

                h2mo_ovov[ka] = h2mo_ovgg[ka][:, :, :nocc_act, nocc_act:]
                h2mo_ovgv[ka] = h2mo_ovgg[ka][:, :, :, nocc_act:]
                h2mo_ovog[ka] = h2mo_ovgg[ka][:, :, :nocc_act, :]

            # --- CỘNG DỒN BCH ---
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]

                # Hàm ene_denom có thể trả về nhiều giá trị, ta lấy giá trị denom đầu tiên
                e_iajb_full = ene_denom(mp, mo_energy, ki, ka, kj, kb)[0]

                # Cắt lấy vùng active denom
                o = slice(0, nocc_act)
                v = slice(0, nvir_act)
                e_iajb_act = e_iajb_full[o, v, o, v]

                w_iajb = h2mo_ovov[ka] - 0.5 * h2mo_ovov[kb].transpose(0, 3, 2, 1)

                tmp1_act[ka] = (h2mo_ovov[ka] / e_iajb_act).conj()
                tmp1_bar_act[ka] = (w_iajb / e_iajb_act).conj()

                mp.ampf = getattr(mp, 'ampf', 1.0)
                tmp1_act[ka] *= mp.ampf
                tmp1_bar_act[ka] *= mp.ampf

                # --- 1st order ---
                c1[kb, nocc_act:, :] -= 2 * numpy.einsum('iajb, iajp -> bp', tmp1_bar_act[ka], h2mo_ovog[ka])
                c1[kj, :, :nocc_act] += 2 * numpy.einsum('iajb, iapb -> pj', tmp1_bar_act[ka], h2mo_ovgv[ka])

                if ki == ka:
                    c1[kj, nocc_act:, :nocc_act] += 2 * numpy.einsum(
                        'iajb, ia -> bj', tmp1_bar_act[ka], fock_hf_act[ka, :nocc_act, nocc_act:]
                    )
                c0 -= 4 * numpy.einsum('iajb, iajb', tmp1_bar_act[ka], h2mo_ovov[ka]).real

                # --- 2nd order ---
                if hasattr(mp, 'second_order') and mp.second_order:
                    if ki == ka:
                        y1[kj] += 2 * numpy.einsum(
                            'ia, iajb -> bj', fock_hf_act[ka][:nocc_act, nocc_act:], tmp1_bar_act[ka]
                        )
                        tmp_bar[ki, kj] = tmp1_bar_act[ka]

                    y2[ka] = numpy.einsum(
                        'ca, iclb -> ialb', fock_hf_act[ka, nocc_act:, nocc_act:], tmp1_bar_act[ka].conj()
                    )
                    y3[ka] = numpy.einsum(
                        'ik, kalb -> ialb', fock_hf_act[ki, :nocc_act, :nocc_act], tmp1_bar_act[ka].conj()
                    )
                    y4[ki, :, :] += numpy.einsum('iajb, kajb -> ki', tmp1_act[ka], tmp1_bar_act[ka].conj())
                    y5[ka, :, :] += numpy.einsum('iajb, icjb -> ac', tmp1_act[ka], tmp1_bar_act[ka].conj())

                    c2[kj, :nocc_act, :nocc_act] += numpy.einsum('ialb, iajb -> lj', y2[ka], tmp1_act[ka])
                    c2[ki, :nocc_act, :nocc_act] += numpy.einsum('kajb, iajb -> ki', y2[ka], tmp1_act[ka])
                    c2[kb, nocc_act:, nocc_act:] -= numpy.einsum('iajd, iajb -> bd', y2[ka], tmp1_act[ka])
                    c2[kj, :nocc_act, :nocc_act] -= numpy.einsum('ialb, iajb -> lj', y3[ka], tmp1_act[ka])
                    c2[ka, nocc_act:, nocc_act:] += numpy.einsum('icjb, iajb -> ac', y3[ka], tmp1_act[ka])
                    c2[kb, nocc_act:, nocc_act:] += numpy.einsum('iajd, iajb -> bd', y3[ka], tmp1_act[ka])

                    c0 -= 4 * numpy.einsum('iajb,iajb ->', y2[ka], tmp1_act[ka]).real
                    c0 += 4 * numpy.einsum('iajb,iajb ->', y3[ka], tmp1_act[ka]).real

    # --- Tổng hợp c2 ---
    if hasattr(mp, 'second_order') and mp.second_order:
        for ki in range(nkpts):
            c2[ki, :nocc_act, :] -= numpy.einsum('ip, ki -> kp', fock_hf_act[ki, :nocc_act, :], y4[ki])
            c2[ki, :, nocc_act:] -= numpy.einsum('pa, ac -> pc', fock_hf_act[ki, :, nocc_act:], y5[ki])
            c2[ki, :nocc_act, nocc_act:] += numpy.einsum('qbj, qjbkc -> kc', y1, tmp_bar[:, ki, :, :, :, :].conj())

    c0 /= nkpts
    c1 += c2
    return c0, c1


def inter_BCH(mp, mo_energy, mo_coeff, fock_hf):
    import numpy
    from pyscf.pbc.lib import kpts_helper

    ncore = mp.ncore
    nocc_act = mp.nocc_act
    nvir_act = mp.nvir_act

    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = len(mo_energy)
    kpts = mp.kpts

    nact = mp.nact

    kconserv = kpts_helper.get_kconserv(mp._scf.cell, kpts)
    fao2mo = mp._scf.with_df.ao2mo

    # [BẢO TOÀN]: Định nghĩa Slices
    occ_act = slice(ncore, ncore + nocc_act)
    vir_act = slice(0, nvir_act)  # Chỉ đúng nếu active vir ở đầu

    # --- Khởi tạo mảng theo cấu trúc tối ưu bộ nhớ MỚI ---
    # Thay vì 8D, ta dùng 5D. Các tính toán nội không gian (inter-space) dùng nocc và nvir
    tmp1 = numpy.zeros((nkpts, nocc, nvir, nocc, nvir), dtype=complex)
    tmp1_bar = numpy.zeros((nkpts, nocc, nvir, nocc, nvir), dtype=complex)

    # Mảng lưu trữ cho bậc 2
    tmp_bar_OVov_stored = numpy.zeros((nkpts, nkpts, nocc, nvir, nocc_act, nvir_act), dtype=complex)

    h2mo_OVOV = numpy.zeros((nkpts, nocc, nvir, nocc, nvir), dtype=complex)
    h2mo_OVgV = numpy.zeros((nkpts, nocc, nvir, nact, nvir), dtype=complex)
    h2mo_OVOg = numpy.zeros((nkpts, nocc, nvir, nocc, nact), dtype=complex)

    c1 = numpy.zeros((nkpts, nact, nact), dtype=complex)
    c2 = numpy.zeros((nkpts, nact, nact), dtype=complex)

    # Mảng cộng dồn trung gian cho Bậc 2
    y1 = numpy.zeros((nkpts, nocc, nvir), dtype=complex)
    y2_OVoV = numpy.zeros((nkpts, nocc, nvir, nocc_act, nvir), dtype=complex)
    y2_oVOV = numpy.zeros((nkpts, nocc_act, nvir, nocc, nvir), dtype=complex)
    y2_OVOv = numpy.zeros((nkpts, nocc, nvir, nocc, nvir_act), dtype=complex)

    y3_OVoV = numpy.zeros((nkpts, nocc, nvir, nocc_act, nvir), dtype=complex)
    y3_OVOv = numpy.zeros((nkpts, nocc, nvir, nocc, nvir_act), dtype=complex)
    y3_OvOV = numpy.zeros((nkpts, nocc, nvir_act, nocc, nvir), dtype=complex)

    y4 = numpy.zeros((nkpts, nocc_act, nocc), dtype=complex)
    y5 = numpy.zeros((nkpts, nvir, nvir_act), dtype=complex)

    # ============================================================
    # Vòng lặp tối ưu bộ nhớ (Tính toán trên từng cặp ki, kj)
    # ============================================================
    for ki in range(nkpts):
        for kj in range(nkpts):
            # 1. Build Integrals (Inter-space)
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]

                o_i = mo_coeff[ki][:, :nocc]  # O
                o_a = mo_coeff[ka][:, nocc:]  # V
                o_j = mo_coeff[kj][:, :nocc]  # O
                o_b = mo_coeff[kb][:, nocc:]  # V
                o_gj = mo_coeff[kj][:, ncore : ncore + nact]  # g
                o_gb = mo_coeff[kb][:, ncore : ncore + nact]  # g

                # Tính (O V | O V)
                g_OVOV = fao2mo((o_i, o_a, o_j, o_b), (kpts[ki], kpts[ka], kpts[kj], kpts[kb]), compact=False)
                h2mo_OVOV[ka] = g_OVOV.reshape(nocc, nvir, nocc, nvir) / nkpts

                # Tính (O V | O g)
                g_OVOg = fao2mo((o_i, o_a, o_j, o_gb), (kpts[ki], kpts[ka], kpts[kj], kpts[kb]), compact=False)
                h2mo_OVOg[ka] = g_OVOg.reshape(nocc, nvir, nocc, nact) / nkpts

                # Tính (O V | g V)
                g_OVgV = fao2mo((o_i, o_a, o_gj, o_b), (kpts[ki], kpts[ka], kpts[kj], kpts[kb]), compact=False)
                h2mo_OVgV[ka] = g_OVgV.reshape(nocc, nvir, nact, nvir) / nkpts

            # 2. Xây dựng Amplitudes và cộng dồn Bậc 1 & Bậc 2
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]

                e_iajb = ene_denom(mp, mo_energy, ki, ka, kj, kb)[0]

                v_iajb = h2mo_OVOV[ka]
                v_ex = h2mo_OVOV[kb].transpose(0, 3, 2, 1)
                w_iajb = v_iajb - 0.5 * v_ex

                # Tính tmp1, tmp1_bar trên toàn không gian O/V
                tmp1[ka] = (v_iajb / e_iajb).conj()
                tmp1_bar[ka] = (w_iajb / e_iajb).conj()

                mp.ampf = getattr(mp, 'ampf', 1.0)
                tmp1[ka] *= mp.ampf
                tmp1_bar[ka] *= mp.ampf

                # ===============================
                # Lấy Slice trực tiếp trong vòng lặp
                # ===============================
                t_bar_OVov = tmp1_bar[ka][:, :, occ_act, vir_act]
                t_bar_OVoV = tmp1_bar[ka][:, :, occ_act, :]
                t_bar_oVOV = tmp1_bar[ka][occ_act, :, :, :]
                t_bar_OVOv = tmp1_bar[ka][:, :, :, vir_act]
                t_bar_OvOV = tmp1_bar[ka][:, vir_act, :, :]

                t_OVoV = tmp1[ka][:, :, occ_act, :]
                t_oVOV = tmp1[ka][occ_act, :, :, :]
                t_OVOv = tmp1[ka][:, :, :, vir_act]
                t_OvOV = tmp1[ka][:, vir_act, :, :]

                # ============================================================
                # BCH First Order
                # ============================================================
                if ki == ka:
                    c1[kj, nocc_act:, :nocc_act] += 2 * numpy.einsum(
                        'ai, iajb -> bj', fock_hf[ka, nocc:, :nocc].conj(), t_bar_OVov
                    )

                c1[kj, :, :nocc_act] += 2 * numpy.einsum('iajb, iapb -> pj', t_bar_OVoV, h2mo_OVgV[ka])

                # [SỬA LỖI SLICE ?????]: Chuẩn hóa lại einsum theo shape thực tế của inter_BCH cũ (rtpb -> pb)
                c1[kb, :, nocc_act:] -= 2 * numpy.einsum('iajb, iajp -> pb', t_bar_OVOv, h2mo_OVOg[ka])

                # ============================================================
                # BCH Second Order
                # ============================================================
                if hasattr(mp, 'second_order') and mp.second_order:
                    # [1]
                    if ki == ka:
                        # y1[kj] += 2 * numpy.einsum('ai, iajb -> bj', fock_hf[ka, nocc:, :nocc].conj(), tmp1_bar[ka])
                        y1[kj] += 2 * numpy.einsum('ai, iajb -> jb', fock_hf[ka, nocc:, :nocc].conj(), tmp1_bar[ka])
                        tmp_bar_OVov_stored[ki, kj] = t_bar_OVov

                    # [2][3][8]
                    y2_OVoV[ka] = numpy.einsum('ca, iclb -> ialb', fock_hf[ka, nocc:, nocc:], t_bar_OVoV.conj())
                    y2_oVOV[ka] = numpy.einsum('ca, kcjb -> kajb', fock_hf[ka, nocc:, nocc:], t_bar_oVOV.conj())
                    y2_OVOv[ka] = numpy.einsum('ca, ickb -> iakb', fock_hf[ka, nocc:, nocc:], t_bar_OVOv.conj())

                    c2[kj, :nocc_act, :nocc_act] += numpy.einsum('ialb, iajb -> lj', y2_OVoV[ka], t_OVoV)
                    c2[ki, :nocc_act, :nocc_act] += numpy.einsum('kajb, iajb -> ki', y2_oVOV[ka], t_oVOV)
                    c2[kb, nocc_act:, nocc_act:] -= numpy.einsum('iajd, iajb -> bd', y2_OVOv[ka], t_OVOv)

                    # [4][6][7]
                    y3_OVoV[ka] = numpy.einsum('ki, kalb -> ialb', fock_hf[ki, :nocc, :nocc], t_bar_OVoV.conj())
                    y3_OVOv[ka] = numpy.einsum('ki, kcjb -> icjb', fock_hf[ki, :nocc, :nocc], t_bar_OVOv.conj())
                    y3_OvOV[ka] = numpy.einsum('ki, kdja -> idja', fock_hf[ki, :nocc, :nocc], t_bar_OvOV.conj())

                    c2[kj, :nocc_act, :nocc_act] -= numpy.einsum('ialb, iajb -> lj', y3_OVoV[ka], t_OVoV)
                    c2[ka, nocc_act:, nocc_act:] += numpy.einsum('icjb, icjd -> db', y3_OVOv[ka], t_OVOv)
                    # c2[kb, nocc_act:, nocc_act:] -= numpy.einsum('idja, ibja -> bd', y3_OvOV[ka], t_OvOV)
                    c2[kb, nocc_act:, nocc_act:] += numpy.einsum('idja, ibja -> bd', y3_OvOV[ka], t_OvOV)

                    # [5] và [9]
                    y4[ki] += numpy.einsum('iajb, kajb -> ki', tmp1[ka], t_bar_oVOV.conj())
                    y5[ki] += numpy.einsum('iajb, icjb -> ac', tmp1[ka], t_bar_OvOV.conj())

    # --- Tổng hợp c2 bên ngoài vòng lặp ---
    if hasattr(mp, 'second_order') and mp.second_order:
        for ki in range(nkpts):
            # [1]
            c2[ki, nocc_act:, :nocc_act] += numpy.einsum(
                'qjb, qjbkc -> ck', y1, tmp_bar_OVov_stored[:, ki, :, :, :, :].conj()
            )

            # [5]
            c2[ki, :, :nocc_act] -= numpy.einsum('pi, ki -> pk', fock_hf[ki, ncore : ncore + nact, :nocc], y4[ki])

            # [9]
            c2[ki, :, nocc_act:] -= numpy.einsum('pa, ac -> pc', fock_hf[ki, ncore : ncore + nact, nocc:], y5[ki])

    # Trả về kết quả
    c1 += c2
    return c1


"""
def make_rdm1(mp): # , t2=None, eris=None, verbose=logger.NOTE, ao_repr=False):
    '''Spin-traced one-particle density matrix.
    The occupied-virtual orbital response is not included.

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)

    Kwargs:
        ao_repr : boolean
            Whether to transfrom 1-particle density matrix to AO
            representation.
    '''
    from pyscf.cc import ccsd_rdm

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia = mp.mo_energy[:nocc,None] - mp.mo_energy[None,nocc:] 
    eris = mp.ao2mo(mp.mo_coeff)
    
    t2 = mp.tmp_dip
    doo, dvv = _gamma1_intermediates(mp, t2, eris)
    nocc = doo.shape[0]
    nvir = dvv.shape[0]
    dov = numpy.zeros((nocc,nvir), dtype=doo.dtype)
    dvo = dov.T
    return ccsd_rdm._make_rdm1(mp, (doo, dov, dvo, dvv), with_frozen=True,
                               ao_repr=False)

def _gamma1_intermediates(mp, t2):
    if t2 is None: t2 = mp.t2
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    if t2 is None:
        if eris is None: eris = mp.ao2mo()
        mo_energy = _mo_energy_without_core(mp, mp.mo_energy)
        eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
        dtype = eris.ovov.dtype
    else:
        dtype = t2.dtype

    dm1occ = numpy.zeros((nocc,nocc), dtype=dtype)
    dm1vir = numpy.zeros((nvir,nvir), dtype=dtype)
    for i in range(nocc):
        if t2 is None:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
            t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
        else:
            t2i = t2[i]
        l2i = t2i.conj()
        dm1vir += numpy.einsum('jca,jcb->ba', l2i, t2i) * 2 \
                - numpy.einsum('jca,jbc->ba', l2i, t2i)
        dm1occ += numpy.einsum('iab,jab->ij', l2i, t2i) * 2 \
                - numpy.einsum('iab,jba->ij', l2i, t2i)
    return -dm1occ, dm1vir
"""


def _add_padding(mp, mo_coeff, mo_energy, mo_occ):
    from pyscf.pbc import tools
    from pyscf.pbc.cc.ccsd import _adjust_occ

    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts

    # Check if these are padded mo coefficients and energies
    if not numpy.all([x.shape[0] == nmo for x in mo_coeff]):
        mo_coeff = padded_mo_coeff(mp, mo_coeff)

    if not numpy.all([x.shape[0] == nmo for x in mo_energy]):
        mo_energy = padded_mo_energy(mp, mo_energy)

    if not numpy.all([x.shape[0] == nmo for x in mo_occ]):
        mo_occ = padded_mo_occ(mp, mo_occ)

    return mo_coeff, mo_energy, mo_occ


def _padding_k_idx(nmo, nocc, kind='split'):
    """A convention used for padding vectors, matrices and tensors in case when occupation numbers depend on the
    k-point index.
    Args:
        nmo (Iterable): k-dependent orbital number;
        nocc (Iterable): k-dependent occupation numbers;
        kind (str): either "split" (occupied and virtual spaces are split) or "joint" (occupied and virtual spaces are
        the joint;

    Returns:
        Two lists corresponding to the occupied and virtual spaces for kind="split". Each list contains integer arrays
        with indexes pointing to actual non-zero entries in the padded vector/matrix/tensor. If kind="joint", a single
        list of arrays is returned corresponding to the entire MO space.
    """
    if kind not in ('split', 'joint'):
        raise ValueError("The 'kind' argument must be one of 'split', 'joint'")

    if kind == 'split':
        indexes_o = []
        indexes_v = []
    else:
        indexes = []

    nocc = numpy.array(nocc)
    nmo = numpy.array(nmo)
    nvirt = nmo - nocc
    dense_o = numpy.amax(nocc)
    dense_v = numpy.amax(nvirt)
    dense_nmo = dense_o + dense_v

    for k_o, k_nmo in zip(nocc, nmo):
        k_v = k_nmo - k_o
        if kind == 'split':
            indexes_o.append(numpy.arange(k_o))
            indexes_v.append(numpy.arange(dense_v - k_v, dense_v))
        else:
            indexes.append(
                numpy.concatenate(
                    (
                        numpy.arange(k_o),
                        numpy.arange(dense_nmo - k_v, dense_nmo),
                    )
                )
            )

    if kind == 'split':
        return indexes_o, indexes_v

    else:
        return indexes


def padding_k_idx(mp, kind='split'):
    """A convention used for padding vectors, matrices and tensors in case when occupation numbers depend on the
    k-point index.

    This implementation stores k-dependent Fock and other matrix in dense arrays with additional dimensions
    corresponding to k-point indexes. In case when the occupation numbers depend on the k-point index (i.e. a metal) or
    when some k-points have more Bloch basis functions than others the corresponding data structure has to be padded
    with entries that are not used (fictitious occupied and virtual degrees of freedom). Current convention stores these
    states at the Fermi level as shown in the following example.

    +----+--------+--------+--------+
    |    |  k=0   |  k=1   |  k=2   |
    |    +--------+--------+--------+
    |    | nocc=2 | nocc=3 | nocc=2 |
    |    | nvir=4 | nvir=3 | nvir=3 |
    +====+========+========+========+
    | v3 |  k0v3  |  k1v2  |  k2v2  |
    +----+--------+--------+--------+
    | v2 |  k0v2  |  k1v1  |  k2v1  |
    +----+--------+--------+--------+
    | v1 |  k0v1  |  k1v0  |  k2v0  |
    +----+--------+--------+--------+
    | v0 |  k0v0  |        |        |
    +====+========+========+========+
    |          Fermi level          |
    +====+========+========+========+
    | o2 |        |  k1o2  |        |
    +----+--------+--------+--------+
    | o1 |  k0o1  |  k1o1  |  k2o1  |
    +----+--------+--------+--------+
    | o0 |  k0o0  |  k1o0  |  k2o0  |
    +----+--------+--------+--------+

    In the above example, `get_nmo(mp, per_kpoint=True) == (6, 6, 5)`, `get_nocc(mp, per_kpoint) == (2, 3, 2)`. The
    resulting dense `get_nmo(mp) == 7` and `get_nocc(mp) == 3` correspond to padded dimensions. This function will
    return the following indexes corresponding to the filled entries of the above table:

    >>> padding_k_idx(mp, kind="split")
    ([(0, 1), (0, 1, 2), (0, 1)], [(0, 1, 2, 3), (1, 2, 3), (1, 2, 3)])

    >>> padding_k_idx(mp, kind="joint")
    [(0, 1, 3, 4, 5, 6), (0, 1, 2, 4, 5, 6), (0, 1, 4, 5, 6)]

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        kind (str): either "split" (occupied and virtual spaces are split) or "joint" (occupied and virtual spaces are
        the joint;

    Returns:
        Two lists corresponding to the occupied and virtual spaces for kind="split". Each list contains integer arrays
        with indexes pointing to actual non-zero entries in the padded vector/matrix/tensor. If kind="joint", a single
        list of arrays is returned corresponding to the entire MO space.
    """
    return _padding_k_idx(mp.get_nmo(per_kpoint=True), mp.get_nocc(per_kpoint=True), kind=kind)


def padded_mo_occ(mp, mo_occ):
    """
    Pads occupancy of active MOs.

    Returns:
        Padded molecular occupancy.
    """
    frozen_mask = get_frozen_mask(mp)
    padding_convention = padding_k_idx(mp, kind='joint')
    nkpts = mp.nkpts

    result = numpy.zeros((nkpts, mp.nmo), dtype=mo_occ[0].dtype)
    for k in range(nkpts):
        result[numpy.ix_([k], padding_convention[k])] = mo_occ[k][frozen_mask[k]]

    return result


def padded_mo_energy(mp, mo_energy):
    """
    Pads energies of active MOs.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        mo_energy (ndarray): original non-padded molecular energies;

    Returns:
        Padded molecular energies.
    """
    frozen_mask = get_frozen_mask(mp)
    padding_convention = padding_k_idx(mp, kind='joint')
    nkpts = mp.nkpts

    result = numpy.zeros((nkpts, mp.nmo), dtype=mo_energy[0].dtype)
    for k in range(nkpts):
        result[numpy.ix_([k], padding_convention[k])] = mo_energy[k][frozen_mask[k]]

    return result


def padded_mo_coeff(mp, mo_coeff):
    """
    Pads coefficients of active MOs.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        mo_coeff (ndarray): original non-padded molecular coefficients;

    Returns:
        Padded molecular coefficients.
    """
    frozen_mask = get_frozen_mask(mp)
    padding_convention = padding_k_idx(mp, kind='joint')
    nkpts = mp.nkpts

    result = numpy.zeros((nkpts, mo_coeff[0].shape[0], mp.nmo), dtype=mo_coeff[0].dtype)
    for k in range(nkpts):
        result[numpy.ix_([k], numpy.arange(result.shape[1]), padding_convention[k])] = mo_coeff[k][:, frozen_mask[k]]

    return result


def _frozen_sanity_check(frozen, mo_occ, kpt_idx):
    """Performs a few sanity checks on the frozen array and mo_occ.

    Specific tests include checking for duplicates within the frozen array.

    Args:
        frozen (array_like of int): The orbital indices that will be frozen.
        mo_occ (:obj:`ndarray` of int): The occupuation number for each orbital
            resulting from a mean-field-like calculation.
        kpt_idx (int): The k-point that `mo_occ` and `frozen` belong to.

    """
    frozen = numpy.array(frozen)
    nocc = numpy.count_nonzero(mo_occ > 0)
    nvir = len(mo_occ) - nocc
    assert nocc, 'No occupied orbitals?\n\nnocc = %s\nmo_occ = %s' % (nocc, mo_occ)
    all_frozen_unique = (len(frozen) - len(numpy.unique(frozen))) == 0
    if not all_frozen_unique:
        raise RuntimeError('Frozen orbital list contains duplicates!\n\nkpt_idx %s\nfrozen %s' % (kpt_idx, frozen))
    if len(frozen) > 0 and numpy.max(frozen) > len(mo_occ) - 1:
        raise RuntimeError(
            'Freezing orbital not in MO list!\n\nkpt_idx %s\n'
            'frozen %s\nmax orbital idx %s' % (kpt_idx, frozen, len(mo_occ) - 1)
        )


def get_nocc(mp, per_kpoint=False):
    """Number of occupied orbitals for k-point calculations.

    Number of occupied orbitals for use in a calculation with k-points, taking into
    account frozen orbitals.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        per_kpoint (bool, optional): True returns the number of occupied
            orbitals at each k-point.  False gives the max of this list.

    Returns:
        nocc (int, list of int): Number of occupied orbitals. For return type, see description of arg
            `per_kpoint`.

    """
    for i, moocc in enumerate(mp.mo_occ):
        if numpy.any(moocc % 1 != 0):
            raise RuntimeError(
                'Fractional occupation numbers encountered @ kp={:d}: {}. This may have been caused by '
                'smearing of occupation numbers in the mean-field calculation. If so, consider '
                'executing mf.smearing_method = False; mf.mo_occ = mf.get_occ() prior to calling '
                'this'.format(i, moocc)
            )
    if mp._nocc is not None:
        return mp._nocc
    if isinstance(mp.frozen, (int, numpy.integer)):
        nocc = [(numpy.count_nonzero(mp.mo_occ[ikpt]) - mp.frozen) for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        [_frozen_sanity_check(mp.frozen, mp.mo_occ[ikpt], ikpt) for ikpt in range(mp.nkpts)]
        nocc = []
        for ikpt in range(mp.nkpts):
            max_occ_idx = numpy.max(numpy.where(mp.mo_occ[ikpt] > 0))
            frozen_nocc = numpy.sum(numpy.array(mp.frozen) <= max_occ_idx)
            nocc.append(numpy.count_nonzero(mp.mo_occ[ikpt]) - frozen_nocc)
    elif isinstance(mp.frozen[0], (list, numpy.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError(
                'Frozen list has a different number of k-points (length) than passed in mean-field/'
                'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                '(length = %d)' % (mp.nkpts, mp.frozen, nkpts)
            )
        [_frozen_sanity_check(frozen, mo_occ, ikpt) for ikpt, frozen, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]

        nocc = []
        for ikpt, frozen in enumerate(mp.frozen):
            max_occ_idx = numpy.max(numpy.where(mp.mo_occ[ikpt] > 0))
            frozen_nocc = numpy.sum(numpy.array(frozen) <= max_occ_idx)
            nocc.append(numpy.count_nonzero(mp.mo_occ[ikpt]) - frozen_nocc)
    else:
        raise NotImplementedError

    assert any(numpy.array(nocc) > 0), 'Must have occupied orbitals! \n\nnocc %s\nfrozen %s\nmo_occ %s' % (
        nocc,
        mp.frozen,
        mp.mo_occ,
    )

    if not per_kpoint:
        nocc = numpy.amax(nocc)

    return nocc


def get_nmo(mp, per_kpoint=False):
    """Number of orbitals for k-point calculations.

    Number of orbitals for use in a calculation with k-points, taking into account
    frozen orbitals.

    Note:
        If `per_kpoint` is False, then the number of orbitals here is equal to max(nocc) + max(nvir),
        where each max is done over all k-points.  Otherwise the number of orbitals is returned
        as a list of number of orbitals at each k-point.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        per_kpoint (bool, optional): True returns the number of orbitals at each k-point.
            For a description of False, see Note.

    Returns:
        nmo (int, list of int): Number of orbitals. For return type, see description of arg
            `per_kpoint`.

    """
    if mp._nmo is not None:
        return mp._nmo

    if isinstance(mp.frozen, (int, numpy.integer)):
        nmo = [len(mp.mo_occ[ikpt]) - mp.frozen for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        [_frozen_sanity_check(mp.frozen, mp.mo_occ[ikpt], ikpt) for ikpt in range(mp.nkpts)]
        nmo = [len(mp.mo_occ[ikpt]) - len(mp.frozen) for ikpt in range(mp.nkpts)]
    elif isinstance(mp.frozen, (list, numpy.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError(
                'Frozen list has a different number of k-points (length) than passed in mean-field/'
                'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                '(length = %d)' % (mp.nkpts, mp.frozen, nkpts)
            )
        [_frozen_sanity_check(fro, mo_occ, ikpt) for ikpt, fro, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]

        nmo = [len(mp.mo_occ[ikpt]) - len(mp.frozen[ikpt]) for ikpt in range(nkpts)]
    else:
        raise NotImplementedError

    assert all(numpy.array(nmo) > 0), 'Must have a positive number of orbitals!\n\nnmo %s\nfrozen %s\nmo_occ %s' % (
        nmo,
        mp.frozen,
        mp.mo_occ,
    )

    if not per_kpoint:
        # Depending on whether there are more occupied bands, we want to make sure that
        # nmo has enough room for max(nocc) + max(nvir) number of orbitals for occupied
        # and virtual space
        nocc = mp.get_nocc(per_kpoint=True)
        nmo = numpy.max(nocc) + numpy.max(numpy.array(nmo) - numpy.array(nocc))

    return nmo


def get_frozen_mask(mp):
    """Boolean mask for orbitals in k-point post-HF method.

    Creates a boolean mask to remove frozen orbitals and keep other orbitals for post-HF
    calculations.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.

    Returns:
        moidx (list of :obj:`ndarray` of `numpy.bool`): Boolean mask of orbitals to include.

    """
    moidx = [numpy.ones(x.size, dtype=numpy.bool) for x in mp.mo_occ]
    if isinstance(mp.frozen, (int, numpy.integer)):
        for idx in moidx:
            idx[: mp.frozen] = False
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        frozen = list(mp.frozen)
        for idx in moidx:
            idx[frozen] = False
    elif isinstance(mp.frozen[0], (list, numpy.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError(
                'Frozen list has a different number of k-points (length) than passed in mean-field/'
                'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                '(length = %d)' % (mp.nkpts, mp.frozen, nkpts)
            )
        [_frozen_sanity_check(fro, mo_occ, ikpt) for ikpt, fro, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]
        for ikpt, kpt_occ in enumerate(moidx):
            kpt_occ[mp.frozen[ikpt]] = False
    else:
        raise NotImplementedError

    return moidx


class OBMP2(lib.StreamObject):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):

        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        self.thresh = 1e-06
        self.shift = 0.0
        self.niter = 100
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen

        self.mom = False
        self.occ_exc = [None, None]
        self.vir_exc = [None, None]

        self.second_order = True
        self.ampf = 0.5

        self.nact = None
        self.nocc_act = None

        ##################################################
        # don't modify the following attributes, they are not input options
        self.kpts = mf.kpts
        self.nkpts = numpy.shape(mf.mo_energy)[0]
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.e_corr = None
        self.t2 = None
        self.tmp1 = None
        self.tmp1_bar = None
        self.IP = None
        self.EA = None
        self.c1 = None
        self.fock_hf = None
        self.c0_1st = None
        self._keys = set(self.__dict__.keys())

    @property
    def nocc(self):
        return self.get_nocc()

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    # int_transform = int_transform

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen != 0:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_memory %d MB (current use %d MB)', self.max_memory, lib.current_memory()[0])
        return self

    """
    @property
    def emp2(self):
        return self.e_corr

    @property
    def e_tot(self):
        return self.ene_tot #+ self._scf.e_tot
    """

    def kernel(self, shift=0.0, mo_energy=None, mo_coeff=None, mo_occ=None, with_t2=WITH_T2, _kern=kernel):
        """
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.
        """
        if mo_occ is None:
            mo_occ = self.mo_occ
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None or mo_occ is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff, mo_energy are not given.\nYou may need to call mf.kernel() to generate them.')

        mo_coeff, mo_energy, mo_occ = _add_padding(self, mo_coeff, mo_energy, mo_occ)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        # self.dump_flags()
        # _kern(self, mo_energy, mo_coeff, eris, with_t2, self.verbose)
        # self.ene_tot, self.mo_energy= _kern(self, mo_energy, mo_coeff, mo_occ, with_t2, self.verbose)
        # self._finalize()

        # sua

        (self.h1mo_act_eff, self.h2mo_act) = _kern(self, mo_energy, mo_coeff, mo_occ, with_t2, self.verbose)
        # self._finalize()
        # return self.ene_tot, self.mo_energy
        return (self.h1mo_act_eff, self.h2mo_act)

    """
    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.note(self, 'E(%s) = %.15g',
                    self.__class__.__name__, self.e_tot)
        return self
    """
    make_veff = make_veff
    # make_amp  = make_amp
    # first_BCH = first_BCH
    # second_BCH = second_BCH
    # make_rdm1 = make_rdm1
    # make_rdm2 = make_rdm2

    # as_scanner = as_scanner

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.mp import dfmp2

        mymp = dfmp2.DFMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
        if with_df is not None:
            mymp.with_df = with_df
        if mymp.with_df.auxbasis != auxbasis:
            mymp.with_df = copy.copy(mymp.with_df)
            mymp.with_df.auxbasis = auxbasis
        return mymp

    def nuc_grad_method(self):
        from pyscf.grad import mp2

        return mp2.Gradients(self)


del WITH_T2
