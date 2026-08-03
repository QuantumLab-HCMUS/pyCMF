import time
import numpy
import scipy.linalg
from pyscf import lib, dft, scf
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
import tracemalloc

def make_veff(mp):
    nocc = mp.nocc
    nocca, noccb = mp.get_nocc()
    mo_coeff = mp.mo_coeff
    mo_occ = mp.mo_occ

    dm = mp._scf.make_rdm1(mo_coeff, mo_occ)
    veff_ao = mp._scf.get_veff(mp.mol, dm)

    veffa = numpy.matmul(mo_coeff[0].T, numpy.matmul(veff_ao[0], mo_coeff[0]))
    veffb = numpy.matmul(mo_coeff[1].T, numpy.matmul(veff_ao[1], mo_coeff[1]))

    c0_hf = 0.
    for i in range(nocc[0]):
        c0_hf -= veffa[i, i]
    for i in range(nocc[1]):
        c0_hf -= veffb[i, i]

    return veffa, veffb, c0_hf

def _get_aux_blksize(mp):
    with_df = mp.with_df
    naux = int(with_df.get_naoaux())
    aux_blksize = getattr(mp, "aux_blksize", None)

    if aux_blksize is None:
        aux_blksize = getattr(with_df, "blockdim", naux)

    aux_blksize = int(aux_blksize)
    if aux_blksize <= 0:
        aux_blksize = naux

    return max(1, min(naux, aux_blksize))


def _iter_ov_blocks(mp, mo_a, nocca, mo_b, noccb):
    with_df = mp.with_df

    mo_a = numpy.asarray(mo_a, order='F')
    mo_b = numpy.asarray(mo_b, order='F')

    nmoa = mo_a.shape[1]
    nmob = mo_b.shape[1]

    ijslice_ov_a = (0, nocca, nocca, nmoa)
    ijslice_ov_b = (0, noccb, noccb, nmob)

    blksize = _get_aux_blksize(mp)

    for eri1 in with_df.loop(blksize=blksize):
        qov_a = _ao2mo.nr_e2(eri1, mo_a, ijslice_ov_a, aosym='s2', out=None)
        qov_b = _ao2mo.nr_e2(eri1, mo_b, ijslice_ov_b, aosym='s2', out=None)
        yield qov_a, qov_b


def _iter_bch_blocks(mp, mo_a, nocca, mo_b, noccb):
    with_df = mp.with_df

    mo_a = numpy.asarray(mo_a, order='F')
    mo_b = numpy.asarray(mo_b, order='F')

    nmoa = mo_a.shape[1]
    nmob = mo_b.shape[1]

    ijslice_ov_a = (0, nocca, nocca, nmoa)
    ijslice_ov_b = (0, noccb, noccb, nmob)
    ijslice_gv_a = (0, nmoa, nocca, nmoa)
    ijslice_gv_b = (0, nmob, noccb, nmob)
    ijslice_og_a = (0, nocca, 0, nmoa)
    ijslice_og_b = (0, noccb, 0, nmob)

    blksize = _get_aux_blksize(mp)

    for eri1 in with_df.loop(blksize=blksize):
        qov_a = _ao2mo.nr_e2(eri1, mo_a, ijslice_ov_a, aosym='s2', out=None)
        qov_b = _ao2mo.nr_e2(eri1, mo_b, ijslice_ov_b, aosym='s2', out=None)
        qgv_a = _ao2mo.nr_e2(eri1, mo_a, ijslice_gv_a, aosym='s2', out=None)
        qgv_b = _ao2mo.nr_e2(eri1, mo_b, ijslice_gv_b, aosym='s2', out=None)
        qog_a = _ao2mo.nr_e2(eri1, mo_a, ijslice_og_a, aosym='s2', out=None)
        qog_b = _ao2mo.nr_e2(eri1, mo_b, ijslice_og_b, aosym='s2', out=None)
        yield qov_a, qov_b, qgv_a, qgv_b, qog_a, qog_b


def make_amp(mp):
    from scipy.linalg.blas import dsyrk, dgemm

    css = mp.css
    cos = mp.cos
    log = logger.new_logger(mp, verbose=5)

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa - nocca, nmob - noccb
    mo_energy = mp.mo_energy
    mo_coeff = mp.mo_coeff

    t0 = (time.process_time(), time.perf_counter())
    from pyscf.lib import current_memory

    n_ov_a = nocca * nvira
    n_ov_b = noccb * nvirb

    assert mo_coeff[0].dtype == numpy.float64 and mo_coeff[1].dtype == numpy.float64, \
        "make_amp support only real matrix mo_coeff (float64)"

    ovov_aa = numpy.zeros((n_ov_a, n_ov_a), dtype=numpy.float64, order='F')
    ovov_bb = numpy.zeros((n_ov_b, n_ov_b), dtype=numpy.float64, order='F')
    ovov_ab = numpy.zeros((n_ov_a, n_ov_b), dtype=numpy.float64, order='F')


    for qov_a, qov_b in _iter_ov_blocks(mp, mo_coeff[0], nocca, mo_coeff[1], noccb):
        # qov_*.T la F-contiguous san;  a @ a.T  ==  qov.T @ qov
        ovov_aa = dsyrk(1.0, qov_a.T, trans=0, beta=1.0,
                        c=ovov_aa, lower=0, overwrite_c=1)
        ovov_bb = dsyrk(1.0, qov_b.T, trans=0, beta=1.0,
                        c=ovov_bb, lower=0, overwrite_c=1)
        ovov_ab = dgemm(1.0, qov_a.T, qov_b.T, trans_b=1, beta=1.0,
                        c=ovov_ab, overwrite_c=1)

    ovov_aa += numpy.triu(ovov_aa, 1).T
    ovov_bb += numpy.triu(ovov_bb, 1).T

    log.debug("qov_ab memory: %.1f MiB", current_memory()[0])
    log.timer('making amplitude: integral transform', *t0)

    d_a = mo_energy[0][:nocca, None] - mo_energy[0][None, nocca:]   # (nocca, nvira)
    d_b = mo_energy[1][:noccb, None] - mo_energy[1][None, noccb:]   # (noccb, nvirb)
    shift = mp.shift

    ovov_aa = ovov_aa.T
    ovov_bb = ovov_bb.T

    tmp1_aa = ovov_aa.reshape(nocca, nvira, nocca, nvira)
    tmp1_aa *= css
    tmp1_aa /= (d_a[:, :, None, None] + d_a[None, None, :, :] - shift)

    tmp1_bb = ovov_bb.reshape(noccb, nvirb, noccb, nvirb)
    tmp1_bb *= css
    tmp1_bb /= (d_b[:, :, None, None] + d_b[None, None, :, :] - shift)

    tmp1_ab = numpy.ascontiguousarray(ovov_ab).reshape(nocca, nvira, noccb, nvirb)
    del ovov_ab
    tmp1_ab *= cos
    tmp1_ab /= (d_a[:, :, None, None] + d_b[None, None, :, :] - shift)

    tmp1_ba = numpy.ascontiguousarray(tmp1_ab.transpose(2, 3, 0, 1))

    tmp1_bar_aa = tmp1_aa - numpy.transpose(tmp1_aa, (0, 3, 2, 1))
    tmp1_bar_bb = tmp1_bb - numpy.transpose(tmp1_bb, (0, 3, 2, 1))
    tmp1_bar_ab = tmp1_ab
    tmp1_bar_ba = tmp1_ba

    log.debug("t_mp1 memory: %.1f MiB", current_memory()[0])

    tmp1 = (tmp1_aa, tmp1_bb, tmp1_ab, tmp1_ba)
    tmp1_bar = (tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab, tmp1_bar_ba)

    return tmp1, tmp1_bar


def first_BCH(mp, fock_hfa, fock_hfb, tmp1_bar, c0):
    
    log = logger.new_logger(mp, verbose=5)
    tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab, tmp1_bar_ba = tmp1_bar

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa - nocca, nmob - noccb
    mo_coeff = mp.mo_coeff

    t0 = (time.process_time(), time.perf_counter())
    from pyscf.lib import current_memory

    c1_a = numpy.zeros((nmoa, nmoa), dtype=fock_hfa.dtype)
    c1_b = numpy.zeros((nmob, nmob), dtype=fock_hfb.dtype)

    # Làm phẳng một lần, dùng lại cho mọi block DF
    Taa = tmp1_bar_aa.reshape(nocca * nvira, nocca * nvira)
    Tbb = tmp1_bar_bb.reshape(noccb * nvirb, noccb * nvirb)
    Tab = tmp1_bar_ab.reshape(nocca * nvira, noccb * nvirb)
    Tba = tmp1_bar_ba.reshape(noccb * nvirb, nocca * nvira)

    for qov_a, qov_b, qgv_a, qgv_b, qog_a, qog_b in _iter_bch_blocks(
        mp, mo_coeff[0], nocca, mo_coeff[1], noccb
    ):
        naux_blk = qov_a.shape[0]

        # --- Bước đắt duy nhất: 4 dgemm ---
        M_A = (numpy.dot(qov_a, Taa) + numpy.dot(qov_b, Tba)).reshape(naux_blk, nocca, nvira)
        M_B = (numpy.dot(qov_b, Tbb) + numpy.dot(qov_a, Tab)).reshape(naux_blk, noccb, nvirb)

        qov_a3 = qov_a.reshape(naux_blk, nocca, nvira)
        qov_b3 = qov_b.reshape(naux_blk, noccb, nvirb)
        qgv_a3 = qgv_a.reshape(naux_blk, nmoa, nvira)
        qgv_b3 = qgv_b.reshape(naux_blk, nmob, nvirb)
        qog_a3 = qog_a.reshape(naux_blk, nocca, nmoa)
        qog_b3 = qog_b.reshape(naux_blk, noccb, nmob)

        # c1[:, occ]  <- contract qua (Q, b)
        c1_a[:, 0:nocca] += 2.0 * numpy.tensordot(qgv_a3, M_A, axes=((0, 2), (0, 2)))
        c1_b[:, 0:noccb] += 2.0 * numpy.tensordot(qgv_b3, M_B, axes=((0, 2), (0, 2)))

        # c1[:, vir]  <- contract qua (Q, j)
        c1_a[:, nocca:nmoa] -= 2.0 * numpy.tensordot(qog_a3, M_A, axes=((0, 1), (0, 1)))
        c1_b[:, noccb:nmob] -= 2.0 * numpy.tensordot(qog_b3, M_B, axes=((0, 1), (0, 1)))

        # c0
        c0 -= numpy.dot(qov_a3.ravel(), M_A.ravel())
        c0 -= numpy.dot(qov_b3.ravel(), M_B.ravel())

        del M_A, M_B

    log.debug("first BCH DF-block memory: %.1f MiB", current_memory()[0])
    log.timer('first BCH: integral transform', *t0)

    c1_a[:nocca, nocca:] += 2.0 * lib.einsum('ijkl,ij->kl', tmp1_bar_aa, fock_hfa[:nocca, nocca:])
    c1_a[:nocca, nocca:] += 2.0 * lib.einsum('ijkl,ij->kl', tmp1_bar_ba, fock_hfb[:noccb, noccb:])
    c1_b[:noccb, noccb:] += 2.0 * lib.einsum('ijkl,ij->kl', tmp1_bar_bb, fock_hfb[:noccb, noccb:])
    c1_b[:noccb, noccb:] += 2.0 * lib.einsum('ijkl,ij->kl', tmp1_bar_ab, fock_hfa[:nocca, nocca:])

    return c0, c1_a, c1_b

def second_BCH(mp, fock_a, fock_b, fock_hfa, fock_hfb, tmp1, tmp1_bar, c0):
    tmp1_aa, tmp1_bb, tmp1_ab, tmp1_ba = tmp1
    tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab, tmp1_bar_ba = tmp1_bar
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()

    log = logger.new_logger(mp, verbose=5)
    t0 = (time.process_time(), time.perf_counter())
    detail = bool(getattr(mp, "timer_detail", False))
    from pyscf.lib import current_memory

    c1_a = numpy.zeros((nmoa, nmoa), dtype=fock_hfa.dtype)
    c1_b = numpy.zeros((nmob, nmob), dtype=fock_hfb.dtype)

    # ---- Block [1]: giữ nguyên, chi phí không đáng kể ----
    y1_a = lib.einsum('ij,ijkl -> kl', fock_hfa[:nocca, nocca:], tmp1_bar_aa)
    y1_a += lib.einsum('ij,ijkl -> kl', fock_hfb[:noccb, noccb:], tmp1_bar_ba)
    c1_a[:nocca, nocca:] += lib.einsum('ijkl,kl -> ij', tmp1_bar_aa, y1_a)
    c1_b[:noccb, noccb:] += lib.einsum('ijkl,kl -> ij', tmp1_bar_ba, y1_a)

    y1_b = lib.einsum('ij,ijkl -> kl', fock_hfb[:noccb, noccb:], tmp1_bar_bb)
    y1_b += lib.einsum('ij,ijkl -> kl', fock_hfa[:nocca, nocca:], tmp1_bar_ab)
    c1_a[:nocca, nocca:] += lib.einsum('ijkl,kl -> ij', tmp1_bar_ab, y1_b)
    c1_b[:noccb, noccb:] += lib.einsum('ijkl,kl -> ij', tmp1_bar_bb, y1_b)

    if detail:
        log.timer('  second BCH [1]', *t0)

    # ---- Block [2]-[9]: gộp, duyệt từng cặp spin một lần ----
    c1 = {'a': c1_a, 'b': c1_b}
    nocc = {'a': nocca, 'b': noccb}
    fock = {'a': fock_hfa, 'b': fock_hfb}
    acc_o = {'a': None, 'b': None}   # tích luỹ cho [5]
    acc_v = {'a': None, 'b': None}   # tích luỹ cho [9]

    t_build_V = t_build_O = t_contract = 0.0

    # (x, y) = spin của cặp chỉ số thứ nhất / thứ hai trong tmp1_xy
    for x, y, t, tb in (('a', 'a', tmp1_aa, tmp1_bar_aa),
                        ('b', 'b', tmp1_bb, tmp1_bar_bb),
                        ('a', 'b', tmp1_ab, tmp1_bar_ab),
                        ('b', 'a', tmp1_ba, tmp1_bar_ba)):
        nx, ny = nocc[x], nocc[y]
        Fx = fock[x]
        tp = (time.process_time(), time.perf_counter())

        # V: contract chỉ số virtual — ĐẮT (1U). Bản gốc dựng cái này 3 lần.
        _w = time.perf_counter()
        V = lib.einsum('ac,icjd -> iajd', Fx[nx:, nx:], tb)
        t_build_V += time.perf_counter() - _w

        _w = time.perf_counter()
        c1[y][:ny, :ny] += lib.einsum('iajb,iakb -> jk', t, V)      # [2]
        c1[x][:nx, :nx] += lib.einsum('iajb,kajb -> ik', t, V)      # [3]
        c0 -= lib.einsum('ijkl,ijkl->', t, V)                       # [2]
        t_contract += time.perf_counter() - _w

        # O: contract chỉ số occupied — rẻ (nocc^3 nvir^2). Bản gốc dựng 3 lần.
        _w = time.perf_counter()
        O = lib.einsum('ik,kajd -> iajd', Fx[:nx, :nx], tb)
        t_build_O += time.perf_counter() - _w

        _w = time.perf_counter()
        c1[y][:ny, :ny] -= lib.einsum('iajb,ialb -> jl', t, O)      # [4]
        c1[x][nx:, nx:] += lib.einsum('iajb,icjb -> ac', t, O)      # [7]
        c0 += lib.einsum('ijkl,ijkl->', t, O)                       # [4]

        # [6] + [8] gộp: c1[vv] += <t | O> - <t | V>  =  <t | O - V>
        O -= V
        del V
        c1[y][ny:, ny:] += lib.einsum('iajb,iajd -> bd', t, O)      # [6]+[8]
        del O

        # [5] va [9]: contract voi chinh tmp1_bar
        r_o = lib.einsum('iajb,kajb -> ik', t, tb)
        r_v = lib.einsum('iajb,icjb -> ac', t, tb)
        acc_o[x] = r_o if acc_o[x] is None else acc_o[x] + r_o
        acc_v[x] = r_v if acc_v[x] is None else acc_v[x] + r_v
        t_contract += time.perf_counter() - _w

        if detail:
            log.timer(f'  second BCH cap ({x}{y})', *tp)

    for s in ('a', 'b'):
        ns = nocc[s]
        c1[s][:, :ns] -= lib.einsum('pk,ik -> pi', fock[s][:, :ns], acc_o[s])   # [5]
        c1[s][:, ns:] -= lib.einsum('pa,ac -> pc', fock[s][:, ns:], acc_v[s])   # [9]

    if detail:
        log.debug("second BCH memory: %.1f MiB", current_memory()[0])
        log.debug("second BCH breakdown (wall): dung V %.2f s | dung O %.2f s "
                  "| contract %.2f s", t_build_V, t_build_O, t_contract)

    log.timer('second BCH', *t0)

    return c0, c1_a, c1_b

def obmp2_iter(mp, mol, mf_emb, xc_code, v_emb=None, niter=1000):
    nmoa = mf_emb.mo_coeff[0].shape[1] 
    nmob = mf_emb.mo_coeff[1].shape[1]
    nocca = numpy.count_nonzero(mf_emb.mo_occ[0] > 0)
    noccb = numpy.count_nonzero(mf_emb.mo_occ[1] > 0)

    idx_a = numpy.argsort(mf_emb.mo_occ[0])[::-1]
    idx_b = numpy.argsort(mf_emb.mo_occ[1])[::-1]

    idx_occ_a = idx_a[:nocca]
    idx_occ_b = idx_b[:noccb]
    idx_vir_a = idx_a[nocca:nmoa]
    idx_vir_b = idx_b[noccb:nmob]

    mo_coeff_init = (
        numpy.empty_like(mf_emb.mo_coeff[0]),
        numpy.empty_like(mf_emb.mo_coeff[1]),
    )

    mo_coeff_init[0][:, :nocca] = mf_emb.mo_coeff[0][:, idx_occ_a]
    mo_coeff_init[0][:, nocca:nmoa] = mf_emb.mo_coeff[0][:, idx_vir_a]

    mo_coeff_init[1][:, :noccb] = mf_emb.mo_coeff[1][:, idx_occ_b]
    mo_coeff_init[1][:, noccb:nmob] = mf_emb.mo_coeff[1][:, idx_vir_b]

    mf_emb.mo_coeff = (
        mo_coeff_init[0].copy(),
        mo_coeff_init[1].copy(),
    )

    mf_emb.mo_energy = (
        numpy.concatenate((mf_emb.mo_energy[0][idx_occ_a], mf_emb.mo_energy[0][idx_vir_a])),
        numpy.concatenate((mf_emb.mo_energy[1][idx_occ_b], mf_emb.mo_energy[1][idx_vir_b])),
    )

    mf_emb.mo_occ = (
        numpy.concatenate((numpy.ones(nocca), numpy.zeros(nmoa - nocca))),
        numpy.concatenate((numpy.ones(noccb), numpy.zeros(nmob - noccb))),
    )

    dm = mf_emb.make_rdm1(mf_emb.mo_coeff, mf_emb.mo_occ)
    s1e = mf_emb.get_ovlp(mol)
    A = scipy.linalg.fractional_matrix_power(s1e, -0.5).real
    h1e = mf_emb.get_hcore(mol)
    vhf = mf_emb.get_veff(mol, dm)
    nuc = mf_emb.energy_nuc()
    
    is_hybrid = getattr(mp, 'is_hybrid', True)

    ks = dft.UKS(mol).density_fit()
    ks.xc = xc_code
    ks.verbose = mol.verbose
    ks.with_df = mp.with_df

    F_list_a = []
    F_list_b = []
    DIIS_RESID = []
    
    ene_old = None
    conv = False
    min_iter = int(getattr(mp, "min_iter", 2))
    r_thresh = float(getattr(mp, "r_thresh", 1e-5))

    if v_emb is None:
        v_emb = [0, 0]

    for it in range(niter):

        _mom_method = (
        bool(getattr(mp, "mom_select", False))
        and it >= int(getattr(mp, "mom_start_cycle", 2))
        and (not mp.use_embed or bool(getattr(mp, "mom_in_embed", False)))
        )

        if _mom_method:
            print(f"[MOM_method]: Running in iter {it}")
            mf_emb = mp.mom_occ_(mf_emb, mo_coeff_init)

            mp.mo_coeff  = mf_emb.mo_coeff
            mp.mo_energy = mf_emb.mo_energy
            mp.mo_occ    = mf_emb.mo_occ
            mp._nocc     = (nocca, noccb)
            mp._nmo      = (nmoa, nmob)

            dm = mf_emb.make_rdm1(mf_emb.mo_coeff, mf_emb.mo_occ)  

        h1ao = mf_emb.get_hcore(mol)
        h1mo_a = numpy.matmul(mf_emb.mo_coeff[0].T, numpy.matmul(h1ao, mf_emb.mo_coeff[0]))
        h1mo_b = numpy.matmul(mf_emb.mo_coeff[1].T, numpy.matmul(h1ao, mf_emb.mo_coeff[1]))
        
        fock_hfa = h1mo_a
        fock_hfb = h1mo_b

        mp.mo_coeff = mf_emb.mo_coeff
        mp.mo_occ = mf_emb.mo_occ
        mp.mo_energy = mf_emb.mo_energy
        mp._scf = mf_emb 
        mp._nocc = (nocca, noccb)
        mp._nmo = (nmoa, nmob)

        veffa, veffb, c0 = make_veff(mp) 
        
        fock_hfa += veffa
        fock_hfb += veffb

        fock_uobmp2_a = numpy.zeros((nmoa,nmoa), dtype=fock_hfa.dtype)
        fock_uobmp2_b = numpy.zeros((nmob,nmob), dtype=fock_hfb.dtype)

        fock_uobmp2_a += fock_hfa
        fock_uobmp2_b += fock_hfb

        ene_hf = 0.
        for i in range(nocca): ene_hf += fock_uobmp2_a[i,i]
        for i in range(noccb): ene_hf += fock_uobmp2_b[i,i]
        c0 *= 0.5
        ene_hf += c0

        if is_hybrid:
            vxc = ks.get_veff(mol, dm)
            fock_dft_raw = ks.get_fock(h1e, s1e, vxc, dm, diis_start_cycle=it)
            fock_dft = numpy.array([fock_dft_raw[0] + v_emb[0], fock_dft_raw[1] + v_emb[1]])
            
            fock_dft_a = numpy.matmul(mf_emb.mo_coeff[0].T, numpy.matmul(fock_dft[0], mf_emb.mo_coeff[0]))
            fock_dft_b = numpy.matmul(mf_emb.mo_coeff[1].T, numpy.matmul(fock_dft[1], mf_emb.mo_coeff[1]))
            ene_dft = ks.energy_elec(dm, h1e, vxc)[0] + nuc
        else:
            ene_dft = 0.0

        vhf = mf_emb.get_veff(mol, dm)
        fock_hf_pyscf = mf_emb.get_fock(h1e, s1e, vhf, dm, diis_start_cycle=it) 
        
        fock_hf_pyscf_a = numpy.matmul(mf_emb.mo_coeff[0].T, numpy.matmul(fock_hf_pyscf[0], mf_emb.mo_coeff[0]))
        fock_hf_pyscf_b = numpy.matmul(mf_emb.mo_coeff[1].T, numpy.matmul(fock_hf_pyscf[1], mf_emb.mo_coeff[1]))
        
        e_elec_hfpyscf = mf_emb.energy_elec(dm, h1e, vhf)[0]
        ene_hfpyscf = e_elec_hfpyscf + nuc

        tmp1, tmp1_bar = make_amp(mp) 
        tmp1_aa, tmp1_bb, tmp1_ab, tmp1_ba = tmp1
        tmp1_bar_aa, tmp1_bar_bb, tmp1_bar_ab, tmp1_bar_ba = tmp1_bar

        if mp.second_order:
            mp.ampf = 1.0

        tmp1_bar_aa *= mp.ampf
        tmp1_bar_bb *= mp.ampf
        tmp1_bar_ab *= mp.ampf
        tmp1_bar_ba *= mp.ampf

        c0, c1_a, c1_b = first_BCH(mp, fock_hfa, fock_hfb, tmp1_bar, c0)
        
        fock_uobmp2_a += 0.5 * (c1_a + c1_a.T)
        fock_uobmp2_b += 0.5 * (c1_b + c1_b.T)  

        if mp.second_order:
            c0, c1_a, c1_b = second_BCH(mp, fock_uobmp2_a, fock_uobmp2_b, fock_hfa, fock_hfb, tmp1, tmp1_bar, c0)
            fock_uobmp2_a += 0.5 * (c1_a + c1_a.T)
            fock_uobmp2_b += 0.5 * (c1_b + c1_b.T) 

        ene = c0
        for i in range(nocca): ene += 1. * fock_uobmp2_a[i,i]
        for i in range(noccb): ene += 1. * fock_uobmp2_b[i,i]
        
        ene_uobmp2 = ene + nuc

        if is_hybrid:
            e_tot = (ene_dft) + (ene_uobmp2 - ene_hfpyscf) * mp.alphaa[1]
            e_corr_hybrid = (ene_uobmp2 - ene_hfpyscf) * mp.alphaa[1]
            fock_udftobmp2_a = (fock_dft_a) + (fock_uobmp2_a - fock_hf_pyscf_a) * mp.alphaa[1] 
            fock_udftobmp2_b = (fock_dft_b) + (fock_uobmp2_b - fock_hf_pyscf_b) * mp.alphaa[1] 
            
            de = abs(e_tot - ene_old) if ene_old is not None else numpy.inf
            ene_old = e_tot
            
            F_eff_mo_a = fock_udftobmp2_a 
            F_eff_mo_b = fock_udftobmp2_b 
        else:
            e_corr = (ene_uobmp2 - ene_hfpyscf)
            e_tot = e_corr # Lấy e_corr làm tiêu chí hội tụ cho loop
            fock_udftobmp2_a = (fock_uobmp2_a - fock_hf_pyscf_a) 
            fock_udftobmp2_b = (fock_uobmp2_b - fock_hf_pyscf_b)
            
            de = abs(e_corr - ene_old) if ene_old is not None else numpy.inf
            ene_old = e_corr

            # For Pure OBMP2, DIIS Fock = HF + UOBMP2 - HF = UOBMP2
            F_eff_mo_a = fock_hf_pyscf_a + fock_udftobmp2_a 
            F_eff_mo_b = fock_hf_pyscf_b + fock_udftobmp2_b 

        # DIIS
        C_a = mf_emb.mo_coeff[0]
        C_b = mf_emb.mo_coeff[1]

        F_a = s1e @ C_a @ F_eff_mo_a @ C_a.T @ s1e
        F_b = s1e @ C_b @ F_eff_mo_b @ C_b.T @ s1e

        F_a = 0.5 * (F_a + F_a.T)
        F_b = 0.5 * (F_b + F_b.T)

        C_occa = C_a[:, :nocca]
        C_occb = C_b[:, :noccb]

        D_a = numpy.einsum('pi,qi->pq', C_occa, C_occa, optimize=True)
        D_b = numpy.einsum('pi,qi->pq', C_occb, C_occb, optimize=True)

        err_a_ao  = F_a @ D_a @ s1e - s1e @ D_a @ F_a
        err_b_ao  = F_b @ D_b @ s1e - s1e @ D_b @ F_b
        err_ab_ao = F_a @ D_a @ s1e - s1e @ D_b @ F_b
        err_ba_ao = F_b @ D_b @ s1e - s1e @ D_a @ F_a

        r_a  = A.T @ err_a_ao  @ A
        r_b  = A.T @ err_b_ao  @ A
        r_ab = A.T @ err_ab_ao @ A
        r_ba = A.T @ err_ba_ao @ A

        diis_r = (r_a + r_b + 50.0 * r_ab + 50.0 * r_ba).real
        dRMS = numpy.mean(diis_r**2) ** 0.5

        # Print tương ứng với method
        if is_hybrid:
            print(f"Iter {it}: E_tot={e_tot:.12f}, E_corr={e_corr_hybrid:.12f}, dE={de:.8e}, dRMS={dRMS:.8e}")
        else:
            print(f"Iter {it}: E_corr={e_corr:.12f}, dE={de:.8e}, dRMS={dRMS:.8e}")

        F_list_a.append(F_a.copy())
        F_list_b.append(F_b.copy())
        DIIS_RESID.append(diis_r.copy())

        diis_space = int(getattr(mp, "diis_space", 8))
        if diis_space < 1:
            diis_space = 1

        while len(DIIS_RESID) > diis_space:
            F_list_a.pop(0)
            F_list_b.pop(0)
            DIIS_RESID.pop(0)

        if it >= 2:
            B_dim = len(DIIS_RESID) + 1
            B = numpy.empty((B_dim, B_dim))
            B[-1, :] = -1.0
            B[:, -1] = -1.0
            B[-1, -1] = 0.0

            for i in range(len(DIIS_RESID)):
                for j in range(len(DIIS_RESID)):
                    B[i, j] = numpy.einsum('ij,ij->', DIIS_RESID[i], DIIS_RESID[j], optimize=True)

            rhs = numpy.zeros(B_dim)
            rhs[-1] = -1.0

            try:
                coeff = numpy.linalg.solve(B, rhs)[:-1]
            except numpy.linalg.LinAlgError:
                coeff = numpy.linalg.lstsq(B, rhs, rcond=1e-12)[0][:-1]

            F_a = numpy.zeros_like(F_list_a[0])
            F_b = numpy.zeros_like(F_list_b[0])
            for c, Fa_i, Fb_i in zip(coeff, F_list_a, F_list_b):
                F_a += c * Fa_i
                F_b += c * Fb_i

            F_a = 0.5 * (F_a + F_a.T)
            F_b = 0.5 * (F_b + F_b.T)

        # Keep the orbital update inside the current CL/truncated MO subspace.
        # Do NOT use scipy.linalg.eigh(F_a, s1e) here, because that would expand
        # the solution back to the full AO space and destroy the truncation.
        F_a_mo = C_a.T @ F_a @ C_a
        F_b_mo = C_b.T @ F_b @ C_b

        F_a_mo = 0.5 * (F_a_mo + F_a_mo.T)
        F_b_mo = 0.5 * (F_b_mo + F_b_mo.T)

        eps_new_a, C_rot_a = scipy.linalg.eigh(F_a_mo)
        eps_new_b, C_rot_b = scipy.linalg.eigh(F_b_mo)

        mo_coeff_new_a = C_a @ C_rot_a
        mo_coeff_new_b = C_b @ C_rot_b

        mf_emb.mo_coeff  = (mo_coeff_new_a, mo_coeff_new_b)
        mf_emb.mo_energy = (eps_new_a,      eps_new_b)

        mp.mo_coeff  = mf_emb.mo_coeff
        mp.mo_energy = mf_emb.mo_energy

        if it + 1 >= min_iter and de <= mp.thresh:
            conv = True
            break
    
        dm = mf_emb.make_rdm1(mf_emb.mo_coeff, mf_emb.mo_occ)
        dm = lib.tag_array(dm, mo_coeff=mf_emb.mo_coeff, mo_occ=mf_emb.mo_occ)
    mp.converged = conv
    dm_total = mf_emb.make_rdm1(mf_emb.mo_coeff, mf_emb.mo_occ)
    return e_tot, ene_dft, (dm_total[0], dm_total[1])