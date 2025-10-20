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


'''
Periodic OB-MP2
'''

import time, logging, tracemalloc
from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf import cc
from pyscf.lib import logger
from pyscf.pbc.lib import kpts_helper
from pyscf import __config__
from pyscf.pbc.mp import kmp2

import opt_einsum as oe
from pyscf import lib
import time
WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)
LARGE_DENOM = getattr(__config__, 'LARGE_DENOM', 1e14)

'''
Trong chương trình này thì hàm kernel là hàm chính để tính toán OB-MP2. 
Hàm này nhận vào các đối số sau:
- mp: Đối tượng MP2 chứa các thông số và phương thức
- mo_energy: Mảng năng lượng của các orbital phân tử (MO)
- mo_coeff: Hệ số của các orbital phân tử
- mo_occ: Trạng thái chiếm đóng của các orbital phân tử
- with_t2: Có tính toán biên độ t2 hay không
- verbose: Mức độ hiển thị thông tin

- Phương pháp OBMP2 tốt hơn MP2 ở chỗ nó TỐI ƯU HÓA các orbital phân tử qua
 một quy trình lặp để cải thiện mô tả năng lượng tương quan điện tử.
- Phương pháp OBMP2 là một cải tiến của MP2 cổ điển, nhằm tối ưu hóa các orbital
 phân tử để MÔ TẢ TỐT HƠN hiệu ứng tương quan điện tử. Thuật toán sử dụng phép biến đổi
   BCH để xây dựng ma trận Fock hiệu dụng, rồi CHÉO HÓA để tạo ra bộ orbital tốt hơn qua
     các vòng lặp. Khi hội tụ, phương pháp này cho ra năng lượng tương quan chính xác hơn
       so với MP2 truyền thống.

Vượt qua giới hạn của orbital Hartree-Fock:
- Phương pháp Hartree-Fock thông thường cung cấp một bộ orbital cố định
-> OBMP2 cho phép "thư giãn" các orbital dựa trên thông tin tương quan MP2.

Cơ sở lý thuyết:
- Biến đổi BCH cho phép đưa hiệu ứng của toán tử tương quan vào ma trận Fock
-> Vòng lặp này thực hiện một dạng của "lý thuyết nhiễu loạn tự hợp" để tối ưu hóa các orbital.

Cải thiện so với MP2 cổ điển:
- MP2 thường thường chỉ tính năng lượng tương quan trên các orbital Hartree-Fock cố định
- OBMP2 vượt trội hơn vì tìm ra các orbital tối ưu cho việc mô tả tương quan
- Năng lượng tương quan (e_corr = ene_tot - ene_hf) thường chính xác hơn

Ý nghĩa của sự hội tụ:
- Khi vòng lặp hội tụ, chúng ta đã tìm được bộ orbital mà khi sử dụng chúng trong lý thuyết
 nhiễu loạn MP2 sẽ cho năng lượng ổn định nhất.
- Các orbital này thường khác với orbital Hartree-Fock và cho phép tính toán chính xác hơn
 các thuộc tính điện tử khác.

'''
def kernel(mp, mo_energy, mo_coeff, mo_occ, with_t2=WITH_T2,
           verbose=logger.NOTE):
    
    # Khởi tạo
    nuc = mp._scf.energy_nuc() # Năng lượng hạt nhân
    nmo = mp.nmo # Số orbital phân tử
    nkpts = numpy.shape(mo_energy)[0] # Số điểm k
    nocc = mp.nocc # Số orbital chiếm đóng
    niter = mp.niter # Số vòng lặp tối đa
    ene_old = 0.  # Năng lượng ban đầu
    dm = mp._scf.make_rdm1(mo_coeff, mo_occ) # Ma trận mật độ
    
    # Chuẩn bị DIIS (Direct Inversion in the Iterative Subspace) <Tìm hiểu sau>
    DIIS_RESID = [[] for _ in range(nkpts)]
    F_list = [[] for _ in range(nkpts)]

    # Khởi tạo quá trình OBMP2
    print()
    print('**********************************')
    print('************** OBMP2 *************')
    sort_idx = numpy.argsort(mo_energy) # Sắp xếp các orbital theo năng lượng (Vẫn là Orbital Hartree-Fock chưa tối ưu hóa)
    print(sort_idx) 

    '''
    Vòng lặp này là trung tâm của phương pháp OBMP2 và đóng vai trò
      tìm orbital phân tử tối ưu để cải thiện mô tả tương quan điện tử.    
    '''

    for it in range(niter):

        '''
        a. Xây dựng Hamiltonian và potential hiệu dụng
        '''

        h1ao = mp._scf.get_hcore() # Tính toán ma trận Hamiltonian lõi (h1ao) trong không gian orbital nguyên tử (AO)
        veffao = mp._scf.get_veff(mp._scf.cell, dm) # Tính toán thế năng hiệu dụng (veffao) dựa trên ma trận mật độ hiện tại
        
        #
        #h1mo = numpy.zeros((nkpts, nmo, nmo), dtype=complex)
        #for k in range(nkpts):
        #    h1mo[k] = numpy.matmul(mo_coeff[k].T.conj(),numpy.matmul(h1ao[k], mo_coeff[k]))

        #####################
        ### Hartree-Fock
        #veff_ao, veff, c0_hf = make_veff(mp, mo_coeff, mo_energy)

        '''
        b. Tính toán Hartree-Fock
        - Chuyển đổi các ma trận từ không gian AO sang không gian MO hiện tại
        - Tạo ma trận Fock cơ sở Hartree-Fock để sử dụng trong các bước tiếp theo
        '''

        # Chuyển đổi potential hiệu dụng từ AO sang MO
        veff= [reduce(numpy.dot, (mo.T.conj(), veffao[k], mo))
                                for k, mo in enumerate(mp.mo_coeff)]
        
        # Tính hằng số c0_hf
        c0_hf = 0
        for kp in range(nkpts):
            for i in range(nocc):
                c0_hf -=  veff[kp][i,i].real
        c0_hf/= nkpts

        #fock_hf = h1mo
        #fock_hf += veff

        # Xây dựng ma trận Fock trong cơ sở Hartree-Fock
        fock_hf = numpy.zeros((nkpts, nmo, nmo), dtype=complex) # Khởi tạo ma trận Fock
        fock_hf += veff
        fock_hf += [reduce(numpy.dot, (mo.T.conj(), h1ao[k], mo))
                                for k, mo in enumerate(mp.mo_coeff)]
        numpy.set_printoptions(precision=6)
        fockao = h1ao + veffao
        #print("fock_hf")
        #print(fock_hf[0])
        #initializing w/ HF
        fock = 0
        fock += fock_hf
        c0 = c0_hf 

        '''
        c. Tính năng lượng Hartree-Fock
        '''

        ene_hf = 0
        for k in range(nkpts):
            for i in range(nocc):
                ene_hf += 2*fock[k][i,i].real/nkpts

        ene_hf +=c0_hf + nuc 
        # Năng lượng Hartree-Fock = năng lượng điện tử + năng lượng hạt nhân
        # <Cần phải xem lại coi c0_hf là năng lượng gì?>

        if  mp.second_order:
            mp.ampf = 1.0
        
        #####################
        ### MP1 amplitude
        #tmp1, tmp1_bar, h2mo_ovgg = (mp, mo_energy, mo_coeff)
        
        #####################

        '''
        d. Thực hiện biến đổi BCH (Baker-Campbell-Hausdorff) bậc 1 và bậc 2
        (Trong thực tế trong hàm first_BCH không chỉ chứa BCH bậc 1 mà còn có chứa cả bậc 2 nữa)
        - Đây là bước quan trọng nhất trong OBMP2: đưa thông tin tương quan MP2 vào ma trận Fock.
        - c1 chứa các hiệu chỉnh cho ma trận Fock dựa trên thông tin tương quan
        '''

        # Tính toán ma trận Fock
        ### BCH 1st order  
        c0_1st, c1, IP, EA = first_BCH(mp, mp.mo_energy, mp.mo_coeff, fock_hf)
        print("c1+c1.T")
        print(c1[0] + c1[0].T.conj())    
        for k in range(nkpts):
            fock[k] += (c1[k] + c1[k].T.conj())
            #print(abs(fock[k] - fock[k].T.conj()) < 1e-15)
        #####################
        ### BCH 2nd order  
        #if mp.second_order:

        #    c0_2nd, c1 = second_BCH(mp, mo_coeff, mo_energy, fock_hf)
            # symmetrize c1
        #    for k in range(nkpts):
        #        fock[k] += (c1[k] + c1[k].T.conj())

        '''
        e. Tính năng lượng tổng
        '''

        ene = 0
        for k in range(nkpts):
            for i in range(nocc):
                ene += 2*fock[k][i,i].real/nkpts

        ene_tot = ene + c0  + nuc + c0_1st 
        print('e_corr = ',ene_tot - ene_hf) 

        '''
        f. Kiểm tra hội tụ
        So sánh với năng lượng của vòng lặp trước đó để kiểm tra hội tụ
        -> Nếu sự thay đổi năng lượng nhỏ hơn ngưỡng (mp.thresh), vòng lặp kết thúc
        '''

        # Nếu độ chệnh lệch của năng lượng trước và sau vòng lặp nhỏ hơn 1 con số nào đó => vòng lặp kết thúc (hội tụ)
        de = abs(ene_tot - ene_old)
        ene_old = ene_tot
        tracemalloc.start(25)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        stat = top_stats[:10]
        total_mem = sum(stat.size for stat in top_stats)
        print()
        print('iter = %d'%it, ' ene = %8.8f'%ene_tot, ' ene diff = %8.8f'%de, flush=True)
        #print("Total allocated size: %.3f Mb" % (total_mem / 1024**2))
        print(mo_energy[0])
        print()
        #print(mo_energy[1])
        if de < mp.thresh:
            break

        ## diagonalizing correlated Fock 
        '''
        g. Cập nhật orbital phân tử
        '''
        
        # Chéo hóa ma trận Fock để tạo orbital mới

        '''
        - Đối góc hóa ma trận Fock đã hiệu chỉnh để tạo ra bộ orbital mới
        - Orbital mới này có thông tin về tương quan điện tử (khác với orbital Hartree-Fock)
        - Sắp xếp lại orbital theo thứ tự năng lượng tăng dần (dùng sort_idx)
        - Orbital mới sẽ được sử dụng làm đầu vào cho vòng lặp tiếp theo
        '''
        new_mo_coeff = numpy.empty_like(mo_coeff)
        new_mo_energy = numpy.empty_like(mo_energy, dtype=complex)
        
        for k in range(nkpts):
            
            new_mo_energy[k], U = scipy.linalg.eigh(fock[k])
            new_mo_coeff[k] = numpy.dot(mo_coeff[k], U)

            # Sắp xếp lại theo thứ tự năng lượng
            mo_energy[k] = new_mo_energy[k][sort_idx[k]].real
            mo_coeff[k] = new_mo_coeff[k][:,sort_idx[k]]
            
            
            #mp.mo_energy = mo_energy
            #mp.mo_coeff  = mo_coeff
        print("=====ip and ea=====")
        #IP, EA = make_IPEA(mp, tmp1_bar, h2mo_ovov)    
        #IP = IP 
        #EA = EA
        print("IP = ", IP - mo_energy[0][nocc-1] ) # mo_energy[0][nocc-1] chính là \varepsilon_{\text{HOMO}}^{\text{HF}}
        print("EA = ", EA - mo_energy[0][nocc] ) # mo_energy[0][nocc] chính là \varepsilon_{\text{LUMO}}^{\text{HF}}
        IP = IP - mo_energy[0][nocc-1]
        EA = EA - mo_energy[0][nocc]
    IP1 = -1*(IP - mo_energy[0][nocc-1])
    EA1 = -1*(EA - mo_energy[0][nocc])
    print(f"(Lần 2) IP = {IP} (Hatree), {IP*27.2114} (eV)| IP1 = {IP1} (Hatree), {IP1*27.2114} (eV)")   #27.2114
    print(f"(Lần 2) EA = {EA} (Hatree), {EA*27.2114} (eV)| EA1 = {EA1} (Hatree), {EA1*27.2114} (eV)")
    print("mo_energy = ")
    print(mo_energy)
    print("ene_tot (eV)= ")
    print(ene_tot*27.2114)
    return ene_tot, mo_energy, IP, EA # Năng lượng tổng và năng lượng orbital

'''
Lưu đồ thuật toán của hàm kernel
┌──────────────────────────┐
│  Khởi tạo các thông số   │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│Sắp xếp orbital theo năng │
│lượng (sort_idx)          │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│     Bắt đầu vòng lặp     │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│ Tính Hamiltonian và      │
│ potential hiệu dụng      │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│ Xây dựng ma trận Fock HF │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│Tính năng lượng Hartree-  │
│Fock (ene_hf)             │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│  Thực hiện BCH bậc 1     │
│  Cập nhật ma trận Fock   │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│Tính năng lượng tổng      │
│(ene_tot)                 │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│   Kiểm tra hội tụ        │
│   de < thresh?           │
└───────────┬──────────────┘
            │
      ┌─────┴─────┐
      │           │
      ▼           ▼
┌──────────┐ ┌────────────────┐
│   Đúng   │ │     Sai        │
│   (break)│ │                │
└──────────┘ └────────┬───────┘
                      │
                      ▼
             ┌──────────────────┐
             │Chéo hóa ma trận  │
             │Fock & cập nhật   │
             │orbital           │
             └─────────┬────────┘
                       │
                       ▼
             ┌──────────────────┐
             │Quay lại vòng lặp │
             └──────────────────┘

'''

#################################################################################################################


def make_veff(mp, mo_coeff, mo_energy):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = numpy.shape(mo_energy)[0]

    kpts = mp.kpts
    dm = mp._scf.make_rdm1()
    veff_ao = mp._scf.get_veff(mp._scf.cell, dm)


    veff = numpy.zeros((nkpts, nmo,nmo), dtype=complex)
    
    for kp in range(nkpts):
        veff[kp] = numpy.matmul(mo_coeff[kp].T.conj(),numpy.matmul(veff_ao[kp], mo_coeff[kp]))
    
    c0_hf = 0
    for kp in range(nkpts):
        for i in range(nocc):
            c0_hf -=  veff[kp][i,i].real
    c0_hf/= nkpts
    
    return veff_ao, veff, c0_hf


def ene_denom(mp, mo_energy, ki, ka, kj, kb):

    '''
    Phần này tương tự như trong code KMP2, tuy nhiên ở chỗ này khác cái là nó tách ra
    phần tính e_ij^ab ra 1 hàm riêng.
    '''
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = numpy.shape(mo_energy)[0]

    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")
    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]


    eia = LARGE_DENOM * numpy.ones((nocc, nvir), dtype=mo_energy[0].dtype)
    n0_ovp_ia = numpy.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
    #e_i (chiếm đóng) - e_a (Ảo)
    eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia] 


    ejb = LARGE_DENOM * numpy.ones((nocc, nvir), dtype=mo_energy[0].dtype)
    n0_ovp_jb = numpy.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
    #e_j (chiếm đóng) - e_b (Ảo)
    ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb] 

    # e_iajb = e_i (Chiếm đóng) - e_a (Ảo) + e_j (Chiếm đóng) - e_b (Ảo)
    e_iajb = lib.direct_sum('ia,jb -> iajb', eia, ejb) 
    
    ejh = LARGE_DENOM * numpy.ones((nocc, nocc), dtype=mo_energy[0].dtype)
    n0_ovp_jh = numpy.ix_(nonzero_opadding[kj], nonzero_opadding[kb])
    # e_j (Chiếm đóng) - e_h (Chiếm đóng)
    ejh[n0_ovp_jh] = (mo_e_o[kj][:,None] - mo_e_o[kb])[n0_ovp_jh] 

    # e_iajh = e_i (Chiếm đóng) - e_a (Ảo) + e_j (Chiếm đóng) - e_h (Chiếm đóng)
    e_iajh = lib.direct_sum('ia,jh -> iajh', eia, ejh)

    elb = LARGE_DENOM * numpy.ones((nvir, nvir), dtype=mo_energy[0].dtype)
    """
    n0_ovp_lb = numpy.ix_(nonzero_opadding[kj], nonzero_opadding[kb])
    # e_l (Ảo) - e_b (Ảo)

    print("10) mo_e_v[kj][:,None] = ")
    print(mo_e_v[kj][:,None])
    print("11) mo_e_v[kb] = ")
    print(mo_e_v[kb])
    print("12) mo_e_v[kj][:,None] - mo_e_v[kb] = ")
    print(mo_e_v[kj][:,None] - mo_e_v[kb])
    print("13) 0_ovp_lb = ")
    print(n0_ovp_lb)

    elb[n0_ovp_lb] = (mo_e_v[kj][:,None] - mo_e_v[kb])[n0_ovp_lb]
    """

    '''
    elb = LARGE_DENOM * numpy.ones((nvir, nvir), dtype=mo_energy[0].dtype)
    n0_ovp_lb = numpy.ix_(nonzero_opadding[kj], nonzero_opadding[kb])
    # e_l (Ảo) - e_b (Ảo)
    elb[n0_ovp_lb] = (mo_e_v[kj][:,None] - mo_e_v[kb])[n0_ovp_lb]
    '''

    # Sửa đoạn code tạo elb thành:
    elb = LARGE_DENOM * numpy.ones((nvir, nvir), dtype=mo_energy[0].dtype)

    # Kiểm tra kích thước mảng trước khi gán
    if len(mo_e_v[kj]) > 0 and len(mo_e_v[kb]) > 0:
        n0_ovp_lb = numpy.ix_(nonzero_opadding[kj], nonzero_opadding[kb])

        # Tạo mask chỉ mục hợp lệ
        valid_indices = (
            (nonzero_opadding[kj] < len(mo_e_v[kj])) & 
            (nonzero_opadding[kb] < len(mo_e_v[kb]))
        )

        # Chỉ gán giá trị cho các chỉ mục hợp lệ
        if numpy.any(valid_indices):
            valid_kj = nonzero_opadding[kj][valid_indices]
            valid_kb = nonzero_opadding[kb][valid_indices]
            valid_ovp = numpy.ix_(valid_kj, valid_kb)

            elb[valid_ovp] = (mo_e_v[kj][valid_kj, None] - mo_e_v[kb][valid_kb])


    e_ialb = lib.direct_sum('ia,lb -> ialb', eia, elb)

    
    # e_ialb = e_i (Chiếm đóng) - e_a (Ảo) + e_l (Ảo) - e_b (Ảo)

    return e_iajb, e_iajh, e_ialb
"""
def make_amp(mp, mo_energy, mo_coeff):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = numpy.shape(mo_energy)[0]
    kpts = mp.kpts
    kconserv = mp.khelper.kconserv
    fao2mo = mp._scf.with_df.ao2mo

    tmp1 = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=complex)
    tmp1_bar = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=complex)
    h2mo_ovgg = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nmo,nmo), dtype=complex)
    h2mo_ovov = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=complex)
    
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                kp = kj
                kq = kb
                o_i = mo_coeff[ki][:,:nocc]
                o_a = mo_coeff[ka][:,nocc:]
                o_p = mo_coeff[kp]
                o_q = mo_coeff[kq]
                h2mo_ovgg[ki,kj,ka] = fao2mo((o_i,o_a,o_p,o_q),
                                (kpts[ki],kpts[ka],kpts[kp],kpts[kq]),
                                compact=False).reshape(nocc,nvir,nmo,nmo)/nkpts
                h2mo_ovov[ki,kj,ka] = h2mo_ovgg[ki,kj,ka][:,:,:nocc, nocc:]
                
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                e_iajb = ene_denom(mp, mo_energy, ki, ka, kj, kb)
                w_iajb = (h2mo_ovov[ki,kj,ka]
                            -0.5*h2mo_ovov[ki,kj,kb].transpose(0,3,2,1))
                
                tmp1[ki,kj,ka]  = (h2mo_ovov[ki,kj,ka]/e_iajb).conj()
                tmp1_bar[ki,kj,ka]  =  (w_iajb/e_iajb).conj()
                
    return tmp1, tmp1_bar, h2mo_ovgg
"""
def first_BCH(mp, mo_energy, mo_coeff, fock_hf):

    star = time.time()

    '''
    Ma trận fock_hf có kích thước (4, 8, 8), nghĩa là đây là một ma trận 3
    chiều với kích thước 4x8x8 (nkpts, nmo, nmo). 

    Dưới đây là phần tử [0, :, :] trong ma trận fock_hf (có kích thước 8x8),
    như ta có thể thấy các phần tử trong ma trận này là phức.

    Này chính là f_i^a

 [[-2.751222e-01+3.500915e-17j -3.253421e-17+4.797473e-18j
   -1.850400e-08+1.056231e-17j -1.043138e-16-4.760173e-18j
   -1.848313e-15-6.968306e-18j -8.612572e-17+1.028306e-18j
    1.042214e-12+6.286847e-18j  1.850165e-11+1.312497e-16j]
  [-1.463318e-17-3.635312e-18j  6.843137e-01+4.570390e-18j
   -6.135213e-16+1.240201e-18j  2.145551e-16+1.105647e-17j
    1.546130e-16-2.260957e-17j  2.517731e-11-4.191427e-17j
   -1.219384e-15+5.342138e-18j -6.325740e-18+2.030173e-17j]
  [-1.850400e-08-3.218890e-18j -5.863493e-16-3.318261e-18j
    7.232611e-01+1.431287e-17j -3.011646e-16-7.821057e-19j
    1.352190e-15-2.513979e-18j -1.856054e-15-8.717872e-19j
    2.651066e-11+1.248211e-17j -1.325065e-12+3.058273e-17j]
  [-1.384110e-16+1.710862e-18j  1.807482e-16-2.546279e-17j
   -3.058997e-16-3.102388e-18j  7.412138e-01+2.103509e-17j
   -2.731608e-11+5.598378e-16j  1.074894e-15-2.897670e-18j
    1.474482e-15-7.570065e-18j  1.224105e-15+2.166894e-18j]
  [-1.819616e-15+5.357939e-18j  1.522031e-16+2.946397e-17j
    1.362376e-15+5.635221e-18j -2.731612e-11-5.485700e-16j
    1.019366e+00+1.392856e-17j -1.051474e-15-5.038091e-18j
   -3.889311e-16+7.904079e-18j -1.626720e-16-3.271464e-18j]
  [-8.432795e-17-2.872320e-18j  2.517739e-11+2.799377e-17j
   -1.871916e-15+2.350313e-19j  1.032914e-15-8.546683e-19j
   -1.170721e-15+1.194635e-18j  1.022002e+00+2.070653e-17j
    3.396042e-16+1.188523e-17j  1.233670e-16-2.560300e-17j]
  [ 1.042237e-12-1.997678e-17j -1.276911e-15-5.795486e-18j
    2.651063e-11-2.769556e-17j  1.516157e-15+1.949115e-18j
   -3.982842e-16-2.287833e-18j  3.890885e-16-3.332690e-18j
    1.052164e+00+1.608084e-17j -4.603489e-08-3.519927e-17j]
  [ 1.850179e-11-1.205531e-16j  7.704884e-17-2.486689e-17j
   -1.325082e-12-2.556691e-17j  1.452113e-15-3.830399e-19j
   -1.927432e-16+2.918121e-17j  7.420728e-17+3.235770e-17j
   -4.603489e-08+2.418232e-17j  1.422188e+00+4.389906e-18j]]
    '''

    nmo = mp.nmo # Số lượng các Orbital phân tử (lấp đầy + trống)
    nocc = mp.nocc # Số lượng các Orbital phân tử đã lấp đầy
    nvir = nmo - nocc # Số lượng các Orbital phân tử trống
    nkpts = numpy.shape(mo_energy)[0] # Cho số lượng các k - point trong tính toán
    kd = mp.kpts
    kpts = kd.kpts # Mảng chứa các k - point trong không gian sóng của phân tử hoặc tinh thể


    print('4. nmo =')
    print(nmo)
    print('5. nocc =')
    print(nocc)
    print('6. nvir =')
    print(nvir)
    print('7. nkpts =')
    print(nkpts)
    print('8. kpts =')
    print(kpts)

    '''
    4. nmo = 8
    5. nocc = 4
    6. nvir = 4
    7. nkpts = 4

    Ma trận kpts có kích thước 4x3 (Ý nghĩa ma trận này là gì thì tính sau)
    8. kpts =
        [[ 0.        0.        0.      ]
         [ 0.466093  0.466093 -0.466093]
         [-0.466093  0.466093  0.466093]
         [ 0.        0.932185  0.      ]]

    '''

    #kconserv = kpts_helper.get_kconserv(mp._scf.cell, kpts) # Sẽ bỏ qua.

    # Đây là hàm mà khi ta truyền vào 3 tham số là ki, kj, ka thì nó sẽ trả về ka 
    # sao cho bảo toàn momen động lượng

    fao2mo = mp._scf.with_df.ao2mo  # Giữ nguyên

    # Đây là hàm giúp tính g^ij_ab từ cơ sở AO sang
    # cơ sở MO, các cái khi chính là cơ sở AO (và nó phức)
    
    '''
    Ma trận `tmp1` và `tmp1_bar`:
        - `tmp1` và `tmp1_bar`: Có hình dạng `(nkpts, nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir)`
        - `tmp1` tương ứng với T_ij^ab
        - `tmp1_bar` tương ứng với $\bar{T}_{ij}^{ab} = T_{ij}^{ab} - T_{ji}^{ab}/2$.
    '''
    tmp1 = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=complex)
    tmp1_bar = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=complex)
    #tmp1_iajh = numpy.zeros((nkpts,nocc,nvir,nocc,nocc), dtype=complex)
    #tmp1_ialb = numpy.zeros((nkpts,nocc,nvir,nvir,nvir), dtype=complex)
    tmp1_bar_ialb = numpy.zeros((nocc,nvir,nvir,nvir), dtype=complex)
    tmp1_bar_iajh = numpy.zeros((nocc,nvir,nocc,nocc), dtype=complex)
    
    '''
    - `h2mo_ovgg`, `h2mo_ovov`, `h2mo_ovog`, `h2mo_ovgv`, `h2mo_ovoo`, `h2mo_ovvv`
      chính là các two-electron integrals trong cơ sở MO. 

    - `h2mo_ovgg` chính là $g^{ab}_{ij}$, còn `h2mo_ovov`, `h2mo_ovog`, `h2mo_ovgv`, 
      `h2mo_ovoo`, `h2mo_ovvv` là các biến thể khác nhau (ma trận con của `h2mo_ovgg`) 
      được tách ra để dễ tính toán. 
    '''
    h2mo_ovgg = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nmo,nmo), dtype=complex)
    h2mo_ovov = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=complex)
    h2mo_ovog = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nmo), dtype=complex)
    h2mo_ovgv = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nmo,nvir), dtype=complex)
    h2mo_ovoo = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), dtype=complex)
    h2mo_ovvv = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nvir,nvir), dtype=complex)

    c1 = numpy.zeros((nkpts,nkpts,nmo,nmo), dtype=complex)
    c0_1st = 0
    
    print('#################################################')

    dummy_tmp1 = numpy.empty((nocc,nvir,nocc,nvir), dtype=complex)
    dummy_tmp1_bar = numpy.empty((nocc,nvir,nocc,nvir), dtype=complex)
    dummy_fock_hf = numpy.empty((nmo,nmo), dtype=complex)

    # 4 câu lệnh tiếp theo là mới, nhằm xử lý các điểm k có trọng số.
    kijab, weight, k4_bz2ibz = kd.make_k4_ibz(sym='s2') 
    _, igroup = numpy.unique(kijab[:,:2], axis=0, return_index=True) # Mới
    igroup = igroup.ravel() # Mới
    igroup = list(igroup) + [len(kijab)] # Mới
    nao2mo = 0 # Mới
    icount = 0 # Mới

    IP = 0
    EA = 0

    '''
    mem_now = lib.current_memory()[0]
    max_memory1 = cc.max_memory
    print("max_memory ", max_memory1, " (MB)")
    max_memory = max(0, max_memory1 - mem_now)
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
    '''

    for i in range(len(igroup)-1):
        istart = igroup[i]
        iend = igroup[i+1]
        kab = []
        print(f"Nhóm thứ {i}, từ {istart} đến {iend}")
        for j in range(istart, iend):
            a, b = kijab[j][2:]
            kab.append([a, b])
            kab.append([b, a])
        kab = numpy.unique(numpy.asarray(kab), axis=0)
        print("kab = ")
        print(kab)

        ki = kijab[istart][0]
        kj = kijab[istart][1]
        print(f"ki = {ki}, kj = {kj}")
        o_i = mo_coeff[ki][:,:nocc]
        kp = kj
        o_p = mo_coeff[kp]

        for (ka, kb) in kab:
            print(f"ka = {ka}, kb = {kb}")
            o_a = mo_coeff[ka][:,nocc:]
            kq = kb
            o_q = mo_coeff[kq]

            h2mo_ovgg[ki,ka,kj]= fao2mo((o_i,o_a,o_p,o_q),
                                (kpts[ki],kpts[ka],kpts[kp],kpts[kq]),
                                compact=False).reshape(nocc,nvir,nmo,nmo)/nkpts
                
            '''
            Đây là sự khác biệt giữa code KOBMP2 với code KMP2, 
            ở code KOBMP2 thì h2mo_ovgg[ki,ka,kj,kb] còn có chứa nhiều tích phân 2 e với các chỉ số khác nhau nữa,
            chứ không chỉ là tích phân 2 e với chỉ số [i], [a], [j], [b] như ở code KMP2.
            => Đây là lý do có đoạn code cắt ở dưới đây.
            '''
            # h2mo_ovov chính là tích phân 2 electron oovv_ij[ka][i][j][a][b] trong code KMP2
            h2mo_ovov[ki,ka,kj] = h2mo_ovgg[ki,ka,kj][:,:,:nocc, nocc:] 
            '''
            14. h2mo_ovov shape: (4, 4, 4, 4, 4, 4, 4, 4) # Đúng như được kỳ vọng.
            '''
            # 2mo_ovgv, h2mo_ovog, h2mo_ovvv, h2mo_ovoo là các g khác được dùng để tính IP, EA.
            h2mo_ovgv[ki,ka,kj] = h2mo_ovgg[ki,ka,kj][:,:,:, nocc:]
            h2mo_ovog[ki,ka,kj] = h2mo_ovgg[ki,ka,kj][:,:,:nocc, :]
            h2mo_ovvv[ki,ka,kj] = h2mo_ovgg[ki,ka,kj][:,:,nocc:,nocc:]
            h2mo_ovoo[ki,ka,kj] = h2mo_ovgg[ki,ka,kj][:,:,:nocc,:nocc]
            '''
                NHẬN XÉT: Trong vòng lặp đầu này thì mục tiêu chính là đi tính tích phân 2 electron,
                tương đồng với code KMP2. Tuy nhiên code KOBMP2 tính nhiều trường hợp tích phân 2 hơn
                và lưu kết quả tính đó trong 1 ma trận có nhiều chỉ số hơn. 
                Lý do mà code KOBMP2 phải tính nhiều tích phân 2 e hơn là để tính IP, EA.
                => Giảm bớt các biến lưu, thay vì lưu h2mo_ovgv[ki,ka,kj,kb] thì chuyển thành h2mo_ovgv[ka]
            '''
            nao2mo += 1 # Đếm số lần biến đổi AO -> MO (mới)

    ki_1 = -1
    kj_1 = -1

    dummy_h2mo_ovgv = numpy.empty((nocc, nvir, nmo, nvir), dtype=complex)
    path_pj = oe.contract_path('iajb, iapb -> pj', dummy_tmp1_bar, dummy_h2mo_ovgv, optimize='greedy')[0]
    
    dummy_h2mo_ovog = numpy.empty((nocc, nvir, nocc, nmo), dtype=complex)
    path_bp = oe.contract_path('iajb, iajp -> bp', dummy_tmp1_bar, dummy_h2mo_ovog, optimize='greedy')[0]
    
    path_bj = oe.contract_path('ai, iajb -> bj', dummy_fock_hf[nocc:,:nocc], dummy_tmp1_bar, optimize='greedy')[0]
    
    dummy_h2mo_ovov = numpy.empty((nocc, nvir, nocc, nvir), dtype=complex)
    path_sum_3 = oe.contract_path('iajb, iajb ->', dummy_tmp1_bar, dummy_h2mo_ovov, optimize='greedy')[0]

    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                print(f"18. Vòng lặp sau: ki = {ki}, kj = {kj}, ka = {ka}, kb = {kb}")
                '''
                18. Vòng lặp sau: ki = 0, kj = 0, ka = 0, kb = 0
                '''
                '''
                - Hàm ene_denom(mp, mo_energy, ki, ka, kj, kb) chính là phần đi tính e_iajb,
                e_iajh, e_ialb, thuật toán tương tự như trong code KMP2, tuy nhiên ở đây thì họ
                tách ra 1 hàm riêng. 
                
                - Trong hàm ene_denom không có dụng tới các biến h2mo.
                '''
                e_iajb, e_iajh, e_ialb = ene_denom(mp, mo_energy, ki, ka, kj, kb)

                

                '''
                Tương tự code KMP2

                Ma trận e_iajb là một ma trận 4x4x4x4 (nocc, nvir, nocc, nvir)
                Ma trận e_iajh là một ma trận 4x4x4x4 (nocc, nvir, nocc, nocc)
                Ma trận e_ialb là một ma trận 4x4x4x4 (nocc, nvir, nvir, nvir)
                '''

                # Câu lệnh này cần phải xem lại coi nó tính như thế nào! để biết mà bỏ vòng lặp for
                w_iajb = (h2mo_ovov[ki,ka,kj]
                            -0.5*h2mo_ovov[ki,kb,kj].transpose(0,3,2,1))
                
                '''
                Phần này mới so với code KMP2,
                ma trận w_iajb kích thước (nocc, nvir, nocc, nvir), giống với kích thước
                ma trận h2mo_ovov, tuy nhiên thứ tự các chỉ số sẽ khác (CẦN XEM LẠI)
                '''

                tmp1[ki,ka,kj]  = (h2mo_ovov[ki,ka,kj]/e_iajb).conj()

                '''
                Phần tính tmp1[ki,ka,kj,kb] tương đương với phần tính t2_ijab trong code KMP2.
                Xét trong BCH 1 thì tmp1[ki,ka,kj,kb] = T_ij^ab
                '''

                tmp1_bar[ki,ka,kj]  =  (w_iajb/e_iajb).conj()
                '''
                Xét trong BCH 1 thì tmp1_bar[ki,ka,kj,kb] = bar{T_ij^ab}
                '''
                # Debugging tmp1 and tmp1_bar
                '''
                 tmp1 shape: (4, 4, 4, 4, 4, 4, 4, 4), tmp1_bar shape: (4, 4, 4, 4, 4, 4, 4, 4)
                '''

                w_iajh = (h2mo_ovoo[ki,ka,kj]
                            -0.5*h2mo_ovoo[kj,ka,ki].transpose(2,1,0,3))
                '''
                w_iajh được dùng để tính bar{T_ij^ah}
                '''

                w_ialb = (h2mo_ovvv[ki,ka,kj]
                            -0.5*h2mo_ovvv[ki,kb,kj].transpose(0,3,2,1))
                '''
                w_ialb được dùng để tính bar{T_il^ab}
                '''
                '''
                Xét trong BCH 1 thì tmp1_iajh[ki,ka,kj,kb] = T_ij^ah
                '''
        
                tmp1_bar_iajh  =  (w_iajh/e_iajh).conj()
                '''
                Xét trong BCH 1 thì tmp1_bar_iajh[ki,ka,kj,kb] = bar{T_ij^ah}
                '''
                '''
                Xét trong BCH 1 thì tmp1_ialb[ki,ka,kj,kb] = T_il^ab
                '''
                tmp1_bar_ialb  =  (w_ialb/e_ialb).conj()
                '''
                Xét trong BCH 1 thì tmp1_bar_ialb[ki,ka,kj,kb] = bar{T_il^ab}
                '''

                '''
                CHÚ Ý: Sau cải tiến thì T, bar{T} bị bỏ đi biến kb (vì kb phụ thuộc vào ki, kj, ka). 
                Còn h2mo được loại bỏ đi biến ki, kj, kb (chỉ giữ lại ka) vì c1, c0 sẽ được tính trong 3 vòng lặp for luôn.
                
                Sau vòng lặp trong, mình đã thu được các đại lượng sau:
                - T_ij^ab, bar{T_ij^ab}
                - T_ij^ah, bar{T_ij^ah}
                - T_il^ab, bar{T_il^ab}

                Các đại lượng này sẽ được dùng để tính IP, EA.
                IP = -e_HOMO + 2*bar{T_ij^ah}*h2mo_ovoo = -e_HOMO + 2*bar{T_ij^ah}*{g_ah^ij}
                EA = -e_LUMO - 2*bar{T_il^ab}*h2mo_ovvv = -e_LUMO - 2*bar{T_il^ab}*{g_ab^il}

                '''

                mp.ampf = 0.5
                tmp1[ki,ka,kj] *= mp.ampf
                tmp1_bar[ki,ka,kj] *= mp.ampf
                tmp1_bar_iajh *= mp.ampf
                tmp1_bar_ialb *= mp.ampf
                
                if kb == 0:
                    IP += 2*numpy.einsum('iaj, iaj ->',tmp1_bar_iajh[:,:,:,nocc-1],h2mo_ovoo[ki,ka,kj,:,:,:,nocc-1]).real
                # IP = 2*numpy.einsum('qwriaj, qwriaj',tmp1_bar_iajh[:,:,:,0,:,:,:,nocc-1],h2mo_ovoo[:,:,:,0,:,:,:,nocc-1]).real

                if kj == 0:
                    EA += -2*numpy.einsum('iab, iab ->',tmp1_bar_ialb[:,:,0,:],h2mo_ovvv[ki,ka,0,:,:,0,:]).real
                # EA = -2*numpy.einsum('qwriab, qwriab',tmp1_bar_ialb[:,:,0,:,:,:,0,:],h2mo_ovvv[:,:,0,:,:,:,0,:]).real

                c1[kj,kj,:,:nocc] += 2*oe.contract('iajb, iapb -> pj',tmp1_bar[ki,ka,kj,:,:,:,:],h2mo_ovgv[ki,ka,kj,:,:,:,:], optimize=path_pj)
                #c1[:,:,:,:nocc] += 2*numpy.einsum('qweriajb, qwtriapb -> tepj',tmp1_bar,h2mo_ovgv)

                c1[kb,kb,nocc:,:] -= 2*oe.contract('iajb, iajp -> bp',tmp1_bar[ki,ka,kj,:,:,:,:],h2mo_ovog[ki,ka,kj,:,:,:,:], optimize=path_bp)  
                #c1[:,:,nocc:,:] -= 2*numpy.einsum('qwetiajb, qweriajp -> trbp',tmp1_bar,h2mo_ovog)  

                if ki != ki_1 or kj != kj_1:
                    ki_1 = ki
                    kj_1 = kj
                    r = kconserv[ki,ki,kj]
                    c1[r,kj,nocc:,:nocc] += 2*oe.contract('ai, iajb -> bj',fock_hf[ki,nocc:,:nocc].conj(),tmp1_bar[ki,ki,kj,:,:,:,:], optimize=path_bj)
                    #c1[:,:,nocc:,:nocc] += 2*numpy.einsum('qai, qqeriajb -> rebj',fock_hf[:,nocc:,:nocc].conj(),tmp1_bar)

                c0_1st += -4*oe.contract('iajb, iajb ->',tmp1_bar[ki,ka,kj,:,:,:,:], h2mo_ovov[ki,ka,kj,:,:,:,:], optimize = path_sum_3)
                #c0_1st = -4*numpy.einsum('qweriajb, qweriajb',tmp1_bar, h2mo_ovov)
                
    print(f"IP: {IP}, EA: {EA}")  # Debugging IP and EA

    print(f"Running time {time.time() - star} seconds")  # Debugging running time

    c1 = numpy.einsum('wwpq -> wpq', c1) * weight[idx_ibz] * nkpts**3


    c0_1st = c0_1st.real/nkpts


    c0 = c0_1st
    c0_2nd = 0.0
    c2 = numpy.zeros((nkpts, nkpts, nmo,nmo), dtype=complex)

    
    if mp.second_order:

        ###################################################################################################################################

        # Tính toán y1 với kconserv (CHÍNH XÁC)

        y1 = numpy.zeros((nkpts, nkpts, nocc, nvir), dtype=complex)
        path_jb = oe.contract_path('ai, iajb -> jb', dummy_fock_hf[nocc:,:nocc], dummy_tmp1_bar, optimize='greedy')[0]

        for ki in range(nkpts):  # q ~ ki
            for kj in range(nkpts):  # e ~ kj
                kb = kconserv[ki, ki, kj]  # r ~ kb = kconserv(ki, ka=ki, kj)
                #y1[kj, kb] += 2 * numpy.einsum('ai,iajb->jb', fock_hf_slide[ki].conj(), tmp1_bar[ki, ki, kj, kb])
                y1[kj, kb, :, :] += 2 * oe.contract('ai, iajb -> jb',fock_hf[ki,nocc:,:nocc].conj(),tmp1_bar[ki, ki, kj, :, :, :, :], optimize=path_jb)
                #y1 = 2*numpy.einsum('qai, qqeriajb -> erjb',fock_hf[:,nocc:,:nocc].conj(),tmp1_bar)

        c2 = numpy.zeros((nkpts, nkpts, nmo,nmo), dtype=complex)

        dummy_y1 = numpy.empty((nocc, nvir), dtype=complex)
        path_ck = oe.contract_path('jb, jbkc -> ck', dummy_y1, dummy_tmp1_bar, optimize='greedy')[0]

        for ki in range(nkpts):     # q ứng với ki (nkpts)
            for kj in range(nkpts):     # e ứng với kj (nkpts)
                for kb in range(nkpts): # r ứng với kb (nkpts)
                    t = kconserv[kj, kb, ki] # t ~ ka = kconserv(kj, kb, ki)
                    c2[t,ki,nocc:nocc+nvir,:nocc] += oe.contract('jb, jbkc -> ck', y1[kj, kb, :, :], tmp1_bar[kj, kb, ki, :, :, :, :].conj(), optimize=path_ck)
                    #c2[:,:,nocc:,:nocc] = numpy.einsum('erjb, erqtjbkc -> tqck', y1, tmp1_bar.conj())



        ###################################################################################################################################


        # Đã chạy chính xác:
        y1 = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=complex)
        path_iakb = oe.contract_path('ca, ickb -> iakb', dummy_fock_hf[nocc:,nocc:], dummy_tmp1_bar, optimize='greedy')[0]


        for ki in range(nkpts):    
            for ka in range(nkpts):     
                for kb in range(nkpts): 
                    #t = kconserv[ka, ki, kb] 
                    y1[ka,ki,kb,:,:,:,:] += oe.contract('ca, ickb -> iakb', fock_hf[ki,nocc:,nocc:], tmp1_bar[ka,ki,kb,:,:,:,:].conj(), optimize=path_iakb)
                    #y1 = numpy.einsum('qca, wqrtickb -> wqrtiakb', fock_hf[:,nocc:,nocc:], tmp1_bar.conj())



        ###################################################################################################################################

        c0_2nd =0

        dummy_y1 = numpy.empty((nocc, nvir, nocc, nvir), dtype=complex)
        path_lj = oe.contract_path('ialb,iajb->lj', dummy_y1, dummy_tmp1, optimize='greedy')[0]
        path_ki = oe.contract_path('kajb,iajb->ki', dummy_y1, dummy_tmp1, optimize='greedy')[0]
        path_bd = oe.contract_path('iajd,iajb->bd', dummy_y1, dummy_tmp1, optimize='greedy')[0]
        path_sum = oe.contract_path('iajb,iajb->', dummy_y1, dummy_tmp1, optimize='greedy')[0]

        for w in range(nkpts):
            for q in range(nkpts):
                for t in range(nkpts):
                    # Xác định chỉ số r dựa trên kconserv
                    e = kconserv[w, q, t]
                    c2[e,e,:nocc,:nocc] += oe.contract('ialb, iajb -> lj', y1[w, q, e,:,:,:,:], tmp1[w, q, e,:,:,:,:], optimize=path_lj)
                    #c2[:,:,:nocc,:nocc] += numpy.einsum('wqrtialb, wqetiajb -> relj', y1, tmp1)

                    c2[w,w,:nocc,:nocc] += oe.contract('kajb, iajb -> ki', y1[w, q, e,:,:,:,:], tmp1[w, q, e,:,:,:,:], optimize=path_ki)
                    #c2[:,:,:nocc,:nocc] += numpy.einsum('wqrtkajb, eqrtiajb -> weki', y1, tmp1)

                    c2[t,t,nocc:nocc+nvir,nocc:nocc+nvir] -= oe.contract('iajd, iajb -> bd', y1[w, q, e,:,:,:,:], tmp1[w, q, e,:,:,:,:], optimize=path_bd)
                    #c2[:,:,nocc:,nocc:] -= numpy.einsum('wqrtiajd, wqreiajb -> etbd', y1, tmp1)

                    c0_2nd -= 4*oe.contract('iajb,iajb ->', y1[w, q, e,:,:,:,:], tmp1[w, q, e,:,:,:,:], optimize=path_sum).real
                    #c0_2nd -= 4*numpy.einsum('qweriajb,qweriajb ->', y1, tmp1).real


        ##############################################################Thứ 3##################################################################

        y1 = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=complex) 
        path_ialb = oe.contract_path('ki, kalb -> ialb', dummy_fock_hf[:nocc, :nocc], dummy_tmp1_bar, optimize='greedy')[0]

        # Tính chính xác
        for ki in range(nkpts):
            for ka in range(nkpts):
                for kb in range(nkpts):
                    #kj = kconserv[ki,ka,kb]
                    y1[ki,ka,kb,:,:,:,:] += oe.contract('ki, kalb -> ialb', fock_hf[ki,:nocc, :nocc], tmp1_bar[ki,ka,kb,:,:,:,:].conj(), optimize=path_ialb)
                    #y1 = numpy.einsum('qki, qwrtkalb -> qwrtialb', fock_hf[:,:nocc, :nocc], tmp1_bar.conj())

        dummy_y1 = numpy.empty((nocc, nvir, nocc, nvir), dtype=complex)
        path_lj = oe.contract_path('ialb, iajb -> lj', dummy_y1, dummy_tmp1, optimize='greedy')[0]
        path_ac = oe.contract_path('icjb, iajb -> ac', dummy_y1, dummy_tmp1, optimize='greedy')[0]
        path_bd = oe.contract_path('iajd, iajb -> bd', dummy_y1, dummy_tmp1, optimize='greedy')[0]
        path_sum = oe.contract_path('iajb, iajb ->', dummy_y1, dummy_tmp1, optimize='greedy')[0]

        for w in range(nkpts):
            for q in range(nkpts):
                for t in range(nkpts):
                    # Xác định chỉ số r dựa trên kconserv
                    r = kconserv[q, w, t]
                    c2[r,r,:nocc,:nocc] -= oe.contract('ialb, iajb -> lj', y1[q, w, r,:,:,:,:], tmp1[q, w, r,:,:,:,:], optimize=path_lj)
                    #c2[:,:,:nocc,:nocc] -= numpy.einsum('qwrtialb, qwetiajb -> relj', y1, tmp1)
                    c2[w,w,nocc:nocc+nvir,nocc:nocc+nvir] += oe.contract('icjb, iajb -> ac', y1[q, w, r,:,:,:,:],
                                                                                 tmp1[q, w, r,:,:,:,:], optimize=path_ac)
                    #c2[:,:,nocc:,nocc:] += numpy.einsum('qwrticjb, qertiajb -> ewac', y1, tmp1)
                    c2[t,t,nocc:nocc+nvir,nocc:nocc+nvir] += oe.contract('iajd, iajb -> bd', y1[q, w, r,:,:,:,:], tmp1[q, w, r,:,:,:,:], optimize=path_bd)
                    #c2[:,:,nocc:,nocc:] += numpy.einsum('qwrtiajd, qwreiajb -> etbd', y1, tmp1)
                    c0_2nd += 4*oe.contract('iajb, iajb ->', y1[q, w, r,:,:,:,:], tmp1[q, w, r,:,:,:,:], optimize=path_sum).real
                    #c0_2nd += 4*numpy.einsum('qwrtiajb, qwrtiajb ->', y1, tmp1).real

        c0_2nd /= nkpts



        ###################################################################################################################################



        ###########################################################Thứ 4####################################################################


        y1 = numpy.zeros((nkpts,nkpts,nocc,nocc), dtype=complex)

        path_ki = oe.contract_path('iajb, kajb -> ki', dummy_tmp1, dummy_tmp1_bar, optimize='greedy')[0]

        # Chạy chính xác

        for w in range(nkpts):
            for r in range(nkpts):
                for t in range(nkpts):
                    # Xác định chỉ số r dựa trên kconserv
                    q = kconserv[w, r, t]
                    y1[q,q,:,:] += oe.contract('iajb, kajb -> ki', tmp1[q, w, r,:,:,:,:], tmp1_bar.conj()[q, w, r,:,:,:,:], optimize=path_ki)
                    #y1 = numpy.einsum('qwrtiajb, ewrtkajb -> eqki', tmp1, tmp1_bar.conj())

        for e in range(nkpts):
            for q in range(nkpts):
                c2[q,e,:,:nocc] -= lib.einsum('pi, ki -> pk', fock_hf[q,:,:nocc], y1[e,q,:,:])
                #c2[:,:,:,:nocc] -= numpy.einsum('qpi, eqki -> qepk', fock_hf[:,:,:nocc], y1)



        ###################################################################################################################################



        ###########################################################Thứ 5####################################################################

        ### [9] f_ap * T_iajb * Tb_icjb -> E_cp

        y1 = numpy.zeros((nkpts,nkpts,nvir,nvir), dtype=complex)

        path_ac = oe.contract_path('iajb, icjb -> ac', dummy_tmp1, dummy_tmp1_bar, optimize='greedy')[0]

        for q in range(nkpts):
            for r in range(nkpts):
                for t in range(nkpts):
                    # Xác định chỉ số r dựa trên kconserv
                    w = kconserv[q, r, t]
                    y1[w,w,:,:] += oe.contract('iajb, icjb -> ac', tmp1[q, w, r,:,:,:,:], tmp1_bar.conj()[q, w, r,:,:,:,:], optimize=path_ac)
                    #y1 = numpy.einsum('qwrtiajb, qerticjb -> weac', tmp1, tmp1_bar.conj())


        # Chạy chính xác
        for w in range(nkpts):
            for e in range(nkpts):
                c2[w,e,:,nocc:] -= lib.einsum('pa, ac -> pc', fock_hf[w,:,nocc:], y1[w,e,:,:])
                #c2[:,:,:,nocc:] -= numpy.einsum('wpa, weac -> wepc',fock_hf[:,:,nocc:], y1)

    c0 = c0_2nd/nkpts + c0_1st
       
    c1 += numpy.einsum('wwps -> wps', c2)  * weight[idx_ibz] * nkpts**3

        
    return c0, c1, IP, EA
    

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

def _padding_k_idx(nmo, nocc, kind="split"):
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
    if kind not in ("split", "joint"):
        raise ValueError("The 'kind' argument must be one of 'split', 'joint'")

    if kind == "split":
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
        if kind == "split":
            indexes_o.append(numpy.arange(k_o))
            indexes_v.append(numpy.arange(dense_v - k_v, dense_v))
        else:
            indexes.append(numpy.concatenate((
                numpy.arange(k_o),
                numpy.arange(dense_nmo - k_v, dense_nmo),
            )))

    if kind == "split":
        return indexes_o, indexes_v

    else:
        return indexes


def padding_k_idx(mp, kind="split"):
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
    padding_convention = padding_k_idx(mp, kind="joint")
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
    padding_convention = padding_k_idx(mp, kind="joint")
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
    padding_convention = padding_k_idx(mp, kind="joint")
    nkpts = mp.nkpts

    result = numpy.zeros((nkpts, mo_coeff[0].shape[0], mp.nmo), dtype=mo_coeff[0].dtype)
    for k in range(nkpts):
        result[numpy.ix_([k], numpy.arange(result.shape[1]), padding_convention[k])] = mo_coeff[k][:, frozen_mask[k]]

    return result


def _frozen_sanity_check(frozen, mo_occ, kpt_idx):
    '''Performs a few sanity checks on the frozen array and mo_occ.

    Specific tests include checking for duplicates within the frozen array.

    Args:
        frozen (array_like of int): The orbital indices that will be frozen.
        mo_occ (:obj:`ndarray` of int): The occupuation number for each orbital
            resulting from a mean-field-like calculation.
        kpt_idx (int): The k-point that `mo_occ` and `frozen` belong to.

    '''
    frozen = numpy.array(frozen)
    nocc = numpy.count_nonzero(mo_occ > 0)
    nvir = len(mo_occ) - nocc
    assert nocc, 'No occupied orbitals?\n\nnocc = %s\nmo_occ = %s' % (nocc, mo_occ)
    all_frozen_unique = (len(frozen) - len(numpy.unique(frozen))) == 0
    if not all_frozen_unique:
        raise RuntimeError('Frozen orbital list contains duplicates!\n\nkpt_idx %s\n'
                           'frozen %s' % (kpt_idx, frozen))
    if len(frozen) > 0 and numpy.max(frozen) > len(mo_occ) - 1:
        raise RuntimeError('Freezing orbital not in MO list!\n\nkpt_idx %s\n'
                           'frozen %s\nmax orbital idx %s' % (kpt_idx, frozen, len(mo_occ) - 1))


def get_nocc(mp, per_kpoint=False):
    '''Number of occupied orbitals for k-point calculations.

    Number of occupied orbitals for use in a calculation with k-points, taking into
    account frozen orbitals.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.
        per_kpoint (bool, optional): True returns the number of occupied
            orbitals at each k-point.  False gives the max of this list.

    Returns:
        nocc (int, list of int): Number of occupied orbitals. For return type, see description of arg
            `per_kpoint`.

    '''
    for i, moocc in enumerate(mp.mo_occ):
        if numpy.any(moocc % 1 != 0):
            raise RuntimeError("Fractional occupation numbers encountered @ kp={:d}: {}. This may have been caused by "
                               "smearing of occupation numbers in the mean-field calculation. If so, consider "
                               "executing mf.smearing_method = False; mf.mo_occ = mf.get_occ() prior to calling "
                               "this".format(i, moocc))
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
            raise RuntimeError('Frozen list has a different number of k-points (length) than passed in mean-field/'
                               'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                               '(length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        [_frozen_sanity_check(frozen, mo_occ, ikpt) for ikpt, frozen, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]

        nocc = []
        for ikpt, frozen in enumerate(mp.frozen):
            max_occ_idx = numpy.max(numpy.where(mp.mo_occ[ikpt] > 0))
            frozen_nocc = numpy.sum(numpy.array(frozen) <= max_occ_idx)
            nocc.append(numpy.count_nonzero(mp.mo_occ[ikpt]) - frozen_nocc)
    else:
        raise NotImplementedError

    assert any(numpy.array(nocc) > 0), ('Must have occupied orbitals! \n\nnocc %s\nfrozen %s\nmo_occ %s' %
           (nocc, mp.frozen, mp.mo_occ))

    if not per_kpoint:
        nocc = numpy.amax(nocc)

    return nocc


def get_nmo(mp, per_kpoint=False):
    '''Number of orbitals for k-point calculations.

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

    '''
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
            raise RuntimeError('Frozen list has a different number of k-points (length) than passed in mean-field/'
                               'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                               '(length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        [_frozen_sanity_check(fro, mo_occ, ikpt) for ikpt, fro, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]

        nmo = [len(mp.mo_occ[ikpt]) - len(mp.frozen[ikpt]) for ikpt in range(nkpts)]
    else:
        raise NotImplementedError

    assert all(numpy.array(nmo) > 0), ('Must have a positive number of orbitals!\n\nnmo %s\nfrozen %s\nmo_occ %s' %
           (nmo, mp.frozen, mp.mo_occ))

    if not per_kpoint:
        # Depending on whether there are more occupied bands, we want to make sure that
        # nmo has enough room for max(nocc) + max(nvir) number of orbitals for occupied
        # and virtual space
        nocc = mp.get_nocc(per_kpoint=True)
        nmo = numpy.max(nocc) + numpy.max(numpy.array(nmo) - numpy.array(nocc))

    return nmo


def get_frozen_mask(mp):
    '''Boolean mask for orbitals in k-point post-HF method.

    Creates a boolean mask to remove frozen orbitals and keep other orbitals for post-HF
    calculations.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.

    Returns:
        moidx (list of :obj:`ndarray` of `numpy.bool`): Boolean mask of orbitals to include.

    '''
    moidx = [numpy.ones(x.size, dtype=numpy.bool) for x in mp.mo_occ]
    if isinstance(mp.frozen, (int, numpy.integer)):
        for idx in moidx:
            idx[:mp.frozen] = False
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        frozen = list(mp.frozen)
        for idx in moidx:
            idx[frozen] = False
    elif isinstance(mp.frozen[0], (list, numpy.ndarray)):
        nkpts = len(mp.frozen)
        if nkpts != mp.nkpts:
            raise RuntimeError('Frozen list has a different number of k-points (length) than passed in mean-field/'
                               'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                               '(length = %d)' % (mp.nkpts, mp.frozen, nkpts))
        [_frozen_sanity_check(fro, mo_occ, ikpt) for ikpt, fro, mo_occ in zip(range(nkpts), mp.frozen, mp.mo_occ)]
        for ikpt, kpt_occ in enumerate(moidx):
            kpt_occ[mp.frozen[ikpt]] = False
    else:
        raise NotImplementedError

    return moidx


class OBMP2(lib.StreamObject):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):

        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

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

        self._IP = None # Thêm dòng này
        self._EA = None # Thêm dòng này

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
    #int_transform = int_transform

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen != 0:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    @property
    def emp2(self):
        return self.e_corr

    @property
    def e_tot(self):
        return self.ene_tot #+ self._scf.e_tot
    
    @property
    def IP(self):
        return self._IP  # Sử dụng biến riêng

    @IP.setter
    def IP(self, value):
        self._IP = value

    @property
    def EA(self):
        return self._EA  # Sử dụng biến riêng

    @EA.setter
    def EA(self, value):
        self._EA = value


    def kernel(self, shift=0.0, mo_energy=None, mo_coeff=None, mo_occ=None, with_t2=WITH_T2,
               _kern=kernel):
        '''
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.     
        '''
        if mo_occ is None:
            mo_occ = self.mo_occ
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None or mo_occ is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff, mo_energy are not given.\n'
                     'You may need to call mf.kernel() to generate them.')

        mo_coeff, mo_energy, mo_occ = _add_padding(self, mo_coeff, mo_energy, mo_occ)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        #self.dump_flags()
        #_kern(self, mo_energy, mo_coeff, eris, with_t2, self.verbose)
        self.ene_tot, self.mo_energy, self.IP, self.EA= _kern(self, mo_energy, mo_coeff, mo_occ, with_t2, self.verbose)
        self._finalize()
        return self.ene_tot, self.mo_energy, self.IP, self.EA

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''        
        # Log tổng năng lượng
        logger.note(self, 'E(%s) = %.15g', self.__class__.__name__, self.e_tot)
    
        # Log IP và EA nếu có giá trị
        '''
        if hasattr(self, 'IP') and self.IP is not None:
            logger.note(self, 'Ionization Potential (IP) = %.5f eV', self.IP * 27.2114)  # Chuyển từ Hartree sang eV
        '''

        if self.IP is not None:
            logger.note(self, 'Ionization Potential (IP) = %.5f eV', self.IP * 27.2114)

        '''
        if hasattr(self, 'EA') and self.EA is not None:
            logger.note(self, 'Electron Affinity (EA)   = %.5f eV', self.EA * 27.2114)
        '''

        if self.EA is not None:
            logger.note(self, 'Electron Affinity (EA) = %.5f eV', self.EA * 27.2114)

        return self

    make_veff = make_veff
    #make_amp  = make_amp
    first_BCH = first_BCH
    #second_BCH = second_BCH
    #make_rdm1 = make_rdm1
    #make_rdm2 = make_rdm2

    #as_scanner = as_scanner

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


del(WITH_T2)
