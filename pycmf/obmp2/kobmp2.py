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

import time, tracemalloc
#import time, logging, tracemalloc
from functools import reduce
import copy
import numpy
import scipy.linalg
#from pyscf import gto
from pyscf import lib
#from pyscf import cc
from pyscf.lib import logger
from pyscf.pbc.lib import kpts_helper
from pyscf import __config__
from numba import jit
import pickle
#from pyscf.pbc.mp import kmp2

import datetime
import opt_einsum as oe
from pyscf import lib
import h5py
import os
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
def prepare_contract_paths(mp):
    """
    Chuẩn bị toàn bộ contraction paths (opt_einsum) dùng trong kernel OBMP2.

    Parameters
    ----------
    mp : object
        Đối tượng MP chứa tối thiểu các thuộc tính:
            - nmo
            - nocc
            - second_order (bool)

    Returns
    -------
    paths : dict
        Từ điển chứa tất cả các path. Các khóa:
            'pj','bp','bj','sum_3',
            'jb','ck','iakb','lj','ki','bd','sum',
            'ialb','lj_2','ac_2','bd_2','sum_2','ki_3','ac_3'
        Với mp.second_order = False, những khóa bậc hai trả về None giống logic cũ.

    Ghi chú
    -------
    - Các dummy array chỉ dùng để suy ra chiến lược contraction tối ưu, không giữ lại.
    - Path chính là phần tử đầu (index 0) do oe.contract_path(...) trả về.
    - Khi sử dụng lại trong einsum/oe.contract bạn gọi:
          numpy.einsum('...', A, B, optimize=paths['pj'])
      hoặc:
          oe.contract('iajb, iapb -> pj', Tbar, V, optimize=paths['pj'])
    - Tất cả shapes phải nhất quán với tensor thực tế trong phần còn lại của code.
    """
    nmo  = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc

    print(f"[prepare_contract_paths] Đang tính toán paths cho nmo={nmo}, nocc={nocc}...")
    start_time = time.time() # Bấm giờ

    # ------------------------------------------
    # Dummy tensors (tạo cùng shape như trong kernel gốc)
    # ------------------------------------------
    dummy_tmp1      = numpy.empty((nocc, nvir, nocc, nvir), dtype=complex)  # (i,a,j,b)
    dummy_tmp1_bar  = numpy.empty((nocc, nvir, nocc, nvir), dtype=complex)  # (i,a,j,b)
    dummy_fock_hf   = numpy.empty((nmo, nmo), dtype=complex)                # (p,q) full MO

    # Các tensor 4D chứa 1 chỉ số "MO tổng quát" và 1 chỉ số virtual:
    # h2mo_ovgv: (i,a, g, v) ~ ov g v  (g: all mo, v: vir)
    dummy_h2mo_ovgv = numpy.empty((nocc, nvir, nmo, nvir), dtype=complex)
    path_pj = oe.contract_path('iajb, iapb -> pj', dummy_tmp1_bar, dummy_h2mo_ovgv,
                               optimize='optimize')[0]

    # h2mo_ovog: (i,a, o, g) ~ o: occ, g: all mo
    dummy_h2mo_ovog = numpy.empty((nocc, nvir, nocc, nmo), dtype=complex)
    path_bp = oe.contract_path('iajb, iajp -> bp', dummy_tmp1_bar, dummy_h2mo_ovog,
                               optimize='optimize')[0]

    # fock (vir, occ) * tmp1_bar (i,a,j,b)
    path_bj = oe.contract_path('ai, iajb -> bj',
                               dummy_fock_hf[nocc:, :nocc], dummy_tmp1_bar,
                               optimize='optimize')[0]

    # ovov block
    dummy_h2mo_ovov = numpy.empty((nocc, nvir, nocc, nvir), dtype=complex)
    path_sum_3 = oe.contract_path('iajb, iajb ->',
                                  dummy_tmp1_bar, dummy_h2mo_ovov,
                                  optimize='optimize')[0]

    if mp.second_order:
        # Bậc hai bổ sung
        path_jb = oe.contract_path('ai, iajb -> jb',
                                   dummy_fock_hf[nocc:, :nocc], dummy_tmp1_bar,
                                   optimize='optimize')[0]

        dummy_y1 = numpy.empty((nocc, nvir), dtype=complex)            # (j,b)
        path_ck  = oe.contract_path('jb, jbkc -> ck',
                                    dummy_y1, dummy_tmp1_bar,
                                    optimize='optimize')[0]

        path_iakb = oe.contract_path('ca, ickb -> iakb',
                                     dummy_fock_hf[nocc:, nocc:], dummy_tmp1_bar,
                                     optimize='optimize')[0]

        dummy_y1_1 = numpy.empty((nocc, nvir, nocc, nvir), dtype=complex)
        path_lj  = oe.contract_path('ialb,iajb->lj', dummy_y1_1, dummy_tmp1,
                                    optimize='optimize')[0]
        path_ki  = oe.contract_path('kajb,iajb->ki', dummy_y1_1, dummy_tmp1,
                                    optimize='optimize')[0]
        path_bd  = oe.contract_path('iajd,iajb->bd', dummy_y1_1, dummy_tmp1,
                                    optimize='optimize')[0]
        path_sum = oe.contract_path('iajb,iajb->',   dummy_y1_1, dummy_tmp1,
                                    optimize='optimize')[0]

        path_ialb = oe.contract_path('ki, kalb -> ialb',
                                     dummy_fock_hf[:nocc, :nocc], dummy_tmp1_bar,
                                     optimize='optimize')[0]

        dummy_y1_2 = numpy.empty((nocc, nvir, nocc, nvir), dtype=complex)
        path_lj_2  = oe.contract_path('ialb, iajb -> lj',
                                      dummy_y1_2, dummy_tmp1,
                                      optimize='optimize')[0]
        path_ac_2  = oe.contract_path('icjb, iajb -> ac',
                                      dummy_y1_2, dummy_tmp1,
                                      optimize='optimize')[0]
        path_bd_2  = oe.contract_path('iajd, iajb -> bd',
                                      dummy_y1_2, dummy_tmp1,
                                      optimize='optimize')[0]
        path_sum_2 = oe.contract_path('iajb, iajb ->',
                                      dummy_y1_2, dummy_tmp1,
                                      optimize='optimize')[0]

        path_ki_3  = oe.contract_path('iajb, kajb -> ki',
                                      dummy_tmp1, dummy_tmp1_bar,
                                      optimize='optimize')[0]
        path_ac_3  = oe.contract_path('iajb, icjb -> ac',
                                      dummy_tmp1, dummy_tmp1_bar,
                                      optimize='optimize')[0]
    else:
        path_jb = path_ck = path_iakb = path_lj = path_ki = path_bd = path_sum = path_ialb = \
        path_lj_2 = path_ac_2 = path_bd_2 = path_sum_2 = path_ki_3 = path_ac_3 = None

    print(f"--- Đã tính toán xong paths. (Tốn {time.time() - start_time:.4f} giây)")

    return {
        'pj': path_pj,
        'bp': path_bp,
        'bj': path_bj,
        'sum_3': path_sum_3,
        'jb': path_jb,
        'ck': path_ck,
        'iakb': path_iakb,
        'lj': path_lj,
        'ki': path_ki,
        'bd': path_bd,
        'sum': path_sum,
        'ialb': path_ialb,
        'lj_2': path_lj_2,
        'ac_2': path_ac_2,
        'bd_2': path_bd_2,
        'sum_2': path_sum_2,
        'ki_3': path_ki_3,
        'ac_3': path_ac_3
    }

def get_or_create_contract_paths(mp, cache_dir="./einsum_cache"):
    """
    Tải (load) các einsum paths đã tối ưu từ file cache nếu có.
    Nếu không, tính toán chúng và lưu vào cache cho lần sau.
    
    Một file cache .pkl duy nhất sẽ được tạo ra cho mỗi
    tổ hợp (nmo, nocc, second_order).
    """
    
    # 1. Xác định các tham số duy nhất của hệ
    nmo = mp.nmo
    nocc = mp.nocc
    so = mp.second_order
    
    # Tạo thư mục cache nếu nó chưa tồn tại
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
            print(f"Đã tạo thư mục cache: {cache_dir}")
        except OSError as e:
            print(f"Lỗi khi tạo thư mục cache {cache_dir}: {e}")
            # Nếu không tạo được, chạy tính toán mà không lưu
            return prepare_contract_paths(mp)

    # 2. Tạo tên file duy nhất
    filename = f"obmp2_paths_nmo{nmo}_nocc{nocc}_so{bool(so)}.pkl"
    filepath = os.path.join(cache_dir, filename)

    # 3. Logic: Tải (Load) hoặc Tính toán (Calculate) & Lưu (Save)
    try:
        # --- THỬ TẢI (LOAD) ---
        if os.path.exists(filepath):
            print(f"--- Đã tìm thấy file path cache: {filepath}")
            print("--- Đang tải paths từ file (siêu nhanh)...")
            start_load = time.time()
            with open(filepath, 'rb') as f:
                paths = pickle.load(f)
            print(f"--- Tải paths thành công! (Tốn {time.time() - start_load:.4f} giây)")
            return paths
            
        # --- TÍNH TOÁN & LƯU (SAVE) ---
        else:
            print(f"--- Không tìm thấy file cache: {filepath}")
            print("--- Bắt đầu tính toán einsum paths (có thể mất vài phút)...")
            
            # Đây là lệnh gọi tốn thời gian
            paths = prepare_contract_paths(mp) 
            
            print(f"--- Tính toán xong. Đang lưu vào file cache để dùng cho lần sau...")
            
            # Lưu kết quả vào file
            with open(filepath, 'wb') as f:
                pickle.dump(paths, f)
                
            print(f"--- Đã lưu paths vào: {filepath}")
            return paths

    except Exception as e:
        print(f"Lỗi nghiêm trọng trong quá trình load/save cache: {e}")
        print("--- Chạy tính toán paths mà không dùng cache ---")
        # Dự phòng: Nếu có lỗi (ví dụ: không có quyền ghi file),
        # thì chạy tính toán như bình thường mà không lưu
        return prepare_contract_paths(mp)
    
@jit(nopython=True) # Biên dịch hàm này sang mã máy
def sum_occ_diag(matrix_kpts, nocc, nkpts, factor):
    '''
    Hàm JIT hóa để tính tổng các phần tử đường chéo 
    của các orbital bị chiếm đóng (occupied).
    '''
    total_sum = 0.0
    for k in range(nkpts):
        for i in range(nocc):
            # Dùng .real vì Numba cần biết kiểu dữ liệu rõ ràng
            total_sum += factor * matrix_kpts[k][i, i].real
            
    return total_sum

def kernel(mp, mo_energy, mo_coeff, mo_occ, with_t2=WITH_T2,
           verbose=logger.NOTE):
    
    # Khởi tạo
    nuc = mp._scf.energy_nuc() # Năng lượng hạt nhân
    nmo = mp.nmo # Số orbital phân tử
    nkpts = numpy.shape(mo_energy)[0] # Số điểm k
    nocc = mp.nocc # Số orbital chiếm đóng
    #nvir = nmo - nocc
    niter = mp.niter # Số vòng lặp tối đa
    ene_old = 0.  # Năng lượng ban đầu
    dm = mp._scf.make_rdm1(mo_coeff, mo_occ) # Ma trận mật độ
    
    # Chuẩn bị DIIS (Direct Inversion in the Iterative Subspace) <Tìm hiểu sau>
    #DIIS_RESID = [[] for _ in range(nkpts)]
    #F_list = [[] for _ in range(nkpts)]

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
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3] # Thêm mili giây.
    
    paths = get_or_create_contract_paths(mp)
    (path_pj, path_bp, path_bj, path_sum_3,
     path_jb, path_ck, path_iakb, path_lj, path_ki,
     path_bd, path_sum, path_ialb, path_lj_2, path_ac_2,
     path_bd_2, path_sum_2, path_ki_3, path_ac_3) = (
        paths['pj'], paths['bp'], paths['bj'], paths['sum_3'],
        paths['jb'], paths['ck'], paths['iakb'], paths['lj'], paths['ki'],
        paths['bd'], paths['sum'], paths['ialb'], paths['lj_2'], paths['ac_2'],
          paths['bd_2'], paths['sum_2'], paths['ki_3'], paths['ac_3']
    )

    '''
    a. Xây dựng Hamiltonian và potential hiệu dụng
    '''
    h1ao = mp._scf.get_hcore() # Tính toán ma trận Hamiltonian lõi (h1ao) trong không gian orbital nguyên tử (AO)
    veffao = mp._scf.get_veff(mp._scf.cell, dm) # Tính toán thế năng hiệu dụng (veffao) dựa trên ma trận mật độ hiện tại

    for it in range(niter):

        '''
        a. Xây dựng Hamiltonian và potential hiệu dụng
        '''
        
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
        
        '''
        # Tính hằng số c0_hf
        c0_hf = 0
        for kp in range(nkpts):
            for i in range(nocc):
                c0_hf -=  veff[kp][i,i].real
        c0_hf/= nkpts
        '''

        # Yếu tố factor là -1.0 và phép chia /nkpts đã được đưa vào hàm
        c0_hf = sum_occ_diag(veff, nocc, nkpts, -1.0) / nkpts

        #fock_hf = h1mo
        #fock_hf += veff

        # Xây dựng ma trận Fock trong cơ sở Hartree-Fock
        fock_hf = numpy.zeros((nkpts, nmo, nmo), dtype=complex) # Khởi tạo ma trận Fock
        fock_hf += veff
        fock_hf += [reduce(numpy.dot, (mo.T.conj(), h1ao[k], mo))
                                for k, mo in enumerate(mp.mo_coeff)]
        numpy.set_printoptions(precision=6)
        #fockao = h1ao + veffao
        #print("fock_hf")
        #print(fock_hf[0])
        #initializing w/ HF
        fock = 0
        fock += fock_hf
        c0 = c0_hf 

        '''
        c. Tính năng lượng Hartree-Fock
        '''
        '''
        ene_hf = 0
        for k in range(nkpts):
            for i in range(nocc):
                ene_hf += 2*fock[k][i,i].real/nkpts

        ene_hf +=c0_hf + nuc 
        '''

        ene_hf_elec = sum_occ_diag(fock, nocc, nkpts, 2.0) / nkpts
        ene_hf = ene_hf_elec + c0_hf + nuc   

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
        c0_1st, c1, IP, EA = BCH(mp, mp.mo_energy, mp.mo_coeff, fock_hf, path_pj, path_bp,
                                  path_bj, path_sum_3, path_jb, path_ck, path_iakb,
                                    path_lj, path_ki, path_bd, path_sum, path_ialb,
                                      path_lj_2, path_ac_2, path_bd_2, path_sum_2,
                                        path_ki_3, path_ac_3, timestamp)
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

        '''
        ene = 0
        for k in range(nkpts):
            for i in range(nocc):
                ene += 2*fock[k][i,i].real/nkpts

        ene_tot = ene + c0  + nuc + c0_1st 
        '''

        # Yếu tố factor là 2.0
        ene = sum_occ_diag(fock, nocc, nkpts, 2.0) / nkpts
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
        #stat = top_stats[:10]
        total_mem = sum(stat.size for stat in top_stats)
        print("total_mem = ",total_mem)
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

    '''
    c0_hf = 0
    for kp in range(nkpts):
        for i in range(nocc):
            c0_hf -=  veff[kp][i,i].real
    c0_hf/= nkpts
    '''

    # Yếu tố factor là -1.0 và phép chia /nkpts đã được đưa vào hàm
    c0_hf = sum_occ_diag(veff, nocc, nkpts, -1.0) / nkpts
    
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
def BCH(mp, mo_energy, mo_coeff, fock_hf,
        path_pj, path_bp, path_bj, path_sum_3,
        path_jb, path_ck, path_iakb, path_lj,
        path_ki, path_bd, path_sum, path_ialb,
        path_lj_2, path_ac_2, path_bd_2, path_sum_2,
        path_ki_3, path_ac_3, timestamp):
    star = time.time()

    nmo = mp.nmo # Số lượng các Orbital phân tử (lấp đầy + trống)
    nocc = mp.nocc # Số lượng các Orbital phân tử đã lấp đầy
    nvir = nmo - nocc # Số lượng các Orbital phân tử trống
    nkpts = numpy.shape(mo_energy)[0] # Cho số lượng các k - point trong tính toán
    kpts = mp.kpts # Mảng chứa các k - point trong không gian sóng của phân tử hoặc tinh thể

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

    kconserv = kpts_helper.get_kconserv(mp._scf.cell, kpts) 

    fao2mo = mp._scf.with_df.ao2mo  
    # Từ dòng fao2mo trở lên là giữ nguyên, không đổi.
    
    # Định nghĩa batch size cho ki và kj
    batch_size = 10  # (NEW) Có thể điều chỉnh tùy theo RAM
    
    '''
    # Tạo thư mục tạm trên ổ đĩa cứng để lưu dữ liệu
    temp_dir = "./tmp_arrays"
    '''

    temp_dir = os.environ.get('TMPDIR', './tmp_arrays')
    print(f"Sử dụng thư mục tạm: {temp_dir}") # (SỬA) Thêm log để kiểm tra

    # Tạo thư mục tạm nếu chưa có
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    try:

        # Tạo file HDF5 để lưu tmp1 và tmp1_bar
        h5_file = os.path.join(temp_dir, f"tmp_arrays_{timestamp}.h5")
        # File lưu tích phân h2mo_ovgg (phase 1) (Cái mới thêm vào)
        h5_int_file = os.path.join(temp_dir, f"h2mo_ovgg_{timestamp}.h5")

        # Khởi tạo các biến tích lũy
        # Các biến không tích lũy sẽ được khởi tạo sau
        c1 = numpy.zeros((nkpts, nkpts, nmo, nmo), dtype=complex)
        c0_1st = 0.0
        IP = 0.0
        EA = 0.0

        print('#################################################')
        print('# Phần 1: Tính và lưu toàn bộ h2mo_ovgg (batch) #')
        print('#################################################')

        # Phần 1: tính toàn bộ h2mo_ovgg và ghi xuống đĩa
        with h5py.File(h5_int_file, 'w') as fint:
            dset_h2mo_ovgg = fint.create_dataset(
                'h2mo_ovgg',
                (nkpts, nkpts, nkpts, nocc, nvir, nmo, nmo),
                dtype=complex,
                chunks=(1, 1, 1, nocc, nvir, nmo, nmo),
                compression='gzip'
            )

            for ki_start in range(0, nkpts, batch_size):
                ki_end = min(ki_start + batch_size, nkpts)
                batch_ki_size = ki_end - ki_start

                for kj_start in range(0, nkpts, batch_size):
                    kj_end = min(kj_start + batch_size, nkpts)
                    batch_kj_size = kj_end - kj_start

                    print(f"[Phần 1] Batch ki=[{ki_start}:{ki_end}], kj=[{kj_start}:{kj_end}]")

                    for ki in range(ki_start, ki_end):
                        o_i = mo_coeff[ki][:, :nocc]
                        for kj in range(kj_start, kj_end):
                            kp = kj
                            o_p = mo_coeff[kp]
                            for ka in range(nkpts):
                                kb = kconserv[ki, ka, kj]
                                kq = kb
                                o_a = mo_coeff[ka][:, nocc:]
                                o_q = mo_coeff[kq]
                                print(f"[Phần 1] 9. Processing k-points: ki={ki}, kj={kj}, ka={ka}, kb={kb}")
                                dset_h2mo_ovgg[ki, ka, kj] = fao2mo(
                                    (o_i, o_a, o_p, o_q),
                                    (kpts[ki], kpts[ka], kpts[kp], kpts[kq]),
                                    compact=False
                                ).reshape(nocc, nvir, nmo, nmo) / nkpts

        print('#################################################')
        print('# Phần 2: Đọc h2mo_ovgg, cắt và tính amplitudes #')
        print('#################################################')

        # PHẦN 1: Tính toán và lưu tmp1, tmp1_bar

        # Tạo file HDF5 để lưu kết quả (do dữ liệu quá lớn để lưu trong RAM)
        # Tại sao lại phải lưu h2mo_ovgg vào 1 file riêng (h5_int_file) và tmp1, tmp1_bar vào một file riêng khác (h5_file)?
        # => Đó là vì ở bước tính c1, ma trận h2mo_ovgg cần được trích xuất ra (đọc file h5_int_file), còn hai ma trận tmp1, tmp1_bar
        # thì cần được lưu vào (ghi file h5_file)
        with h5py.File(h5_file, 'w') as famp, h5py.File(h5_int_file, 'r') as fint:
            # Tạo dataset cho tmp1 và tmp1_bar
            tmp1_dset = famp.create_dataset('tmp1', 
                                         (nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir), 
                                         dtype=complex,
                                         chunks=(1, 1, 1, nocc, nvir, nocc, nvir),
                                         compression='gzip')
            tmp1_bar_dset = famp.create_dataset('tmp1_bar', 
                                             (nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir), 
                                             dtype=complex,
                                             chunks=(1, 1, 1, nocc, nvir, nocc, nvir),
                                             compression='gzip')

            # Batch giống phase ban đầu: trên ki,kj
            # Chia vòng lặp ki thành các batch
            for ki_start in range(0, nkpts, batch_size): # VD nkpts = 8, batch_size = 2 thì range(0, 8, 2) = [0, 2, 4, 6]
                ki_end = min(ki_start + batch_size, nkpts) 
                batch_ki_size = ki_end - ki_start

                # VD nkpts = 8, batch_size = 2 thì ki_end = [2, 4, 6, 8], 
                #ngoài ra hàm min(_,nkpts) đảm bảo rằng giá trị lớn nhất là nkpts

                # Chia vòng lặp kj thành các batch
                for kj_start in range(0, nkpts, batch_size): # tương tự ki_start
                    kj_end = min(kj_start + batch_size, nkpts) # tương tự ki_end
                    batch_kj_size = kj_end - kj_start

                    # Khởi tạo các mảng tạm cho batch hiện tại
                    # Khúc này xác định lại khoảng giữa ki_end và ki_start, có thể bạn thấy hơi thừa vì mình đã
                    # xác định lại khoảng giữa ki_end và ki_start ở trên (biến batch_size),
                    # nhưng sẽ có trường hợp mà ki_end hoặc kj_end đặt đến giá trị của nkpts, nhưng ki_end - ki_start
                    # lại nhỏ hơn batch_size.
                    # VD: nkpts = 8, batch_size = 6 => [ki_start = 0, ki_end = 6], [ki_start = 6, kj_end = 8], 
                    # rõ ràng batch_size của bộ cuối là số 2 KHÁC số 6.

                    # Tính toán tích phân 2 electron cho batch hiện tại
                    # Tại sao lại phải có 2 biến ki_idx, ki?
                    # -> Đó là do tính chất của hàm enumerate, dưới đây là 1 code ví dụ:
                    '''
                    names = ['Bob', 'Alice', 'Guido']
                    for index, value in enumerate(names):
                        print(f'{index}: {value}')

                    //output: 
                    0: Bob
                    1: Alice
                    2: Guido

                    x = range(0, 8, 6)
                    z = enumerate(range(6, 8))
                    for n,i in z:
	                    print(f"i_1={n}")
	                    print(f"i={i}")

                    //output:
                    i_1=0
                    i=6
                    i_1=1
                    i=7
                    '''

                    print(f"[Phần 2] Processing batch: ki=[{ki_start}:{ki_end}], kj=[{kj_start}:{kj_end}]")

                    # Khởi tạo mảng batch cho tmp1 / tmp1_bar
                    tmp1_batch = numpy.zeros(
                        (batch_ki_size, nkpts, batch_kj_size, nocc, nvir, nocc, nvir),
                        dtype=complex
                    )
                    tmp1_bar_batch = numpy.zeros_like(tmp1_batch)

                    ki_1 = -1
                    kj_1 = -1

                    # h2mo_ovov[ki_idx,ka,kj_idx] = h2mo_ovgg[ki_idx,ka,kj_idx][:,:,:nocc, nocc:] 
                    # h2mo_ovgv[ki_idx,ka,kj_idx] = h2mo_ovgg[ki_idx,ka,kj_idx][:,:,:, nocc:]
                    # h2mo_ovog[ki_idx,ka,kj_idx] = h2mo_ovgg[ki_idx,ka,kj_idx][:,:,:nocc, :]
                    # h2mo_ovvv[ki_idx,ka,kj_idx] = h2mo_ovgg[ki_idx,ka,kj_idx][:,:,nocc:,nocc:]
                    # h2mo_ovoo[ki_idx,ka,kj_idx] = h2mo_ovgg[ki_idx,ka,kj_idx][:,:,:nocc,:nocc]

                    for ki_idx, ki in enumerate(range(ki_start, ki_end)): 
                        for kj_idx, kj in enumerate(range(kj_start, kj_end)):
                            for ka in range(nkpts):
                                kb = kconserv[ki,ka,kj]
                                print(f"[PHASE 2] 18. Processing: ki={ki}, kj={kj}, ka={ka}, kb={kb}")

                                # Đọc block h2mo_ovgg tương ứng
                                block_ovgg = fint['h2mo_ovgg'][ki, ka, kj]  # shape (nocc,nvir,nmo,nmo)

                                # Cắt theo yêu cầu (mọi thao tác cắt đều dưới 3 vòng for)
                                h2mo_ovov = block_ovgg[:, :, :nocc,  nocc:]   # (nocc,nvir,nocc,nvir)
                                h2mo_ovgv = block_ovgg[:, :, :,      nocc:]   # (nocc,nvir,nmo,nvir)
                                h2mo_ovog = block_ovgg[:, :, :nocc,  :]       # (nocc,nvir,nocc,nmo)
                                h2mo_ovvv = block_ovgg[:, :, nocc:,  nocc:]   # (nocc,nvir,nvir,nvir)
                                h2mo_ovoo = block_ovgg[:, :, :nocc,  :nocc]   # (nocc,nvir,nocc,nocc)

                                # Cắt h2mo_ovoo[kj,ka,ki] phục vụ w_iajh (theo yêu cầu)
                                block_ovgg_kj_ka_ki = fint['h2mo_ovgg'][kj, ka, ki]
                                h2mo_ovoo_kj_ka_ki = block_ovgg_kj_ka_ki[:, :, :nocc, :nocc]

                                e_iajb, e_iajh, e_ialb = ene_denom(mp, mo_energy, ki, ka, kj, kb) # Đúng

                                # w_iajb
                                w_iajb = (h2mo_ovov - 0.5 * h2mo_ovov.transpose(0, 3, 2, 1)
                                          if kb == ka else
                                          (h2mo_ovov - 0.5 * fint['h2mo_ovgg'][ki, kb, kj][:, :, :nocc, nocc:].transpose(0, 3, 2, 1)))
                                # w_iajb = (h2mo_ovov[ki,ka,kj] - 0.5*h2mo_ovov[ki,kb,kj].transpose(0,3,2,1))

                                #tmp1[ki_idx,ka,kj_idx] = (h2mo_ovov[ki_idx,ka,kj_idx]/e_iajb).conj()
                                #tmp1_bar[ki_idx,ka,kj_idx] = (w_iajb/e_iajb).conj()

                                tmp1_batch[ki_idx,ka,kj_idx] = (h2mo_ovov / e_iajb).conj()
                                tmp1_bar_batch[ki_idx,ka,kj_idx] = (w_iajb / e_iajb).conj()

                                # w_iajh: dùng h2mo_ovoo[ki,ka,kj] và h2mo_ovoo[kj,ka,ki]
                                w_iajh = (h2mo_ovoo - 0.5 * h2mo_ovoo_kj_ka_ki.transpose(2, 1, 0, 3))
                                #w_iajh = (h2mo_ovoo[ki,ka,kj]-0.5*h2mo_ovoo[kj,ka,ki].transpose(2,1,0,3))
                                tmp1_bar_iajh = (w_iajh / e_iajh).conj()

                                w_ialb = (h2mo_ovvv - 0.5 * h2mo_ovvv.transpose(0,3,2,1)
                                          if kb == ka else 
                                          h2mo_ovvv - 0.5 * fint['h2mo_ovgg'][ki, kb, kj][:, :, nocc:,  nocc:].transpose(0,3,2,1)) # Đúng
                                #w_ialb = (h2mo_ovvv[ki,ka,kj]-0.5*h2mo_ovvv[ki,kb,kj].transpose(0,3,2,1))
                                tmp1_bar_ialb = (w_ialb/e_ialb).conj() # Đúng

                                mp.ampf = 0.5
                                tmp1_batch[ki_idx,ka,kj_idx] *= mp.ampf
                                tmp1_bar_batch[ki_idx,ka,kj_idx] *= mp.ampf
                                tmp1_bar_iajh *= mp.ampf
                                tmp1_bar_ialb *= mp.ampf # Đúng

                                # Lưu vào HDF5 (mới) 
                                tmp1_dset[ki,ka,kj] = tmp1_batch[ki_idx,ka,kj_idx]
                                tmp1_bar_dset[ki,ka,kj] = tmp1_bar_batch[ki_idx,ka,kj_idx]

                                if kb == 0:
                                    IP += 2 * numpy.einsum('iaj, iaj ->',
                                                           tmp1_bar_iajh[:, :, :, nocc-1],
                                                           h2mo_ovoo[:, :, :, nocc-1]).real
                                    #IP += 2*numpy.einsum('iaj, iaj ->',tmp1_bar_iajh[:,:,:,nocc-1],h2mo_ovoo[ki,ka,kj,:,:,:,nocc-1]).real

                                if kj == 0:
                                    h2mo_ovvv_ki_ka_0 = fint['h2mo_ovgg'][ki, ka, 0][:, :, nocc:,  nocc:]
                                    EA += -2*numpy.einsum('iab, iab ->',
                                                          tmp1_bar_ialb[:,:,0,:],
                                                          h2mo_ovvv_ki_ka_0[:,:,0,:]).real # Hiện tại đang chạy chính xác
                                    #EA += -2*numpy.einsum('iab, iab ->',tmp1_bar_ialb[:,:,0,:],h2mo_ovvv[ki,ka,0,:,:,0,:]).real
                                    del h2mo_ovvv_ki_ka_0 # Giải phóng bộ nhớ ngay sau khi sử dụng xong

                                c1[kj,kj,:,:nocc] += 2 * oe.contract('iajb, iapb -> pj',
                                                                 tmp1_bar_batch[ki_idx,ka,kj_idx],
                                                                 h2mo_ovgv, 
                                                                 optimize=path_pj)

                                c1[kb,kb,nocc:,:] -= 2 * oe.contract('iajb, iajp -> bp',
                                                                  tmp1_bar_batch[ki_idx,ka,kj_idx],
                                                                  h2mo_ovog, 
                                                                  optimize=path_bp)  

                                if ki != ki_1 or kj != kj_1:
                                    ki_1 = ki
                                    kj_1 = kj
                                    r = kconserv[ki,ki,kj]
                                    c1[r,kj,nocc:,:nocc] += 2*oe.contract('ai, iajb -> bj',
                                                                        fock_hf[ki,nocc:,:nocc].conj(),
                                                                        tmp1_bar_batch[ki_idx,ki,kj_idx], 
                                                                        optimize=path_bj) # Cần kiểm tra lại

                                c0_1st += -4*oe.contract('iajb, iajb ->',
                                                       tmp1_bar_batch[ki_idx,ka,kj_idx], 
                                                       h2mo_ovov, 
                                                       optimize = path_sum_3)

                    # Giải phóng bộ nhớ sau mỗi batch
                    del tmp1_batch, tmp1_bar_batch
                    del block_ovgg, h2mo_ovov, h2mo_ovog, h2mo_ovgv, h2mo_ovoo, h2mo_ovvv, h2mo_ovoo_kj_ka_ki # Lỗi ở dòng này

                    #del block_ovgg, h2mo_ovov, h2mo_ovog, h2mo_ovgv, h2mo_ovoo, h2mo_ovvv, h2mo_ovvv_ki_ka_0, h2mo_ovoo_kj_ka_ki # Lỗi ở dòng này
                    # Lỗi chính là không có ma trận h2mo_ovvv_ki_ka_0 để xóa
    
    finally:
        print('#################################################')
        print(f'# BẮT ĐẦU DỌN DẸP FILE TẠM TRONG {temp_dir} #')
        print('#################################################')
        
        # Xóa file tích phân h2mo_ovgg
        try:
            if os.path.exists(h5_int_file):
                os.remove(h5_int_file)
                print(f"Đã xóa thành công: {h5_int_file}")
            else:
                print(f"File không tồn tại, không cần xóa: {h5_int_file}")
        except (IOError, OSError) as e: # Bắt các lỗi có thể xảy ra khi xóa file
            print(f"Lỗi khi đang xóa file {h5_int_file}: {e}")
            
        print('#################################################')
        print(f'# KẾT THÚC DỌN DẸP FILE TẠM                #')
        print('#################################################')
            
    print(f"IP: {IP}, EA: {EA}")
    print(f"Running time {time.time() - star} seconds")

    c1 = numpy.einsum('wwpq -> wpq', c1)
    print("c1 = ", c1)
    c0_1st = c0_1st.real/nkpts

    c0 = c0_1st
    c0_2nd = 0.0
    c2 = numpy.zeros((nkpts, nkpts, nmo,nmo), dtype=complex)

    try: 
        if mp.second_order:
            # Mở file HDF5 ở chế độ đọc
            with h5py.File(h5_file, 'r') as f:
                tmp1_dset = f['tmp1']
                tmp1_bar_dset = f['tmp1_bar']

                # Tính toán y1
                y1 = numpy.zeros((nkpts, nkpts, nocc, nvir), dtype=complex)

                for ki in range(nkpts):
                    for kj in range(nkpts):
                        kb = kconserv[ki, ki, kj]
                        # Load tmp1_bar[ki, ki, kj] từ HDF5
                        tmp1_bar_data = tmp1_bar_dset[ki, ki, kj]
                        y1[kj, kb, :, :] += 2 * oe.contract('ai, iajb -> jb',
                                                           fock_hf[ki,nocc:,:nocc].conj(),
                                                           tmp1_bar_data, 
                                                           optimize=path_jb)

                # Tính c2 từ y1
                for ki in range(nkpts):
                    for kj in range(nkpts):
                        for kb in range(nkpts):
                            t = kconserv[kj, kb, ki]
                            # Load tmp1_bar[kj, kb, ki] từ HDF5
                            tmp1_bar_data = tmp1_bar_dset[kj, kb, ki]
                            c2[t,ki,nocc:nocc+nvir,:nocc] += oe.contract('jb, jbkc -> ck', 
                                                                        y1[kj, kb, :, :], 
                                                                        tmp1_bar_data.conj(), 
                                                                        optimize=path_ck)

                # Tính y1_1 theo batch
                for ki_start in range(0, nkpts, batch_size):
                    ki_end = min(ki_start + batch_size, nkpts)

                    for ka_start in range(0, nkpts, batch_size):
                        ka_end = min(ka_start + batch_size, nkpts)

                        # Khởi tạo y1_1 cho batch hiện tại
                        y1_1_batch = numpy.zeros((ka_end-ka_start, ki_end-ki_start, nkpts, nocc, nvir, nocc, nvir), dtype=complex)

                        for ki_idx, ki in enumerate(range(ki_start, ki_end)):
                            for ka_idx, ka in enumerate(range(ka_start, ka_end)):
                                for kb in range(nkpts):
                                    # Load tmp1_bar từ HDF5
                                    tmp1_bar_data = tmp1_bar_dset[ka, ki, kb]
                                    y1_1_batch[ka_idx,ki_idx,kb,:,:,:,:] += oe.contract('ca, ickb -> iakb', 
                                                                                       fock_hf[ki,nocc:,nocc:], 
                                                                                       tmp1_bar_data.conj(), 
                                                                                       optimize=path_iakb)

                        # Xử lý y1_1_batch
                        for w_idx, w in enumerate(range(ka_start, ka_end)):
                            for q_idx, q in enumerate(range(ki_start, ki_end)):
                                for t in range(nkpts):
                                    e = kconserv[w, q, t]

                                    # Load tmp1 từ HDF5
                                    tmp1_data = tmp1_dset[w, q, e]

                                    c2[e,e,:nocc,:nocc] += oe.contract('ialb, iajb -> lj', 
                                                                     y1_1_batch[w_idx, q_idx, e,:,:,:,:], 
                                                                     tmp1_data, 
                                                                     optimize=path_lj)

                                    c2[w,w,:nocc,:nocc] += oe.contract('kajb, iajb -> ki', 
                                                                     y1_1_batch[w_idx, q_idx, e,:,:,:,:], 
                                                                     tmp1_data, 
                                                                     optimize=path_ki)

                                    c2[t,t,nocc:nocc+nvir,nocc:nocc+nvir] -= oe.contract('iajd, iajb -> bd', 
                                                                                        y1_1_batch[w_idx, q_idx, e,:,:,:,:], 
                                                                                        tmp1_data, 
                                                                                        optimize=path_bd)

                                    c0_2nd -= 4*oe.contract('iajb,iajb ->', 
                                                          y1_1_batch[w_idx, q_idx, e,:,:,:,:], 
                                                          tmp1_data, 
                                                          optimize=path_sum).real

                        del y1_1_batch

                # Tính y1_2 theo batch (tương tự như trên)
                for ki_start in range(0, nkpts, batch_size):
                    ki_end = min(ki_start + batch_size, nkpts)

                    for ka_start in range(0, nkpts, batch_size):
                        ka_end = min(ka_start + batch_size, nkpts)

                        y1_2_batch = numpy.zeros((ki_end-ki_start, ka_end-ka_start, nkpts, nocc, nvir, nocc, nvir), dtype=complex)

                        for ki_idx, ki in enumerate(range(ki_start, ki_end)):
                            for ka_idx, ka in enumerate(range(ka_start, ka_end)):
                                for kb in range(nkpts):
                                    tmp1_bar_data = tmp1_bar_dset[ki, ka, kb]
                                    y1_2_batch[ki_idx,ka_idx,kb,:,:,:,:] += oe.contract('ki, kalb -> ialb', 
                                                                                       fock_hf[ki,:nocc, :nocc], 
                                                                                       tmp1_bar_data.conj(), 
                                                                                       optimize=path_ialb)

                        for q_idx, q in enumerate(range(ki_start, ki_end)):
                            for w_idx, w in enumerate(range(ka_start, ka_end)):
                                for t in range(nkpts):
                                    r = kconserv[q, w, t]

                                    tmp1_data = tmp1_dset[q, w, r]

                                    c2[r,r,:nocc,:nocc] -= oe.contract('ialb, iajb -> lj', 
                                                                     y1_2_batch[q_idx, w_idx, r,:,:,:,:], 
                                                                     tmp1_data, 
                                                                     optimize=path_lj_2)

                                    c2[w,w,nocc:nocc+nvir,nocc:nocc+nvir] += oe.contract('icjb, iajb -> ac', 
                                                                                        y1_2_batch[q_idx, w_idx, r,:,:,:,:],
                                                                                        tmp1_data, 
                                                                                        optimize=path_ac_2)

                                    c2[t,t,nocc:nocc+nvir,nocc:nocc+nvir] += oe.contract('iajd, iajb -> bd', 
                                                                                        y1_2_batch[q_idx, w_idx, r,:,:,:,:], 
                                                                                        tmp1_data, 
                                                                                        optimize=path_bd_2)

                                    c0_2nd += 4*oe.contract('iajb, iajb ->', 
                                                          y1_2_batch[q_idx, w_idx, r,:,:,:,:], 
                                                          tmp1_data, 
                                                          optimize=path_sum_2).real

                        del y1_2_batch

                c0_2nd /= nkpts

                # Tính y1_3
                y1_3 = numpy.zeros((nkpts,nkpts,nocc,nocc), dtype=complex)

                for w in range(nkpts):
                    for r in range(nkpts):
                        for t in range(nkpts):
                            q = kconserv[w, r, t]
                            tmp1_data = tmp1_dset[q, w, r]
                            tmp1_bar_data = tmp1_bar_dset[q, w, r]
                            y1_3[q,q,:,:] += oe.contract('iajb, kajb -> ki', 
                                                       tmp1_data, 
                                                       tmp1_bar_data.conj(), 
                                                       optimize=path_ki_3)

                for e in range(nkpts):
                    for q in range(nkpts):
                        c2[q,e,:,:nocc] -= lib.einsum('pi, ki -> pk', fock_hf[q,:,:nocc], y1_3[e,q,:,:])

                # Tính y1_4
                y1_4 = numpy.zeros((nkpts,nkpts,nvir,nvir), dtype=complex)

                for q in range(nkpts):
                    for r in range(nkpts):
                        for t in range(nkpts):
                            w = kconserv[q, r, t]
                            tmp1_data = tmp1_dset[q, w, r]
                            tmp1_bar_data = tmp1_bar_dset[q, w, r]
                            y1_4[w,w,:,:] += oe.contract('iajb, icjb -> ac', 
                                                       tmp1_data, 
                                                       tmp1_bar_data.conj(), 
                                                       optimize=path_ac_3)

                for w in range(nkpts):
                    for e in range(nkpts):
                        c2[w,e,:,nocc:] -= lib.einsum('pa, ac -> pc', fock_hf[w,:,nocc:], y1_4[w,e,:,:])
       
    finally:
        print('#################################################')
        print(f'# BẮT ĐẦU DỌN DẸP FILE TẠM TRONG {temp_dir} #')
        print('#################################################')    
        # Xóa file amplitudes tmp_arrays
        try:
            if os.path.exists(h5_file):
                os.remove(h5_file)
                print(f"Đã xóa thành công: {h5_file}")
            else:
                print(f"File không tồn tại, không cần xóa: {h5_file}")
        except (IOError, OSError) as e:
            print(f"Lỗi khi đang xóa file {h5_file}: {e}")
        print('#################################################')
        print(f'# KẾT THÚC DỌN DẸP FILE TẠM                #')
        print('#################################################')
        
    c0 = c0_2nd/nkpts + c0_1st
    
    c1 += numpy.einsum('wwps -> wps', c2)
    print("c1 (sau) = ", c1)

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
    first_BCH = BCH
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
