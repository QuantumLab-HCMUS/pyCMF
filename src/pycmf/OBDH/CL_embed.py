import numpy as np
import scipy.linalg as la


def concentric_localization(C_vir_eff, S, F, active_aos, n_shells=1, tol=1e-5,
                             verbose=True):
    """
    Thu gọn không gian ảo bằng Concentric Localization (CL).

    Tham số
    -------
    C_vir_eff : ndarray (nao, n_vir)
        Ma trận MO virtual sau khi đã lọc orbital bị đẩy bởi mu*P_B.
    S         : ndarray (nao, nao)   Overlap matrix.
    F         : ndarray (nao, nao)   Fock matrix (spin-specific).
    active_aos: list[int]            Chỉ số AO thuộc fragment A.
    n_shells  : int
        Số CL shells muốn giữ (1 = chỉ shell 0, 2 = shell 0+1, ...).
        Dùng n_shells=-1 để lấy TẤT CẢ shells (= full virtual space,
        tương đương không dùng CL).
    tol       : float   Ngưỡng singular value để phân biệt span/kernel.
    verbose   : bool    In thông tin diagnostic về từng shell.

    Trả về
    ------
    C_vir_CL : ndarray (nao, n_vir_CL)
        Không gian virtual đã thu gọn (chưa pseudo-canonicalize).

    Thông tin shells
    ----------------
    Shell 0  : AO-projected span of fragment A → kích thước ≤ n_A_aos
    Shell k  : span của Coupling(C_{k-1}, C_ker) qua Fock operator
    Max shells: dừng tự động khi kernel cạn (C_ker.shape[1] == 0)
    """
    n_vir = C_vir_eff.shape[1]
    n_A_aos = len(active_aos)

    # ------------------------------------------------------------------
    #Chuyển sang hệ cơ sở trực giao (S^1/2)
    # ------------------------------------------------------------------
    S_half = la.fractional_matrix_power(S, 0.5)
    C_bar_vir = S_half @ C_vir_eff
    C_bar_vir_A = C_bar_vir[active_aos, :]   # shape: (n_A_aos, n_vir)

    # ------------------------------------------------------------------
    #Shell 0 — SVD của C_bar_vir_A
    # QUAN TRỌNG: full_matrices=True để giữ toàn bộ kernel
    # full_matrices=False sẽ cắt mất (n_vir - n_A_aos) kernel vectors!
    # ------------------------------------------------------------------
    U, Sigma, V_T = la.svd(C_bar_vir_A, full_matrices=True)
    # V_T shape: (n_vir, n_vir) — đầy đủ

    idx_span = Sigma > tol          # chỉ Sigma[:min(n_A_aos,n_vir)] có nghĩa
    # Padding: Sigma từ SVD full có thể ngắn hơn n_vir
    n_sig = len(Sigma)
    span_mask = np.zeros(n_vir, dtype=bool)
    span_mask[:n_sig] = (Sigma > tol)

    n_span_0 = int(span_mask.sum())
    n_ker_0  = n_vir - n_span_0

    V_span = V_T[span_mask, :].T      # (n_vir, n_span_0)
    V_ker  = V_T[~span_mask, :].T     # (n_vir, n_ker_0)

    C_0   = C_vir_eff @ V_span        # Shell 0
    C_ker = C_vir_eff @ V_ker         # Kernel còn lại

    if verbose:
        print(f"   [CL] n_vir_eff = {n_vir},  n_A_aos = {n_A_aos}")
        print(f"   [CL] Shell 0: size = {n_span_0}  | kernel remaining = {n_ker_0}")

    CL_shells = [C_0]
    C_n = C_0

    # ------------------------------------------------------------------
    #Mở rộng các shells tiếp theo qua Fock coupling
    # n_shells=1  → chỉ shell 0, không expand
    # n_shells=2  → shell 0 + shell 1
    # n_shells=-1 → tất cả shells cho đến khi kernel cạn
    # ------------------------------------------------------------------
    shell_sizes = [n_span_0]
    max_possible_shells = 1

    # Số shells cần expand thêm (ngoài shell 0)
    n_extra = (n_vir) if n_shells == -1 else max(n_shells - 1, 0)

    for i in range(n_extra):
        if C_ker.shape[1] == 0:
            break   # kernel cạn kiệt

        Coupling = C_n.T.conj() @ F @ C_ker     # shape: (n_prev_span, n_ker)

        # full_matrices=True để kernel tiếp theo đầy đủ
        U_n, Sigma_n, V_T_n = la.svd(Coupling, full_matrices=True)
        n_sig_n   = len(Sigma_n)
        n_ker_cur = C_ker.shape[1]

        span_mask_n = np.zeros(n_ker_cur, dtype=bool)
        span_mask_n[:n_sig_n] = (Sigma_n > tol)

        n_span_n = int(span_mask_n.sum())
        n_ker_n  = n_ker_cur - n_span_n

        if n_span_n == 0:
            break   # không còn coupling có nghĩa

        V_span_n = V_T_n[span_mask_n, :].T    # (n_ker_cur, n_span_n)
        V_ker_n  = V_T_n[~span_mask_n, :].T   # (n_ker_cur, n_ker_n)

        C_next = C_ker @ V_span_n
        C_ker  = C_ker @ V_ker_n

        CL_shells.append(C_next)
        C_n = C_next

        max_possible_shells += 1
        shell_sizes.append(n_span_n)

        if verbose:
            cumulative = sum(shell_sizes)
            print(f"   [CL] Shell {i+1}: size = {n_span_n}"
                  f"  | cumulative = {cumulative}/{n_vir}"
                  f"  | kernel remaining = {n_ker_n}")

    # ------------------------------------------------------------------
    # Thông tin tổng kết
    # ------------------------------------------------------------------
    n_kept   = sum(c.shape[1] for c in CL_shells)
    n_shells_used = len(CL_shells)

    if verbose:
        print(f"   [CL] ── Tóm tắt ──────────────────────────────────────")
        print(f"   [CL]   Shells được dùng  : {n_shells_used}  "
              f"(max có thể = {max_possible_shells})")
        print(f"   [CL]   Virtual giữ lại   : {n_kept} / {n_vir}  "
              f"({100*n_kept/n_vir:.1f}%)")
        if C_ker.shape[1] > 0 and n_shells != -1:
            print(f"   [CL]   Virtual bị loại   : {n_vir - n_kept}"
                  f"  (kernel còn lại = {C_ker.shape[1]})")
        print(f"   [CL] ──────────────────────────────────────────────────")

    C_vir_CL = np.hstack(CL_shells)
    return C_vir_CL


def cl_shell_analysis(C_vir_eff, S, F, active_aos, tol=1e-5):
    """
    Phân tích CL để biết số shells tối đa và kích thước từng shell
    MÀ KHÔNG cần chạy OBMP2. Gọi hàm này trước khi chọn n_shells.

    Cách dùng:
        cl_shell_analysis(C_vir_eff, S_mat, F_mat[0], active_aos)
    """
    print("\n" + "="*55)
    print("  CL SHELL ANALYSIS (dry run)")
    print("="*55)
    concentric_localization(C_vir_eff, S, F, active_aos,
                            n_shells=-1, tol=tol, verbose=True)
    print("="*55 + "\n")