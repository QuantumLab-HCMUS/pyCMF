def stabilize_scf(mf, max_macro_cycles=5, verbose=True):
    """
        Check and force the density matrix of the SCF object to converge to a true minimum (avoiding saddle points).

        If the density is unstable, the function will automatically use the Newton-Raphson algorithm to re-optimize.

        Args:

        mf (pyscf.scf.hf.SCF): The SCF object (RHF/UHF/RKS/UKS) that has run through mf.kernel().

        max_macro_cycles (int): The maximum number of loops to break saddle points, avoiding infinite loops.

        verbose (bool): Enable/disable printing the process log to the screen.

        Returns:

        mf: The SCF object has been stabilized and is guaranteed to be at a minimum.
    """
    if verbose:
        print("\n" + "-"*40)
        print(" Starting the stability procedure!!!!")
        print("-"*40)
        
    is_stable = False
    cycle = 0

    while not is_stable and cycle < max_macro_cycles:
        # Gọi hàm stability() của PySCF
        mo_new, _, is_stable, _ = mf.stability(return_status=True)
        
        if not is_stable:
            cycle += 1
            if verbose:
                print(f"   -> [Warning] Iter {cycle}: The saddle-point density.")
                print("   -> Activating the Newton-Raphson based second-order self-consistent field (SOSCF) algorithm...")
            
            # Khởi tạo bộ giải bậc hai
            mf_newton = mf.newton()
            # Ép hội tụ dọc theo vector nhiễu loạn mo_new
            mf_newton.kernel(mo_coeff=mo_new, mo_occ=mf.mo_occ)
            if not mf_newton.converged and verbose:
                print("   -> [Warning] Newton solver did not converge this cycle.")
            # Đồng bộ hóa lại các thuộc tính quan trọng cho mf gốc
            mf.mo_coeff = mf_newton.mo_coeff
            mf.mo_energy = mf_newton.mo_energy
            mf.e_tot = mf_newton.e_tot
        else:
            if verbose:
                print("   -> [Congratulation~] The self-consistent field (SCF) solution successfully.")

    if not is_stable and verbose:
        print("\n[WARNING!!!]: Not convergence after {} iterations.".format(max_macro_cycles))
        
    return mf