import numpy as np
from pyscf.pbc import gto, dft, df, scf
import pywannier90

#khai bao tinh the
cell = gto.Cell()
cell.atom = '''
H 1.5 1.5 1
H 1.5 1.5 2
'''
cell.basis = '6-31g'
cell.a = np.eye(3) * 3
cell.gs = [10] * 3
cell.verbose = 5
cell.build()

nk = [2, 1, 1]
kpts = cell.make_kpts(nk)

#chay scf
#kmf = dft.KRKS(cell, abs_kpts)
kmf = scf.KRHF(cell, kpts)
kmf.xc = 'pbe'
ekpt = kmf.run()


#chay kOBMP2=========================================================================
import Khong_chinh_sua_kobmp2_nguyenban_0print as kobmp2
import copy

khf = copy.deepcopy(kmf)

krobmp = kobmp2.OBMP2(khf)
krobmp.second_order = True
krobmp.kernel()
#====================================================================================

# ===== downfold ====================================================================
import krobdf

nocc_inact = [0, 0]
nact = [2, 2] 
num_elec_act = [1, 1]
caslist = [1, 2]


krobact = krobdf.OBMP2(khf)
#robact.mo_coeff = mo_sorted

krobact.nact      = nact[0]
krobact.nocc_act  = num_elec_act[0]

krobact.mo_coeff  = krobmp.mo_coeff
krobact.mo_energy = krobmp.mo_energy
krobact.fock_hf   = krobmp.fock_hf
krobact.second_order = True

#krobact.c0_tot = getattr(krobmp, "c0_tot", None)
#krobact.ene_tot = getattr(krobmp, "ene_tot", None)
#krobact.c1 = getattr(krobmp, "c1", None)

krobact.kernel()
#lay h1_eff_k
h1_eff_k, h2_k = krobact.h1mo_act_eff, krobact.h2mo_act

#======================================================================================


#======================================================================================
#lay U tu pywannier90

kmf.mo_coeff = krobmp.mo_coeff.copy()
kmf.mo_energy = krobmp.mo_energy.copy()

num_wann = 2
keywords = \
'''
exclude_bands : 3,4
begin projections
H:s
end projections
'''

w90 = pywannier90.W90(kmf, cell, nk, num_wann, other_keywords = keywords)
w90.kernel() 

U = w90.U_matrix
#======================================================================================

print("h1_eff_k:\n", h1_eff_k)
print("h1_eff_k shape:\n", h1_eff_k.shape)
print("U matrix shape:\n", U)

# TRUNCATED H1
# ============================================================
# h1_eff_k : (Nk_total, Norb, Norb)
# U        : (Nk_total, Norb, Norb)
# nk       : [Nkx, Nky, Nkz]
# ============================================================

Nk_total, Norb, _ = h1_eff_k.shape
Nkx, Nky, Nkz = nk

assert Nk_total == Nkx * Nky * Nkz, "Nk khong khop voi so k-point"

# ============================================================
# Kep U, dua h1(k) sang Wannier gauge
# h_W(k) = U+(k) h(k) U(k)
# ============================================================

h1_wann_k = np.zeros_like(h1_eff_k, dtype=complex)

for ik in range(Nk_total):
    h1_wann_k[ik] = U[ik].conj().T @ h1_eff_k[ik] @ U[ik]

print("h1_wann_k:\n", h1_wann_k)
print("h1_wann_k.shape:\n", h1_wann_k.shape)
# Shape: (Nk_total, Norb, Norb)

# ============================================================
# Reshape sang kgrid 3D
# (Nk_total, ...) => (Nkx, Nky, Nkz, ...)
# ============================================================

'''
h1_wann_k_grid = h1_wann_k.reshape(
    Nkx, Nky, Nkz, Norb, Norb
)
'''

h1_wann_k_grid = np.zeros((Nkx, Nky, Nkz, Norb, Norb), dtype=complex)

iR = 0
for ix in range(Nkx):
  for iy in range(Nky):
    for iz in range(Nkz):
        h1_wann_k_grid[ix,iy,iz] = h1_wann_k[iR]
        iR += 1


print("h1_wann_k_grid:\n", h1_wann_k_grid)
print("h1_wann_k_grid.shape:\n", h1_wann_k_grid.shape)

# ============================================================
# Fourier transform k => R (no shift)
# ------------------------------------------------------------
# Bloch trong PySCF: exp(+i k.r)
# ==> Hamiltonian real space need exp(-i k.R)
# ==> use FFT (fftn), no use ifftn
# ============================================================

Nk = Nkx * Nky * Nkz

# shift cell g?c v? (0,0,0)
h1k_shift = np.fft.ifftshift(h1_wann_k_grid, axes=(0,1,2))

# fourier
h1_wann_R = np.fft.fftn(h1k_shift, axes=(0,1,2)) / Nk

print("h1_wann_R:\n", h1_wann_R)
print("h1_wann_R.shape:\n", h1_wann_R.shape)

# h1_wann_R : (Nkx, Nky, Nkz, Norb, Norb)
# ============================================================

# Truncated===================================================
def truncate_h_real_space(hR, R_cut):
    """
    Truncate real-space one-body Hamiltonian h1(R) by distance |R| <= R_cut.

    Parameters
    ----------
    hR : ndarray
        Real-space Hamiltonian with shape (Nkx, Nky, Nkz, Norb, Norb)

    lattice_vectors : tuple of ndarray
        (a1, a2, a3), each shape (3,)
        Primitive lattice vectors in real space

    R_cut : float
        Cutoff radius in real-space (same unit as lattice vectors)

    Returns
    -------
    hR_trunc : ndarray
        Truncated Hamiltonian with same shape as hR
    """

    # 1. Unpack sizes
    Nkx, Nky, Nkz, Norb, _ = hR.shape
    print("Nkx, Nky, Nkz, Norb", Nkx, Nky, Nkz, Norb)

    # 2. Prepare output
    h_R_trunc = np.zeros_like(hR, dtype=hR.dtype)
    h_R_trunc_dict = {}

    # 3. Loop over R-grid indices
    for ix in range(Nkx):
        for iy in range(Nky):
            for iz in range(Nkz):

                # 3.1 Unwrap index -> lattice integer n
                nx = ix if ix <= Nkx // 2 else ix - Nkx
                ny = iy if iy <= Nky // 2 else iy - Nky
                nz = iz if iz <= Nkz // 2 else iz - Nkz

                print("nx, ny, nz", nx, ny, nz)
                
                R_norm = np.sqrt(nx**2 + ny**2 + nz**2)
                                
                print("n_norm", R_norm)
                # 3.2 Truncation condition
                if R_norm <= R_cut:
                    h_R_trunc[ix, iy, iz, :, :] = hR[ix, iy, iz, :, :]
                    
                    R = (nx, ny, nz) 
                    h_R_trunc_dict[R] = hR[ix, iy, iz, :, :].copy()

                # else: implicitly zero (already zeroed)
    return h_R_trunc, h_R_trunc_dict

# Truncate within nearest neighbors
R_cut = 1.01  # keep onsite + NN

h1_R_trunc, h1_R_trunc_dict = truncate_h_real_space(
    h1_wann_R,
    R_cut=R_cut
)

print("h1_R_trunc:\n", h1_R_trunc)
print("h1_R_trunc.shape:\n", h1_R_trunc.shape)

print("h1_R_trunc_dict:\n", h1_R_trunc_dict)

# Build h1 big
def build_h1_big(h1_R, cluster_R, n_wann, tol=1e-12):
    """
    Build real-space cluster one-body Hamiltonian H1_big.

    Parameters
    ----------
    h1_R : dict
        Dictionary mapping R = (Rx, Ry, Rz) -> h1(R),
        where h1(R) is an (n_wann, n_wann) complex matrix.
        Only truncated R need to be provided.

    cluster_R : list of tuple
        List of lattice vectors R_i = (n1, n2, n3) defining the cluster.

    n_wann : int
        Number of Wannier orbitals per unit cell.

    tol : float
        Numerical tolerance for Hermiticity check.

    Returns
    -------
    h1_big : ndarray
        Cluster Hamiltonian of shape (n_cell*n_wann, n_cell*n_wann)
    """

    n_cell = len(cluster_R)
    dim = n_cell * n_wann
    h1_big = np.zeros((dim, dim))
    
    #build h1_big
    for i, Ri in enumerate(cluster_R):
        for j, Rj in enumerate(cluster_R):

            # deltaR = Ri - Rj
            dR = (Ri[0] - Rj[0],
                  Ri[1] - Rj[1],
                  Ri[2] - Rj[2])

            # Determine block h1(dR)
            if dR in h1_R:
                print("---------------------------------------------")
                print("Ri:\n", Ri)
                print("Rj:\n", Rj)
                print("dR:\n", dR)
                block = h1_R[dR]
                print("block:\n", block)
                
            else:
                continue  # truncated zero block

            # Insert block
            I0 = i * n_wann
            J0 = j * n_wann
            h1_big[I0:I0+n_wann, J0:J0+n_wann] = block
            
            # If missing -dR, generate Hermitian conjugate
            dR_minus = (-dR[0], -dR[1], -dR[2])
            if dR_minus not in h1_R and i != j: # add i != j for safety
                block_minus = block.conj().T
                print("dR_minus:\n", dR_minus)
                print("block_minus:\n", block_minus)
                I1 = j * n_wann
                J1 = i * n_wann
                h1_big[I1:I1+n_wann, J1:J1+n_wann] = block_minus
                
    # Optional: enforce Hermiticity numerically
    if np.linalg.norm(h1_big - h1_big.conj().T) > tol:
        print("Warning: H1_big is not perfectly Hermitian (numerical noise?)")

    return h1_big

cluster_R = [
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]

h1_big = build_h1_big(h1_R_trunc_dict, cluster_R, num_wann) #num_wann = 2

print("h1_big:\n", h1_big)  # (14, 14)

print("h1_big.shape:\n", h1_big.shape)  # (14, 14)

# H2
# ============================================================
from pyscf.pbc.lib import kpts_helper
kconserv = kpts_helper.get_kconserv(kmf.cell, kpts)
fao2mo = kmf.with_df.ao2mo

print("h2_k:\n", h2_k)
print("h2_k.shape:\n", h2_k.shape)

# K?p U
N1, N2, N3, N4, n, _, _, _ = h2_k.shape
h2_wann_k = np.zeros_like(h2_k, dtype=complex)

for kp in range(N1):
    for kq in range(N2):
        for kr in range(N3):
            ks = kconserv[kp, kq, kr]# k-conservation

            Up = U[kp]
            Uq = U[kq]
            Ur = U[kr]
            Us = U[ks]

            eri = h2_k[kp, kq, kr, ks]  # (a b c d)

            h2_wann_k[kp, kq, kr, ks] = np.einsum(
                'ai,bj,abcd,ck,dl->ijkl',
                Up.conj(), Uq.conj(),
                eri,
                Ur, Us,
                optimize=True
            )

print("h2_wann_k:\n", h2_wann_k)
print("h2_wann_k.shape:\n", h2_wann_k.shape)

# Fourier sang real space
h2_R = np.fft.fftn(
    h2_wann_k[..., :, :, :, :],
    axes=(0,1,2)
) / (Nkx * Nky * Nkz)

print("h2_R:\n", h2_R)
print("h2_R.shape:\n", h2_R.shape)

'''
R_list = []
for ix in range(Nkx):
    nx = ix if ix <= Nkx//2 else ix - Nkx
    for iy in range(Nky):
        ny = iy if iy <= Nky//2 else iy - Nky
        for iz in range(Nkz):
            nz = iz if iz <= Nkz//2 else iz - Nkz
            R_list.append((nx, ny, nz))

print("R_list:\n", R_list)
'''

def flat_R_to_xyz(iR, Nkx, Nky, Nkz):
    """
    Convert flat R index (0 ... Nkx*Nky*Nkz-1)
    into (x,y,z) integer lattice coordinate
    relative to centered cell.
    
    iR = ix.(Nky.Nkz) + iy.Nkz + iz
    """

    # ---- unflatten ----
    iz = iR % Nkz
    tmp = iR // Nkz

    iy = tmp % Nky
    ix = tmp // Nky

    # ---- shift to centered lattice ----
    def shift_center(i, N):
        return (i + N//2) % N - N//2

    ix = shift_center(ix, Nkx)
    iy = shift_center(iy, Nky)
    iz = shift_center(iz, Nkz)

    return [ix, iy, iz]

h2_blocks = {}

for R1 in range(N1):
    for R2 in range(N2):
        for R3 in range(N3):

            vec1 = flat_R_to_xyz(R1, Nkx, Nky, Nkz)
            vec2 = flat_R_to_xyz(R2, Nkx, Nky, Nkz)
            vec3 = flat_R_to_xyz(R3, Nkx, Nky, Nkz)

            vec4 = [
                (vec1[0] + vec2[0] - vec3[0]),
                (vec1[1] + vec2[1] - vec3[1]),
                (vec1[2] + vec2[2] - vec3[2]),
            ]
            
            vec4 = [
                ((vec4[0] + Nkx//2) % Nkx) - Nkx//2,
                ((vec4[1] + Nky//2) % Nky) - Nky//2,
                ((vec4[2] + Nkz//2) % Nkz) - Nkz//2,
            ]
            
            n1 = np.linalg.norm(vec1)
            n2 = np.linalg.norm(vec2)
            n3 = np.linalg.norm(vec3)
            n4 = np.linalg.norm(vec4)

            '''
            print("vec1", vec1)
            print("vec2", vec2)
            print("vec3", vec3)
            print("vec4", vec4)

            
            print("n1", n1)
            print("n2", n2)
            print("n3", n3)
            print("n4", n4)
            '''

            # centered => index
            ix4 = vec4[0] % Nkx
            iy4 = vec4[1] % Nky
            iz4 = vec4[2] % Nkz
            
            # flatten 
            R4 = ix4*(Nky*Nkz) + iy4*Nkz + iz4
               
            '''       
            print("R1", R1)
            print("R2", R2)
            print("R3", R3)
            print("R4", R4)
            '''
            
            if max(n1,n2,n3,n4) <= R_cut:
              
                block = h2_R[R1,R2,R3,R4]
                # only save none zero block
                if not np.allclose(block, 0):
                    h2_blocks[(R1, R2, R3, R4)] = block.copy()
                    
                    #print("h2_blocks", h2_blocks)
                    
np.set_printoptions(precision=6, linewidth=120)
for key, val in h2_blocks.items():
    print(f"\nBlock {key}")
    print(val)

# ============================================================
# Build H2_big from truncated h2_blocks
# ============================================================

ncell = 7
Ntot = num_wann*ncell

print("Ntot    =", Ntot)
print("Allocating H2_big ...")

H2_big = np.zeros((Ntot, Ntot, Ntot, Ntot), dtype=complex)

for (R1, R2, R3, R4), block in h2_blocks.items():

    base1 = R1 * num_wann
    base2 = R2 * num_wann
    base3 = R3 * num_wann
    base4 = R4 * num_wann

    for a in range(num_wann):
        p = base1 + a
        for b in range(num_wann):
            q = base2 + b
            for c in range(num_wann):
                r = base3 + c
                for d in range(num_wann):
                    s = base4 + d

                    H2_big[p, q, r, s] += block[a, b, c, d]

print("H2_big build done.")
#print("H2_big =", H2_big)
print("H2_big shape =", H2_big.shape)

