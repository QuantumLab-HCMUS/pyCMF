import numpy as np
import os

import pyscf
from pyscf import cc, gto, scf, mcscf, fci
from pyscf.mp import UOBMP2_faster, UOBMP2_downfold, OBMP2_faster, obmp2_faster, ROBMP2_downfold
from pyscf.fci import direct_uhf, direct_spin1, sci_uhf
from pyscf.mrpt import nevpt2
from pyscf.tools import mo_mapping
from pyscf import lib
from pyscf import lo
from pyscf.lo import nao
import psutil

import scipy
import scipy.linalg

import sys
sys.path.append('/home/tnthien/MyLibrary/Basis/')

import importlib
BASIS = "Molecule_Input_"+os.getenv('BASIS')
print(f"Basis: {BASIS}")
Molecule_Input = importlib.import_module(BASIS)


#import Molecule_Input_ccpvdz as Molecule_Input

import math

lib.param.MAX_MEMORY=50000
print("lib.param.MAX_MEMORY = ", lib.param.MAX_MEMORY)
print("available memory = ",psutil.virtual_memory().available / 1024**3)

#==============================Mute Printing==============================

def set_quiet(obj):
    if hasattr(obj, "verbose"):
        obj.verbose = 0
    if hasattr(obj, "stdout"):
        obj.stdout = None
    return obj

#==============================Mute Printing==============================
"""
mole        = 'C2'
arrange     = 'triplet'
ncore       = 0
Jmol        = False
act_orb     = 'full'
"""


mole            = os.getenv('MOLE')
ncore           = int(os.getenv('NCORE'))
act_orb         = os.getenv('ACT_ORB')
arrange         = os.getenv('ARRANGE')


print(f"#====================")
print(f"{'Mole: ':<15} {mole}")
print(f"{'Ncore: ':<15} {ncore}")
print(f"{'Act Orb: ':<15} {act_orb}")
print(f"{'Arrange: ':<15} {arrange}")
print(f"#====================")

#==============================Khai báo input==============================
cutoff      = 1e-2 

molecule = Molecule_Input.molecule_input(molecule = mole, arrange = arrange, ncore = ncore, act_orb = act_orb)
mol = molecule.mol
basis = molecule.basis


dR = 0.1
start = molecule.R
end = molecule.R
R_val    = np.arange(start, end + dR, dR)
print('\n')
print(f"Basis: {basis}")
print(f"Full Orbitals: {mol.nao_nr()}")
print(f"Cutoff: {cutoff}")
print('\n')
#==============================Khai báo input==============================


def Plot_Data(R):

    #==============================Thông tin đầu vào==============================
    
    molecule = Molecule_Input.molecule_input(molecule = mole, arrange = arrange, R=R, ncore = ncore, act_orb = act_orb)
    basis = molecule.basis

    mol = molecule.mol
    nocc_inact = molecule.nocc_inact
    nact = molecule.nact
    num_particles = molecule.num_particles
    caslist_a = molecule.caslist_a
    caslist_b = molecule.caslist_b
    caslist = molecule.caslist
    active_space = molecule.active_space
    num_orbitals = molecule.num_orbitals
    angle = molecule.angle

    
    mol.max_memory=10000
    print(f'R: {R}')
    print("mol.max_memory       =", mol.max_memory)

    #==============================RHF==============================

    myuhf = pyscf.scf.UHF(mol)
    e_uhf=myuhf.kernel()

    #==============================RHF==============================




    #==============================OBMP2 full-space==============================

    uobmp = UOBMP2_faster(myuhf)
    uobmp.second_order = True
    uobmp.kernel()
    e_uobmp2 = getattr(uobmp, "ene_tot", None)
    import numpy as np 
    S = mol.intor("int1e_ovlp")
    eigval, eigvec = np.linalg.eigh(S)

    S12 = eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T


    """Cho Unrestricted
    for j in range(mol.nao_nr()):
    print(f"\n===== MO {j} =====  energy: {robmp.mo_energy[0][j]:12.6f} ===== "
          f"normalization: {robmp.mo_coeff[0][:,j].T @ S @ robmp.mo_coeff[0][:,j]}")
    """
    for s in [0, 1]:
        print(f"\n===== SPIN {s} =====" )
        print("#==================================================")
        print("\n")
        print("AO contribution")
        for j in range(mol.nao_nr()):
            print(f"\n===== MO {j} =====  energy: {uobmp.mo_energy[s][j]:12.6f} ===== normalization: {uobmp.mo_coeff[s][:,j].T @ S @ uobmp.mo_coeff[s][:,j]}" )
            prob = (S12 @ uobmp.mo_coeff[s][:,j])**2
            prob /= prob.sum()
            print(f"orth: {sum(prob)}")
            C = uobmp.mo_coeff[s][:,j]
            contrbS = C*(S@C)
            print(f"norm: {sum(contrbS)}")
            for ao, coeff in enumerate(uobmp.mo_coeff[s][:,j]):
                if abs(coeff) < 1e-15:
                    coeff = 0
                else: coeff
                if prob[ao] > cutoff:
                    print(f"{ao:3d}: {mol.ao_labels()[ao]:15s}: {coeff:12.6f};  prob: {prob[ao]:12.6f}; contrb: {contrbS[ao]:12.6f}")
        print("#==================================================")
        print("\n")

    #==============================OBMP2 full-space==============================

    #==============================Vẽ Orbital==============================
    import numpy as np
    import os

    folder = f"/home/tnthien/Work/Dfold_ANY/SQD/Run/Molden/{mole}/{arrange}/{basis}"
    os.makedirs(folder, exist_ok=True)

    """
    molecule = os.path.join(folder, f"{mole}_{arrange}_{basis}_R={np.round(R,1)}.molden")

    pyscf.tools.molden.from_mo(
        mol,
        molecule,
        robmp.mo_coeff,
        ene=robmp.mo_energy,
        occ=robmp.mo_occ
    )
    """

    mole_alpha = os.path.join(folder, f"{mole}_{arrange}_{basis}_R={np.round(R,1)}_alpha.molden")
    mole_beta = os.path.join(folder, f"{mole}_{arrange}_{basis}_R={np.round(R,1)}_beta.molden")

    
    pyscf.tools.molden.from_mo(
        mol,
        mole_alpha,
        uobmp.mo_coeff[0],
        ene=uobmp.mo_energy[0],
        occ=uobmp.mo_occ[0],
        spin='Alpha'
    )

    pyscf.tools.molden.from_mo(
        mol,
        mole_beta,
        uobmp.mo_coeff[1],
        ene=uobmp.mo_energy[1],
        occ=uobmp.mo_occ[1],
        spin='Beta'
    )

    """
    mole_alpha = os.path.join(folder, f"{mole}_{arrange}_{basis}_R={R}_angle={angle}_alpha.molden")
    mole_beta = os.path.join(folder, f"{mole}_{arrange}_{basis}_R={R}_angle={angle}_beta.molden")

    
    pyscf.tools.molden.from_mo(
        mol,
        mole_alpha,
        robmp.mo_coeff,
        ene=robmp.mo_energy,
        occ=robmp.mo_occ,
        spin='Alpha'
    )

    pyscf.tools.molden.from_mo(
        mol,
        mole_beta,
        robmp.mo_coeff,
        ene=robmp.mo_energy,
        occ=robmp.mo_occ,
        spin='Beta'
    )
    """
    #==============================Vẽ Orbital==============================

    """
    #==============================Sort trong biểu diễn MO theo caslist (Restricted)==============================

    mycas = mcscf.CASCI(myrhf, ncas=num_orbitals, nelecas=sum(num_particles))
    mo = mycas.sort_mo(active_space, base=0)
    e_casci = mycas.run().e_tot

    hcore, nuclear_repulsion_energy = mycas.get_h1cas(mo)
    eri = pyscf.ao2mo.restore(1, mycas.get_h2cas(mo), num_orbitals)

    #==============================Sort trong biểu diễn MO theo caslist (Restrictded)==============================


    #==============================Sort trong biểu diễn MO theo caslist (Unrestricted)==============================
    
    #myucas = mcscf.UCASCI(myuhf, ncas=nact[0], nelecas=num_particles)
    #e_ucasci = myucas.run().e_tot
    
    #myucasscf = mcscf.UCASSCF(myuhf, ncas=nact[0], nelecas=num_particles)
    #e_ucasci = myucasscf.run().e_tot
    
    mo_sorted = mcscf.sort_mo(mycas, robmp.mo_coeff, caslist_a)
    #mo_sorted = mcscf.sort_mo(myucasscf, robmp.mo_coeff, caslist_a)
    
    
    

    #==============================Sort trong biểu diễn MO theo caslist (Unrestrictded)==============================




    #==============================RDfold==============================

    robact = ROBMP2_downfold(myrhf, nact=nact[0], nocc_act=num_particles[0])
    robact.mo_coeff = mo_sorted
    robact.mo_energy = robmp.mo_energy
    robact.c0_tot = getattr(robmp, "c0_tot", None)
    robact.ene_tot = getattr(robmp, "ene_tot", None)
    robact.fock_hf = getattr(robmp, "fock_hf", None)
    robact.c1 = getattr(robmp, "c1", None)
    robact.second_order = True

    #----------Sắp lại tmp1/tmp1_bar cho đúng thứ tự MO----------
    robact.tmp1 = robmp.tmp1
    robact.tmp1_bar = robmp.tmp1_bar
    #----------Sắp lại tmp1/tmp1_bar cho đúng thứ tự MO----------

    robact.kernel()

    #==============================RDfold==============================


    #==============================Lấy kết quả downfold==============================

    h1mo_act_eff = robact.h1mo_act_eff          
    h2mo_act = robact.h2mo_act                  
    ene_inact = robact.ene_inact                
     


    h1=h1mo_act_eff
    h2=h2mo_act

    #==============================Lấy kết quả downfold==============================



    #==============================FCI==============================

    #cis_full = fci.FCI(myuhf)
    #E_CCSDT, _ = cis_full.kernel()

    #==============================FCI==============================



    #==============================Dfold_FCI==============================
    
    cis = direct_spin1.FCI()
    cis.nroots = 1
    cis.max_memory=10000
    e_dfold_fci, _ = cis.kernel(h1, h2, norb, (nalpha,nbeta))
    e_dfold_fci = e_dfold_fci + ene_inact
    
    print(f"E_Dfold_FCI: {e_dfold_fci}")

    #==============================Dfolf_FCI==============================



    import ffsim
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import CPhaseGate, XXPlusYYGate, XGate
    from qiskit.primitives import BitArray


    #==============================QuantumRegister: Tạo qubit==============================

    qubits = QuantumRegister(2 * num_orbitals, name="q")                                    #Tạo qubit

    init_state = QuantumCircuit(qubits)                                                     #Tạo mạch lượng tử cho qubit
    init_state.append(ffsim.qiskit.PrepareHartreeFockJW(num_orbitals, nelec), qubits)       #Gắn thông tin qubit lên mạch lượng tử
    init_state.measure_all()                                                                #Đo tất cả mạch lượng tử

    #==============================QuantumRegister: Tạo qubit==============================


    #==============================Mô phỏng mạch lượng tử==============================

    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    backend = FakeSherbrooke()
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    #----------Sơ đồ sắp xếp trong mạch lượng tử (cách xếp không phụ thuộc vào phân tử)----------
    spin_a_layout = [0, 14, 18, 19, 20, 33, 39, 40, 41, 53, 60, 61, 62, 72, 81, 82]
    spin_b_layout = [2, 3, 4, 15, 22, 23, 24, 34, 43, 44, 45, 54, 64, 65, 66, 73]
    initial_layout = spin_a_layout + spin_b_layout
    #----------Sơ đồ sắp xếp trong mạch lượng tử (cách xếp không phụ thuộc vào phân tử)----------

    #==============================Mô phỏng mạch lượng tử==============================



    #==============================Đăng ký mạch lượng tử==============================

    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation

    pass_manager = PassManager([
        Optimize1qGates(),
        CommutativeCancellation()
    ])


    isa_circuit = pass_manager.run(init_state)

    #==============================Đăng ký mạch lượng tử==============================

    #==============================SQD==============================

    import numpy as np
    from qiskit_addon_sqd.counts import generate_bit_array_uniform

    rng = np.random.default_rng(25)                                                     #default_rng(25): Cố định bộ tạo số ngẫu nhiên seed=25
    bit_array = generate_bit_array_uniform(100_000, 2 * num_orbitals, rand_seed=rng)    #Tạo dãy bit: generate_bit_array_uniform(số mẫu, độ dài mẫu, bộ sinh số)


    from functools import partial
    import qiskit_addon_sqd
    from qiskit_addon_sqd import fermion, ufermion
    from qiskit_addon_sqd.ufermion import SCIResult, diagonalize_fermionic_hamiltonian, solve_sci_batch
    from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian, solve_sci_batch

    #----------Điều kiện gội tụ----------#
    energy_tol = 1e-5
    occupancies_tol = 1e-5
    max_iterations = 100
    num_batches = 1                         #Lấy mẫu theo từng batch, num_batches=1: Lấy toàn bộ số mẫu trong 1 lần duy nhất
    samples_per_batch = 400                 #Số mẫu mỗi batch
    symmetrize_spin = True                 #Điều kiện đối xứng spin
    carryover_threshold = 1e-5              #Ngưỡng giữ lại cho vòng lặp kế tiế
    max_cycle = 500                         #Số vòng lặp tối đa khi chéo hóa
    #----------Điều kiện gội tụ----------#


    #----------SQD----------
    result = fermion.diagonalize_fermionic_hamiltonian(
        hcore,
        eri,
        bit_array,
        samples_per_batch=samples_per_batch,
        norb=num_orbitals,
        nelec=nelec,
        num_batches=num_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        max_iterations=max_iterations,
        #sci_solver=sci_solver,
        symmetrize_spin=symmetrize_spin,
        carryover_threshold=carryover_threshold,
        #callback=callback,
        seed=rng,
    )

    e_sqd = result.energy + nuclear_repulsion_energy
    print(f"SQD: {e_sqd}")
    #----------SQD----------




    #----------Dfold_SQD----------
    result_dfold = fermion.diagonalize_fermionic_hamiltonian(
        h1,
        h2,
        bit_array,
        samples_per_batch=samples_per_batch,
        norb=num_orbitals,
        nelec=nelec,
        num_batches=num_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        max_iterations=max_iterations,
        #sci_solver=sci_solver,
        symmetrize_spin=symmetrize_spin,
        carryover_threshold=carryover_threshold,
        #callback=callback,
        seed=rng,
    )

    e_dfold_sqd = result_dfold.energy + ene_inact
    print(f"Dfold SQD: {e_dfold_sqd}")
    
    e_ccsdt = 0
    #----------Dfold_SQD----------
    print("\n")
    print("#==================================================")
    """
    e_casci=0
    e_dfold_fci = 0 
    e_sqd = 0
    e_dfold_sqd = 0
    e_ccsdt = 0

    return e_uhf, e_casci, e_uobmp2, e_dfold_fci, e_sqd, e_dfold_sqd, e_ccsdt


    #==============================SQD==============================

#==============================Plot==============================

#--------------------R (Trục x)--------------------


#start=float(os.getenv('start'))
#end=float(os.getenv('end'))
#dR=float(os.getenv('dR'))

#H6_R_val    = np.arange(start, end, dR)
#--------------------R (Trục x)--------------------

#--------------------Năng lượng (Trục y)--------------------
E_RHF, E_CASCI, E_OBMP2, E_Dfold_FCI, E_SQD, E_Dfold_SQD, E_CCSDT = ([] for _ in range(7))
#--------------------Năng lượng (Trục y)--------------------


#--------------------Kéo giãn--------------------
for R in R_val:
    Data = Plot_Data(R)
    E_RHF.append(Data[0])
    E_CASCI.append(Data[1])
    E_OBMP2.append(Data[2])
    E_Dfold_FCI.append(Data[3])
    E_SQD.append(Data[4])
    E_Dfold_SQD.append(Data[5])
    E_CCSDT.append(Data[6])
#--------------------Kéo giãn--------------------

#--------------------Định dạng array--------------------
E_RHF       = np.array(E_RHF)
E_CASCI    = np.array(E_CASCI)
E_OBMP2    = np.array(E_OBMP2)
E_Dfold_FCI = np.array(E_Dfold_FCI)
E_SQD       = np.array(E_SQD)
E_Dfold_SQD = np.array(E_Dfold_SQD)
E_CCSDT = np.array(E_CCSDT) 
#--------------------Định dạng array--------------------

"""
#--------------------Lưu File--------------------
import pathlib
from pathlib import Path

save_dir = Path(f"/home/tnthien/Work/Dfold_ANY/SQD/Run/Plot/{mole}/PlotData/{arrange}/{basis}")

save_dir.mkdir(parents=True, exist_ok=True)

filename = f"{mole}_{arrange}_{basis}_{nact[0]}o{caslist_a}_[dR={dR}]_RPlot_data.py"
file_path = save_dir / filename


#--------------------Lưu File--------------------


#--------------------Ghi dữ liệu--------------------
with open(file_path, "w") as f:
    f.write("import numpy as np\n\n")
    f.write(f"R_val = np.array({R_val.tolist()})\n")
    f.write(f"E_RHF = np.array({E_RHF.tolist()})\n")
    f.write(f"E_CASCI = np.array({E_CASCI.tolist()})\n")
    f.write(f"E_OBMP2 = np.array({E_OBMP2.tolist()})\n")
    f.write(f"E_Dfold_FCI = np.array({E_Dfold_FCI.tolist()})\n")
    f.write(f"E_SQD = np.array({E_SQD.tolist()})\n")
    f.write(f"E_Dfold_SQD = np.array({E_Dfold_SQD.tolist()})\n\n")
    f.write(f"E_CCSDT = np.array({E_CCSDT.tolist()})\n\n")
    f.write("R_val = np.round(R_val, 2)\n\n")
    f.write("import matplotlib.pyplot as plt\n\n")
    f.write("fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 4),gridspec_kw={'height_ratios': [5, 1]})\n\n")
    f.write("linewidth=0.7\n\n")
    f.write("#'orange', 'purple', 'brown', 'pink', 'lime', 'teal', 'navy', 'gold', 'crimson', 'indigo', 'turquoise'\n\n")
    f.write("axs[0].plot(R_val, E_RHF, linestyle='-', color='purple', label='RHF', linewidth=linewidth)\n")
    f.write("axs[0].plot(R_val, E_CASCI, linestyle=':', color='brown', label='CASCI', linewidth=linewidth)\n")
    f.write("axs[0].plot(R_val, E_OBMP2, linestyle='-', color='purple', marker='o', markersize=2, markevery=1, label='OBMP2', linewidth=linewidth)\n")
    f.write("axs[0].plot(R_val, E_SQD, linestyle=':', color='blue', marker='o', markersize=2, markevery=1, label='CAS-SQD', linewidth=linewidth)\n")
    f.write("axs[0].plot(R_val, E_Dfold_SQD, linestyle=':', color='red', marker='o', markersize=2, markevery=1, label='Dfold-SQD', linewidth=linewidth)\n")
    f.write("axs[0].plot(R_val, E_CCSDT, linestyle='-', color='crimson', label='FCI', linewidth=linewidth)\n\n\n")
    f.write("axs[0].set_ylabel('E (Hartree)')\n")
    f.write("axs[0].legend(frameon=False)\n")
    f.write("axs[0].grid(True, linestyle=':', linewidth=0.5)\n\n\n")
    f.write("axs[1].plot(R_val, E_RHF-E_CCSDT, linestyle='-', color='purple', linewidth=linewidth)\n")
    f.write("axs[1].plot(R_val, E_CASCI-E_CCSDT, linestyle=':', color='brown', linewidth=linewidth)\n")
    f.write("axs[1].plot(R_val, E_OBMP2-E_CCSDT, linestyle='-', color='purple', marker='o', markersize=2, markevery=1, linewidth=linewidth)\n")
    f.write("axs[1].plot(R_val, E_SQD-E_CCSDT, linestyle=':', color='blue', marker='o', markersize=2, markevery=1, linewidth=linewidth)\n")
    f.write("axs[1].plot(R_val, E_Dfold_SQD-E_CCSDT, linestyle=':', color='red', marker='o', markersize=2, markevery=1, linewidth=linewidth)\n\n\n")
    f.write("axs[1].set_ylim(0.0, 0.3)\n")
    f.write("axs[1].set_xlabel('Bond distance (armstrong)')\n")
    f.write("axs[1].set_ylabel('\u0394E (Hartree)')\n")
    f.write("axs[1].grid(True, linestyle=':', linewidth=0.5)\n\n\n")
    f.write("plt.tight_layout()\n\n")
    f.write(f"plt.savefig('{mole}_{arrange}_{basis}_{nact[0]}o{caslist_a}_[dR={dR}]_Restricted.png', dpi=700, format='png')\n")
    f.write("plt.show()\n")
    f.write("plt.close()\n")

#--------------------Ghi dữ liệu--------------------

#==============================Plot==============================
"""





















