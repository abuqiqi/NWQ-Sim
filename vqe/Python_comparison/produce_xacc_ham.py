import re
from qiskit_nature.units import DistanceUnit
import numpy


def qiskit_normal_order_switch(qiskit_matrix):
    ## qiskit matrix to normal matrix or verse versa
    num_qubits = int(numpy.log2(qiskit_matrix.shape[0]))
    bin_str = '{0:0'+str(num_qubits)+'b}'
    new_matrix = numpy.zeros(qiskit_matrix.shape, dtype=complex)
    for i in range(qiskit_matrix.shape[0]):
        for j in range(qiskit_matrix.shape[1]):
            normal_i = int(bin_str.format(i)[::-1],2)
            normal_j = int(bin_str.format(j)[::-1],2)
            new_matrix[normal_i,normal_j] = qiskit_matrix[i,j]
    return new_matrix

def a1a2_to_a1b1_mapping(n_qubits):
    ## Qubit i in a1a2 order is mapped to perm_arr[i] in a1b1 order
    perm_arr = list(range(0, n_qubits, 2)) + list(range(1, n_qubits, 2))
    # mapping_dict = {i:perm_arr[i] for i in range(n_qubits)}
    return perm_arr 

def a1b1_to_a1a2_mapping(n_qubits):
    ## Qubit i in a1b1 order is mapped to perm_arr[i] in a1a2 order
    backward_perm_arr = a1a2_to_a1b1_mapping(n_qubits)
    new_list = [0]*n_qubits
    for i in range(n_qubits):
        new_list[backward_perm_arr[i]] = i
    return new_list 

def extract_orbitals(string):
    # This regex pattern matches one or more digits, optionally preceded by '^'
    pattern = r'\^?\d+'
    
    # Find all matches in the string
    matches = re.findall(pattern, string)
    
    # Convert matches to integers, removing '^' if present
    orbitals = [int(match.replace('^', '')) for match in matches]
    
    return orbitals

def format_save_ham(pathtofile, ham_op, n_qubits, xacc_scheme=False):
    ## openfermion FermionOperator
    ## E.g., from H4L().fermi_ham

    with open(pathtofile, 'w') as f:
        ## product the line
        for term in ham_op:
            coef, op = str(term).split(' [')
            coef = f'({coef}, 0)'
            op = op[:-1]
            if len(op) == 0:
                linestr = coef 
            elif not xacc_scheme: ## not need to change orbital index
                linestr = coef+op
            else: ## change orbital index
                perm_arr = a1b1_to_a1a2_mapping(n_qubits)
                a1b1_spin_orbitals = re.findall(r'\d+',op)
                a1a2_spin_orbitals = [str(perm_arr[int(i)]) for i in a1b1_spin_orbitals]
                perm_oplist = []
                for i in range(len(a1a2_spin_orbitals)):
                    perm_oplist.append(a1a2_spin_orbitals[i])
                    if i < len(a1a2_spin_orbitals)*0.5:
                        perm_oplist.append('^')
                    if i < len(a1a2_spin_orbitals)-1:
                        perm_oplist.append(' ')
                opstr = ''.join(perm_oplist)
                linestr = coef+opstr
            ## write to file
            f.write(f"{linestr}\n")





def save_and_validate(folder_path, mole_obj):
    ## Save 
    n_qubits = mole_obj.n_orb*2
    format_save_ham(f'{folder_path}mz_{mole_obj.mol_name}.hamil',mole_obj.fermi_ham, n_qubits, xacc_scheme=True)
    ## Also save matrix in npz
    import openfermion as of
    ham_mat = of.linalg.get_sparse_operator(mole_obj.fermi_ham, n_qubits)
    scipy.sparse.save_npz(f'{folder_path}mz_{mole_obj.mol_name}_mat.npz', ham_mat, compressed=True)

    ## Validate the saving
    print("\n----------- Validate the Output by Reloading It -----------")
    from _gcim_utilis import parse_hamiltonian
    loaded_fermi_ham = parse_hamiltonian(f'{folder_path}mz_{mole_obj.mol_name}.hamil', n_qubits//2, use_interleaved=True, return_type = 'of')
    loaded_ham_mat = of.linalg.get_sparse_operator(loaded_fermi_ham, n_qubits)

    import scipy.sparse as ss
    error = ss.linalg.norm(ham_mat - loaded_ham_mat)
    print(f"  (ADA) {mole_obj.mol_name}: Error between original and loaded hamiltonian is {error}")
    return loaded_fermi_ham



###-------------------------------------------------------------------------------------------------------

def qis2of_opstr(string):
    # This regex pattern matches '+_' or '-_' followed by a number
    pattern = r'([+-])_(\d+)'
    
    def replacement(match):
        sign, number = match.groups()
        return f"{number}^" if sign == '+' else f"{number}"
    
    # Use re.sub to replace all occurrences
    transformed = re.sub(pattern, replacement, string)
    
    return transformed

def format_save_qisham_and_mat(pathtofile, geom, unit=DistanceUnit.BOHR):
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper, InterleavedQubitMapper

    # geom = "H 0 0 0; H 2 0 0; H 4 0 0; H 6 0 0"
    # unit = DistanceUnit.BOHR

    driver = PySCFDriver(
        atom=geom,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=unit,
    )

    es_problem = driver.run()
    mapper = JordanWignerMapper()
    # mapper = InterleavedQubitMapper(JordanWignerMapper())
    qiskit_ham = es_problem.hamiltonian
    ## Add Nuclear Repulsion to Hamiltonian
    from qiskit_nature.second_q.operators import PolynomialTensor
    qiskit_ham.electronic_integrals.alpha += PolynomialTensor({
        "": qiskit_ham.nuclear_repulsion_energy
    })
    qiskit_ham.nuclear_repulsion_energy = None
    ##
    qiskit_ferm_ham = qiskit_ham.second_q_op()
    ##
    with open(pathtofile, 'w') as f:
        ## product the line
        for op,coef in qiskit_ferm_ham.items():
            coef = f'({coef}, 0)'
            op = qis2of_opstr(op)
            if len(op) == 0: ## if constant
                linestr = coef 
            else: ## Qiskit should already use the same scheme as XACC
                linestr = coef+op
            ## write to file
            f.write(f"{linestr}\n")

    qiskit_ham_mat = mapper.map(qiskit_ferm_ham).to_matrix(sparse=True) ## Note that qiskit Hamiltonian does not added Nuclear Repulsion
    # order_adjusted = qiskit_normal_order_switch(qiskit_ham_mat)
    return qiskit_ham_mat



def save_and_validate_qis(folder_path, mole_name, geom, unit=DistanceUnit.BOHR):
    ## Save 
    qis_ham_mat = format_save_qisham_and_mat(f'{folder_path}mzqis_{mole_name}.hamil', geom, unit=unit)
    scipy.sparse.save_npz(f'{folder_path}mzqis_{mole_name}_mat.npz', qis_ham_mat, compressed=True)
    
    ## Validate the saving
    print("\n----------- Validate the Output by Reloading .hamil and compare to sparse matrix -----------")
    from _gcim_utilis import parse_hamiltonian
    n_qubits = int(np.log2(qis_ham_mat.shape[0]))
    loaded_fermi_ham = parse_hamiltonian(f'{folder_path}mzqis_{mole_name}.hamil', n_qubits//2, use_interleaved=True, return_type = 'of')
    loaded_ham_mat = of.linalg.get_sparse_operator(loaded_fermi_ham).todense()

    qiskit_ham_mat = qis_ham_mat.todense()
    import scipy.linalg as la
    diff = la.eigh(qiskit_ham_mat)[0] - la.eigh(loaded_ham_mat)[0]
    error = np.linalg.norm(diff)
    print("  Different ev entries:", diff[abs(diff) > 1e-8].shape , diff[abs(diff) > 1e-8])
    print(f"  (Qis) {mole_name}: Error between original and loaded hamiltonian in ev is {error}")
    return loaded_fermi_ham



def geom_scf2qis(geo_list):
    ## E.g., [('H', (0,0,0)), ('H', (2,0,0)), ('H', (4,0,0)), ('H', (6,0,0))] to "H 0 0 0; H 2 0 0; H 4 0 0; H 6 0 0"
    return '; '.join([f"{ele[0]} {ele[1][0]} {ele[1][1]} {ele[1][2]}" for ele in geo_list])



if __name__ == "__main__":

    from moles import *
    def big_save_function(mole_obj):
        ## Save 
        mole_name = mole_obj.mol_name
        folder_path = './vqe/example_hamiltonians/mz_tests/'
        print(f"\n\n---------- {mole_name} ----------")


        ## Adapt-vqe code Save
        mole_obj.initialize()
        ada_fermi_ham = save_and_validate(folder_path, mole_obj)
        ada_ham_mat = of.linalg.get_sparse_operator(ada_fermi_ham)
        ## Qiskit Save
        geom = geom_scf2qis(mole_obj.geometry)
        if 'A' in mole_obj.mol_unit.upper():
            unit = DistanceUnit.ANGSTROM
        else:
            unit = DistanceUnit.BOHR
        print("  Unit: ", unit)
        qis_fermi_ham = save_and_validate_qis(folder_path, mole_name, geom, unit=unit)
        qis_ham_mat = of.linalg.get_sparse_operator(qis_fermi_ham)
        
        
        # ## Compare beteen Qiskit and ADAPT-VQE
        import scipy.sparse as ss
        import scipy.linalg as la
        ##
        # mz_mat = ss.load_npz(f'{folder_path}mz_{mole_obj.mol_name}_mat.npz').todense()
        # qis_mat = ss.load_npz(f'{folder_path}mzqis_{mole_obj.mol_name}_mat.npz').todense()
        mz_mat = ada_ham_mat.todense()
        qis_mat = qis_ham_mat.todense()
        ##
        qiskit_evs = la.eigh(qis_mat)[0]
        mz_evs = la.eigh(mz_mat)[0]
        diff1 = qis_mat - mz_mat
        diff2 = qiskit_evs - mz_evs
        print("\n----------- Compare Qiskit and ADAPT-VQE (Rconstruct from .hamil file) -----------")
        print( "  Qiskit and Mine, Difference in matrix entry > 1e-8:", diff1[abs(diff1) > 1e-8].shape , diff1[abs(diff1) > 1e-8])
        print( "  Qiskit and Mine, Difference in each eigenvalue > 1e-8:", diff2[abs(diff2) > 1e-8] )


    # big_save_function(H4L())
    # big_save_function(H4S())
    # big_save_function(LiH())
    # big_save_function(H635())
    # big_save_function(H65A())


    # folder_path = './vqe/example_hamiltonians/mz_tests/'
    folder_path = './'

    # save_and_validate_qis(folder_path, "LiH25", "Li 0 0 0; H 0 0 2.5", unit=DistanceUnit.ANGSTROM)
    
    # a = 1.8
    # save_and_validate_qis(folder_path, "H618", f"H 0 0 0; H {a} 0 0; H {a*2} 0 0; H {a*3} 0 0; H {a*4} 0 0; H {a*5} 0 0", unit=DistanceUnit.ANGSTROM)

    # a = 1.0
    # save_and_validate_qis(folder_path, "H410", f"H 0 0 0; H {a} 0 0; H {a*2} 0 0; H {a*3} 0 0", unit=DistanceUnit.ANGSTROM)

    # a = 1.5
    # save_and_validate_qis(folder_path, "H415SCF", f"H 0 0 0; H {a} 0 0; H {a*2} 0 0; H {a*3} 0 0", unit=DistanceUnit.ANGSTROM)


    # a = 5.0
    # save_and_validate_qis(folder_path, "H450", f"H 0 0 0; H {a} 0 0; H {a*2} 0 0; H {a*3} 0 0", unit=DistanceUnit.ANGSTROM)
    
    a = 4.0
    save_and_validate_qis(folder_path, "H640", f"H 0 0 0; H {a} 0 0; H {a*2} 0 0; H {a*3} 0 0; H {a*4} 0 0; H {a*5} 0 0", unit=DistanceUnit.ANGSTROM)

