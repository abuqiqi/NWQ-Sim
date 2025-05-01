# NWQ-VQE
These documents describe the structure and functionality of the NWQ-VQE simulator. 

## Contents
- [Operators](components/operators.md): Utilities for creating and manipulating Hamiltonians, Pauli operators, and Fermionic operators. Covers:
  - vqe/include/hamiltonian.hpp
  - vqe/include/pauli_operator.hpp
  - vqe/include/observable/fermionic_operator.hpp
  - vqe/include/environment.hpp
- [Fermion to Qubit Transformers](components/transformers.md): Transformer function prototype and the Jordan-Wigner mapper
  - vqe/include/transform/transform.cpp
- [Ansätze](components/circuits.md): Ansatz base and UCCSD classes
  - vqe/include/ansatz.hpp
  - vqe/src/uccsd.cpp
  - vqe/src/uccsdmin.cpp
  - vqe/src/singletgsd.cpp
- [VQE States](components/vqe_states.md): Covers the base VQEState interface as well as the backend-specific implementations
  - vqe/include/vqe_state.hpp
  - vqe/include/svsim_vqe/sv_cpu_vqe.hpp
  - vqe/include/svsim_vqe/sv_cuda_vqe.hpp
  - vqe/include/svsim_vqe/sv_mpi_vqe.hpp
  - vqe/include/svsim_vqe/sv_cuda_mpi_vqe.hpp
- [Measurement](components/measurement.md): Construction of QWC groups via Sorted Insertion, measurement circuit construction, and `ObservableList` initialization
  - vqe/include/circuit/measurement.hpp
  - Related functions in vqe/include/utils.hpp, vqe/src/utils.cpp
  - Related functions in vqe_state.hpp and assorted backends
- [Expectation Value Calculation](components/expectation_values.md): Expectation value calculation functions and `ObservableList` usage
  - Modifications to include/gates.hpp
  - EXPECT_GATE implementations in include/svsim/* backends
  - Related functions in vqe/include/utils.hpp, vqe/src/utils.cpp
  - Related functions in vqe_state.hpp and assorted backends
- [Function Optimization](components/optimization.md): Interface with NLOpt and associated options
  - Related code in vqe/include/vqe_state.hpp
- [ADAPT-VQE](components/adapt.md): Fermionic-ADAPT and Qubit-ADAPT optimizers. Covers:
  - [vqe_adapt.hpp](../vqe/include/vqe_adapt.hpp)
  - Related functions in [ansatz_pool.cpp](../vqe/src/ansatz_pool.cpp) ~~[utils.cpp](../vqe/src/utils.cpp)~~
- [Utilities](components/utils.md): General utilities defined in [utils.cpp](../vqe/src/utils.cpp) and [utils.hpp](../vqe/include/utils.hpp)



# A Guide for Adding New Things and Important Points

This is a guide for things you need to change when you want to add new things, such as ansatzes.

## Important Points
1. In `vqe/include/environment.hpp`, structure `MolecularEnvironment` assumes the the number of occupied orbitals `n_occ` equal number of alpha- and beta-spin electron `n_part` by setting
```c++
n_occ = n_part / 2
n_virt = n_spatial - n_occ
```
While the number of spatial orbital `n_spatial` is user-defined, the number of virtual orbital is defined by 
```c++
n_virt = n_spatial - n_occ
```
2. In general, before considering XACC or DUCC scheme for orbital index in the output, the code is implemented under the assumption that **occupied orbitals come before virtual orbitals**.


## Adding new ansatz for VQE and QFLOW

This is for the ansatz associate with `main.cpp` and `qflow.cpp`. You need to create a new class that inherits the `Ansatz` class in `vqe/include/circuit/ansatz.hpp`. You can 
1. Similar to the `UCCSD` class: create a new class inherited from `Ansatz` inside the `ansatz.hpp`, then create a `.cpp` file (or copy-paste `uccsd.cpp`) in `vqe/src/` to fill excitation generation functions
2. Similar to the `UCCSDmin` class: create a new class in the `.cpp` file (or copy-paste `uccsdmin.cpp`) only in `vqe/src/`, and include the file in both `main.cpp` and `qflow.cpp`.

A few points to note
1. Compute the number of single and double excitations, pre-allocate enough space for `fermion_operators`.
2. Assign value for `ansatz_name`
3. Override/Create function `getFermionOps()` to fill excitation operators to `fermion_operators`. Use class `FermionOperator` for each creation and annihilation operator, i.e., $a^\dagger$ and $a$
    1. If copy-paste the file, use `add_single_excitation()` and `add_double_excitation()` to push back two or four creation and annihilation operators as one excitation operator. NOTE: using "Annihilation Creation" or "Ann Ann Cre Cre" order.
        * Note that the each excitation operator is printed through `getFermionicOperatorParameters()` function, and it uses `opstring = op.toString(env.n_occ, env.n_virt) + opstring` so  "Ann Ann Cre Cre" is the correct order, see `Singlet_GSD` class for more clarity.
        * This order is also verified through comparing the optimized parameter values with those from the Qiskit's UCCSD on a H4 example.
    2. The class `FermionOperator` accept 
    ```c++
    FermionOperator iop (i,  // This is the (i-1)^th OCCUPIED or VIRTUAL orbital
                    Occupied,  // orbital type: Occupied or Virtual
                    Up,   // spin: Up or Down
                    Annihilation, // Fermion operator type: Creation or Annihilation
                    env.xacc_scheme, // If use xacc scheme for SPARTIAL orbital index, True or False
                    1.0);  // (optional) Coefficient
    0.5* iop; // a complex or double multiply a FermionOperator modifiy the coefficient
    iop * 0.5; //
    ```
    NOTE, as it is popular to use spartial orbital indeice to generate excitation operators in other packages, when you do the same thing here, you need to detect if the index is an occupied or virtual orbital as following
    ```c++
    for (IdxType p = 0; p < env.n_spatial; p++) {
        if (p < env.n_occ) { // assume that first `n_occ` number of orbitals are occupied.
            std::cout << "Orbital " << p << "is Occupied" << std::endl;
        } else {
            std::cout << "Orbital " << p << "is Virtual" << std::endl;
        }
    }
    ```
    The functions `occ_or_vir()` and `spind_to_ind()` in `vqe/observable/fermionic_operator.hpp` will be helpful to determine this and do re-index. An example is shown in `vqe/src/singletgsd.cpp`.
4. Remeber to modify `pauli_oplist.reserve(4 * n_singles + 16 * n_doubles);` in `buildAnsatz()` if you are using more complicated excitation operators.
5. Remeber to change `main.cpp` and `qflow.cpp` so your new ansatz can be enabled in the command line.


## Adding new ansatz for ADAPT-VQE

You need to 
1. in the file `vqe/include/observable/fermionic_operator.hpp`
    1. append the function that generate new dynamic ansatz
2. in the file `vqe/include/circuit/dynamic_ansatz.hpp`
    1. modify the function `make_op_pool()` under class `DynamicAnsatz` 
    2. Enlarge class `PoolType`
    3. Assign `ansatz_name`
    4. See the important points about using class `FermionOperator` in the above section
3. in the file `vqe/src/ansatz_pool.cpp` ~~`vqe/src/utils.cpp`~~
    1. elabrate your actual function that generate new dynamic ansatz
5. Remeber to add the new pool type into the following place
    1. `vqe/include/circuit/dynamic_ansatz.hpp`: `get_operator_string()`, `add_operator()`, and `getFermionicOperatorParameters()`
4. Remeber to change `show_help()` and `parse_args()` in `main.cpp` so your new ansatz can be enabled in the command line.

Use `generate_fermionic_excitations()` in `vqe/src/ansatz_pool.cpp` ~~`vqe/src/utils.cpp`~~ as an example.


## Change things related to NLOPT optimizer

Note that there are three `optimize()` functions:
    1. `vqe/include/vqe_state.hpp`
    2. `vqe/include/svsim_vqe/sv_mpi_vqe.hpp`
    3. `vqe/include/svsim_vqe/sv_cuda_mpi_vqe.cuh`



# Calculations


## Grouping Paulis in the Hamiltonian

The Hamiltonian QWC clique generation happens through the `sorted_insertion()` function in `utils.hpp`, which is indirectly called during the initialization of the `VQEState`  class. The function partitions the Pauli operators of the Hamiltonian into cliques where all operators within a clique commute with each other qubit-wise.
This partitioning is crucial for efficient measurement of the Hamiltonian expectation value, as operators in the same QWC clique can be measured simultaneously, reducing the number of required measurements.

Function `sorted_insertion()` is called by `getJordanWignerTransform()` in `jw.cpp`.

## Expectation Calculation

The returned expectation value is stored in the `expectation_value` field of the `VQEState` class. The `expectation_value` in `VQEState` is updated in the `call_simulator()` method of each backend implementation. The actual computation of expectation values happens during the simulation of the measurement circuit, where the `EXPECT` gate operation calculates the expectation value for each Pauli term in the Hamiltonian and stores it in the corresponding `ObservableList` structure. The `call_simulator()` method then accumulates these values to get the total expectation value.