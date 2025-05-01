from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, InterleavedQubitMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP, COBYLA, P_BFGS,ADAM,AQGD,NFT
from qiskit.primitives import Estimator

import scipy
import numpy as np


## Generate the Hamiltonian
a = 1.0
geom = f"H 0 0 0; H {a} 0 0;"

print("Geometry: ", geom)
unit = DistanceUnit.ANGSTROM
##
driver = PySCFDriver(
    atom=geom,
    basis="sto3g",
    charge=0,
    spin=0,
    unit=unit,
)
es_problem = driver.run()
mapper = JordanWignerMapper()


## Generate the ansatz
ansatz = UCCSD(
    es_problem.num_spatial_orbitals,
    es_problem.num_particles,
    mapper,
    initial_state=HartreeFock(
        es_problem.num_spatial_orbitals,
        es_problem.num_particles,
        mapper,
    ),
)

## VQE
vqe_solver = VQE(Estimator(), ansatz,  COBYLA(maxiter=2000, disp=True))
vqe_solver.initial_point = [0.0] * ansatz.num_parameters
##
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
calc = GroundStateEigensolver(mapper, vqe_solver)
res = calc.solve(es_problem)
print(res)


## Assign the parameters
optimal_circuit = ansatz.assign_parameters(res.raw_result.optimal_point)
print(optimal_circuit)

from qiskit.quantum_info import Operator
optimal_circuit = ansatz.assign_parameters(res.raw_result.optimal_point)
print( Operator(optimal_circuit).to_matrix() )

print(ansatz.operators)





mpirun -n 8 ./nwq_qflow --nparticles 6 --hamiltonian ./1_2_3_4_5_6-xacc --backend MPI --maxeval 5000 --abstol 1e-10 --xacc --optimizer LN_NEWUOA --symm 4 --lbound -2 --ubound 2 --verbose --qis 1

mpirun -n 4 ./nwq_vqe --nparticles 6 --hamiltonian 1_2_3_4_5_6-xacc --backend MPI --maxeval 5000 --abstol 1e-10 --xacc --optimizer LN_NEWUOA --symm 4 --lbound -2 --ubound 2 --verbose --qis 0