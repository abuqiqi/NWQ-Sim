#include "environment.hpp"
#include "circuit/ansatz.hpp"
#include "src/uccsd.cpp"
#include "src/uccsdmin.cpp"
#include "src/jw.cpp"
#include "src/singletgsd.cpp"
#include <iostream>


using namespace NWQSim::VQE;
using namespace NWQSim;
using namespace std;

//  cmake .. && make && clear
// cmake . && make && clear && ./test_ansatz

int main() {
    int n_spatial = 4;
    int n_particles = 4;
    bool use_xacc = true;
    MolecularEnvironment env(n_spatial, n_particles, use_xacc);

    // FermionOperator test_fem (1, Virtual, Down, Creation, env.xacc_scheme);
    // auto test_fem2 = test_fem*0.5;
    // auto test_fem21 = 0.5*test_fem;
    // std::complex<ValType> scalar(2.0, 1.0);
    // auto test_fem3 = test_fem*scalar;

    // std::cout << test_fem.getCoeff() << std::endl;
    // std::cout << test_fem2.getCoeff() << std::endl;
    // std::cout << test_fem21.getCoeff() << std::endl;
    // std::cout << test_fem3.getCoeff() << std::endl;

    std::cout << "\n\n UCCSD Qiskit Sym 2" << std::endl;
    std::shared_ptr<NWQSim::VQE::Ansatz> ansatz;
    ansatz  = std::make_shared<NWQSim::VQE::UCCSDmin>(
        env,
        NWQSim::VQE::getJordanWignerTransform,
        1,
        2
    );
    ansatz->buildAnsatz();
    std::cout << "Sym 2: " << ansatz->numParams() << " Parameters " << ansatz->numOps() << " Operators" << std::endl;

    // std::cout << "\n\n UCCSD Singlet GSD" << std::endl;
    // auto singletgsd = std::make_shared<NWQSim::VQE::Singlet_GSD>(
    //     env,
    //     NWQSim::VQE::getJordanWignerTransform,
    //     1
    // );
    // singletgsd->buildAnsatz();
    // std::cout << "Singlet GSD: " << singletgsd->numParams() << " Parameters " << singletgsd->numOps() << " Operators" << std::endl;


    return 0;
}



