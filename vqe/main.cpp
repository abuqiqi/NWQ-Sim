#include "vqeBackendManager.hpp"
#include "utils.hpp"
#include <cmath>
#include <string>
#include "circuit/dynamic_ansatz.hpp"
#include "vqe_adapt.hpp"
#include <chrono>

#define UNDERLINE "\033[4m"

#define CLOSEUNDERLINE "\033[0m"
using IdxType = NWQSim::IdxType;
using ValType = NWQSim::ValType;

struct VQEParams {
  /**
   * @brief  Structure to hold command line arguments for NWQ-VQE
   */
  // Problem Info
  std::string hamiltonian_path = "";
  IdxType nparticles = -1;
  bool xacc = true;

  // Simulator options
  std::string backend = "CPU";
  std::string config = "../default_config.json";
  uint32_t seed;
  // Optimizer settings
  NWQSim::VQE::OptimizerSettings optimizer_settings;
  nlopt::algorithm algo;

  // ADAPT-VQE options
  bool adapt = false;
  bool qubit = false;
  IdxType adapt_maxeval = 100;
  ValType adapt_fvaltol = 1e-6;
  ValType adapt_gradtol = 1e-3;
  IdxType adapt_pool_size = -1;
};
int show_help() {
  std::cout << "NWQ-VQE Options" << std::endl;
  std::cout << UNDERLINE << "REQUIRED" << CLOSEUNDERLINE << std::endl;
  std::cout << "--hamiltonian, -f     Path to the input Hamiltonian file (formatted as a sum of Fermionic operators, see examples)" << std::endl;
  std::cout << "--nparticles, -n      Number of electrons in molecule" << std::endl;
  std::cout << UNDERLINE << "OPTIONAL" << CLOSEUNDERLINE << std::endl;
  std::cout << "--backend, -b         Simulation backend. Defaults to CPU" << std::endl;
  std::cout << "--list-backends, -l   List available backends and exit." << std::endl;
  std::cout << "--seed                Random seed for initial point and empirical gradient estimation. Defaults to time(NULL)" << std::endl;
  std::cout << "--config              Path to NWQ-Sim config file. Defaults to \"../default_config.json\"" << std::endl;
  std::cout << "--opt-config          Path to config file for NLOpt optimizer parameters" << std::endl;
  std::cout << "--optimizer           NLOpt optimizer name. Defaults to LN_COBYLA" << std::endl;
  std::cout << "--lbound              Lower bound for classical optimizer. Defaults to -PI" << std::endl;
  std::cout << "--ubound              Upper bound for classical optimizer. Defaults to PI" << std::endl;
  std::cout << "--reltol              Relative tolerance termination criterion. Defaults to -1 (off)" << std::endl;
  std::cout << "--abstol              Relative tolerance termination criterion. Defaults to -1 (off)" << std::endl;
  std::cout << "--maxeval             Maximum number of function evaluations for optimizer. Defaults to 200" << std::endl;
  std::cout << "--maxtime             Maximum optimizer time (seconds). Defaults to -1.0 (off)" << std::endl;
  std::cout << "--stopval             Cutoff function value for optimizer. Defaults to -MAXFLOAT (off)" << std::endl;
  std::cout << "--xacc                Use XACC indexing scheme, otherwise uses DUCC scheme. (Deprecated, true by default)" << std::endl;
  std::cout << "--ducc                Use DUCC indexing scheme, otherwise uses XACC scheme. (Defaults to true)" << std::endl;
  std::cout << UNDERLINE << "ADAPT-VQE OPTIONS" << CLOSEUNDERLINE << std::endl;
  std::cout << "--adapt               Use ADAPT-VQE for dynamic ansatz construction. Defaults to false" << std::endl;
  std::cout << "--adapt-maxeval       Set a maximum iteration count for ADAPT-VQE. Defaults to 100" << std::endl;
  std::cout << "--adapt-gradtol       Cutoff absolute tolerance for operator gradient norm. Defaults to 1e-3" << std::endl;
  std::cout << "--adapt-fvaltol       Cutoff absolute tolerance for function value. Defaults to 1e-6" << std::endl;
  std::cout << "--qubit               Uses Qubit instead of Fermionic operators for ADAPT-VQE. Defaults to false" << std::endl;
  std::cout << "--adapt-pool          Sets the pool size for Qubit operators. Defaults to -1" << std::endl;
  return 1;
}

using json = nlohmann::json;

/**
 * @brief  Parse command line arguments
 * @note   
 * @param  argc: Number of command line arguments passed
 * @param  argv: Pointer to command line arg C strings
 * @param  manager: Backend manager object
 * @param  params: Data structure to store command line arguments
 * @retval 
 */
int parse_args(int argc, char** argv,
               VQEBackendManager& manager,
               VQEParams& params) {
  std::string opt_config_file = "";
  params.config = "../default_config.json";
  std::string algorithm_name = "LN_COBYLA";
  NWQSim::VQE::OptimizerSettings& settings = params.optimizer_settings;
  params.seed = time(NULL);
  for (size_t i = 1; i < argc; i++) {
    std::string argname = argv[i];
    if (argname == "-h" || argname == "--help") {
      return show_help();
    } if (argname == "-l" || argname == "--list-backends") {
      manager.print_available_backends();
      return 1;
    } else
    if (argname == "-b" || argname == "--backend") {
      params.backend = argv[++i];//-2.034241 -1.978760  -1.825736
      continue;
    } else
    if (argname == "-f" || argname == "--hamiltonian") {
      params.hamiltonian_path = argv[++i];
      continue;
    } else 
    if (argname == "-n" || argname == "--nparticles") {
      params.nparticles = std::atoll(argv[++i]);
    } else 
    if (argname == "--seed") {
      params.seed = (unsigned)std::atoi(argv[++i]);
    }  else 
    if (argname == "--adapt-pool") {
      params.adapt_pool_size = (long long)std::atoi(argv[++i]);
    } else if (argname == "--xacc") {
      params.xacc = true;
    } else if (argname == "--ducc") {
      params.xacc = false;
    } else 
    if (argname == "--config") {
      params.config = argv[++i];
    } else 
    if (argname == "--opt-config") {
      opt_config_file = argv[++i];
    } else  
    if (argname == "--optimizer") {
      algorithm_name = argv[++i];
    } else 
    if (argname == "--reltol") {
      params.optimizer_settings.rel_tol = std::atof(argv[++i]);
    } else 
    if (argname == "--abstol") {
      settings.abs_tol = std::atof(argv[++i]);
    }  else 
    if (argname == "--ubound") {
      settings.ubound = std::atof(argv[++i]);
    }  else 
    if (argname == "--lbound") {
      settings.lbound = std::atof(argv[++i]);
    } else 
    if (argname == "--adapt-fvaltol") {
      params.adapt_fvaltol = std::atof(argv[++i]);
    } else 
    if (argname == "--adapt-gradtol") {
      params.adapt_gradtol = std::atof(argv[++i]);
    }  else 
    if (argname == "--adapt-maxeval") {
      params.adapt_maxeval = std::atoll(argv[++i]);
    } else 
    if (argname == "--adapt") {
      params.adapt = true;
    }  else 
    if (argname == "--qubit") {
      params.qubit = true;
    } else 
    if (argname == "--maxeval") {
      settings.max_evals = std::atoll(argv[++i]);
    } else if (argname == "--stopval") {
      settings.stop_val = std::atof(argv[++i]);
    } else if (argname == "--maxtime") {
      settings.max_time = std::atof(argv[++i]);
    } else {
      fprintf(stderr, "\033[91mERROR:\033[0m Unrecognized option %s, type -h or --help for a list of configurable parameters\n", argv[i]);
      return show_help();
    }
  }
  if (params.hamiltonian_path == "") {
      fprintf(stderr, "\033[91mERROR:\033[0m Must pass a Hamiltonian file (--hamiltonian, -f)\n");

      return show_help();
  }
  if (params.nparticles == -1) {
      fprintf(stderr, "\033[91mERROR:\033[0m Must pass a particle count (--nparticles, -n)\n");
      return show_help();
  }
  params.algo = (nlopt::algorithm)nlopt_algorithm_from_string(algorithm_name.c_str());
  if (opt_config_file != "") {
    std::ifstream f(opt_config_file);
    json data = json::parse(f); 
    for (json::iterator it = data.begin(); it != data.end(); ++it) {
      settings.parameter_map[it.key()] = it.value().get<NWQSim::ValType>();
    }
  }
  return 0;
}


// Callback function, requires signature (void*) (const std::vector<NWQSim::ValType>&, NWQSim::ValType, NWQSim::IdxType)
void carriage_return_callback_function(const std::vector<NWQSim::ValType>& x, NWQSim::ValType fval, NWQSim::IdxType iteration) {
  printf("\33[2K\rEvaluation %lld, fval = %f", iteration, fval);fflush(stdout);
}

// Callback function, requires signature (void*) (const std::vector<NWQSim::ValType>&, NWQSim::ValType, NWQSim::IdxType)
void callback_function(const std::vector<NWQSim::ValType>& x, NWQSim::ValType fval, NWQSim::IdxType iteration) {
  std::string paramstr = "[";
  for (auto i: x) {
    paramstr += std::to_string(i) + ", ";
  }
  printf("\33[2KEvaluation %lld, fval = %f, x=%s\n", iteration, fval, paramstr.c_str());fflush(stdout);
}
// Callback function, requires signature (void*) (const std::vector<NWQSim::ValType>&, NWQSim::ValType, NWQSim::IdxType)
void silent_callback_function(const std::vector<NWQSim::ValType>& x, NWQSim::ValType fval, NWQSim::IdxType iteration) {
  
}


/**
 * @brief  Optimized the UCCSD (or ADAPT-VQE) Ansatz and Report the Fermionic Excitations
 * @note   
 * @param  manager: API to handle calls to different backends
 * @param  params: Data structure with runtime-configurable options
 * @param  ansatz: Shared pointer to a parameterized quantum circuit
 * @param  hamil: Share pointer to a Hamiltonian observable
 * @param  x: Parameter vector (reference, output)
 * @param  fval: Energy value (reference, output)
 * @retval None
 */
void optimize_ansatz(const VQEBackendManager& manager,
                     VQEParams params,
                     std::shared_ptr<NWQSim::VQE::Ansatz> ansatz,
                     std::shared_ptr<NWQSim::VQE::Hamiltonian> hamil,
                     std::vector<double>& x,
                     double& fval) {
  // Set the callback function (silent is default)
  NWQSim::VQE::Callback callback = (params.adapt ? silent_callback_function : callback_function);
  std::shared_ptr<NWQSim::VQE::VQEState> state = manager.create_vqe_solver(params.backend,
                                                                           params.config,
                                                                           ansatz, 
                                                                           hamil, 
                                                                           params.algo, 
                                                                           callback, 
                                                                           params.seed, 
                                                                           params.optimizer_settings);  
  x.resize(ansatz->numParams());
  std::fill(x.begin(), x.end(), 0);

  
  if (params.adapt) {
    /***** ADAPT-VQE ******/
    // recast the ansatz pointer
    std::shared_ptr<NWQSim::VQE::DynamicAnsatz> dyn_ansatz = std::reinterpret_pointer_cast<NWQSim::VQE::DynamicAnsatz>(ansatz);
    // make the operator pool (either Fermionic or ADAPT)
    dyn_ansatz->make_op_pool(hamil->getTransformer(), params.seed, params.adapt_pool_size);
    // construct the AdaptVQE controller
    NWQSim::VQE::AdaptVQE adapt_instance(dyn_ansatz, state, hamil);
    // timing utilities
    auto start_time = std::chrono::high_resolution_clock::now();
    adapt_instance.make_commutators(); // Start making the commutators
    auto end_commutators = std::chrono::high_resolution_clock::now();
    double commutator_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_commutators - start_time).count() / 1e9;
    manager.safe_print("Constructed ADAPT-VQE Commutators in %.2e seconds\n", commutator_time); // Report the commutator overhead
    adapt_instance.optimize(x, fval, params.adapt_maxeval, params.adapt_gradtol, params.adapt_fvaltol); // MAIN OPTIMIZATION LOOP
    auto end_optimization = std::chrono::high_resolution_clock::now();
    double optimization_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_optimization - end_commutators ).count() / 1e9;
    manager.safe_print("Completed ADAPT-VQE Optimization in %.2e seconds\n", optimization_time); // Report the total time
  } else {
    state->optimize(x, fval); // MAIN OPTIMIZATION LOOP

  }
  
}


int main(int argc, char** argv) {
  VQEBackendManager manager;
  VQEParams params;

  if (parse_args(argc, argv, manager, params)) {
    return 1;
  }
#ifdef MPI_ENABLED
  int i_proc;
  if (params.backend == "MPI" || params.backend == "NVGPU_MPI")
  {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &i_proc);
}
#endif

  // Get the Hamiltonian from the external file
  manager.safe_print("Reading Hamiltonian...\n");
  std::shared_ptr<NWQSim::VQE::Hamiltonian> hamil = std::make_shared<NWQSim::VQE::Hamiltonian>(params.hamiltonian_path, 
                                                                                               params.nparticles,
                                                                                               params.xacc);
  manager.safe_print("Constructed %lld Pauli Observables\n", hamil->num_ops());
  manager.safe_print("Constructing UCCSD Ansatz...\n");
  
  // Build the parameterized ansatz
  std::shared_ptr<NWQSim::VQE::Ansatz> ansatz;
  if (params.adapt)
  {
    // Iteratively build an ADAPT-VQE ansatz (starts from HF state)
    NWQSim::VQE::PoolType pool = params.qubit ? NWQSim::VQE::PoolType::Pauli : NWQSim::VQE::PoolType::Fermionic;
    ansatz = std::make_shared<NWQSim::VQE::DynamicAnsatz>(hamil->getEnv(), pool);
  } else {
    // Static UCCSD ansatz
    ansatz  = std::make_shared<NWQSim::VQE::UCCSD>(
      hamil->getEnv(),
      NWQSim::VQE::getJordanWignerTransform,
      1
    );
  }
  ansatz->buildAnsatz();

  manager.safe_print("%lld Gates with %lld parameters\n" ,ansatz->num_gates(), ansatz->numParams());
  std::vector<double> x;
  double fval;
  manager.safe_print("Beginning VQE loop...\n");
  optimize_ansatz(manager, params, ansatz,  hamil, x, fval);
  
  // Print out the Fermionic operators with their excitations
  std::vector<std::pair<std::string, double> > param_map = ansatz->getFermionicOperatorParameters();
  manager.safe_print("\nFinished VQE loop.\n\tFinal value: %e\n\tFinal parameters:\n", fval);
  for (auto& pair: param_map) {
    manager.safe_print("%s :: %e\n", pair.first.c_str(), pair.second);

  }
#ifdef MPI_ENABLED
  if (params.backend == "MPI" || params.backend == "NVGPU_MPI")
  {
    MPI_Finalize();
  }
#endif
  return 0;
}