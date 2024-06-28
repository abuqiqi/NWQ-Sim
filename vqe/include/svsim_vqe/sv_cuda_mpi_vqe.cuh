#ifndef VQE_CUDA_MPI_STATE
#define VQE_CUDA_MPI_STATE
#include "svsim/sv_cuda_mpi.cuh"
#include "vqe_state.hpp"
#include "observable/pauli_operator.hpp"
#include "utils.hpp"
#include "circuit/ansatz.hpp"
#include "circuit/measurement.hpp"
#include "gradient/sa_gradient.hpp"
#include "observable/hamiltonian.hpp"
#include "nlopt.hpp"
#include <memory>
#include <cmath>

namespace NWQSim
{
  namespace VQE {
    class SV_CUDA_MPI_VQE: public VQEState, public SV_CUDA_MPI {
      public:
        SV_CUDA_MPI_VQE(std::shared_ptr<Ansatz> a, 
                   std::shared_ptr<Hamiltonian> h, 
                   nlopt::algorithm optimizer_algorithm,
                   Callback _callback,
                   IdxType seed = 0,
                   OptimizerSettings opt_settings = OptimizerSettings()): 
                                      SV_CUDA_MPI(a->num_qubits()),
                                      VQEState(a, h, optimizer_algorithm, _callback, seed, opt_settings) {
        
        // Pauli term data sizes
        IdxType nterms = xmasks.size();
        obs.numterms = nterms;
        IdxType isize = nterms * sizeof(IdxType);
        IdxType vsize = nterms * sizeof(ValType);
        expvals.resize(1);
        // Allocate space on the GPU for the Pauli expectation information
        //  For now, duplicate info on each GPU (i.e. don't use nvshmem)
        SAFE_ALOC_GPU(obs.xmasks, isize);
        SAFE_ALOC_GPU(obs.zmasks, isize);
        SAFE_ALOC_GPU(obs.x_index_sizes, isize);
        SAFE_ALOC_GPU(obs.x_indices, x_indices.size() * sizeof(IdxType));
        SAFE_ALOC_GPU(obs.exp_output, expvals.size()* sizeof(ValType));
        SAFE_ALOC_GPU(obs.coeffs, vsize);

        // Copy Pauli masks and indices from host to device
        cudaSafeCall(cudaMemcpy(obs.xmasks, xmasks.data(), isize,
                                    cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(obs.zmasks, zmasks.data(), isize,
                                    cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(obs.x_index_sizes, x_index_sizes.data(), isize,
                                    cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(obs.x_indices, x_indices.data(), x_indices.size() * sizeof(IdxType),
                                    cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(obs.coeffs, coeffs.data(), coeffs.size() * sizeof(ValType),
                                    cudaMemcpyHostToDevice));
        ObservableList* obs_device;
        SAFE_ALOC_GPU(obs_device, sizeof(ObservableList));
        cudaSafeCall(cudaMemcpy(obs_device, &obs, sizeof(ObservableList),
                                    cudaMemcpyHostToDevice));
        ansatz->EXPECT(obs_device);
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        n_cpus = size;

      };
      virtual void call_simulator() override {  
        std::vector<ValType> xparams;   
        if (iteration > 0){
          Config::PRINT_SIM_TRACE = false;
        }
        if (i_proc == 0) {
          std::vector<double>* ansatz_params = ansatz->getParams();
          stat = CALL_SIMULATOR;
          for(IdxType i = 1; i < n_gpus; i++) {
            MPI_Send(&stat, 1, MPI_INT, i, iteration, MPI_COMM_WORLD);
            MPI_Send(ansatz_params->data(), ansatz->numParams(), MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
          }
        } else {
          xparams.resize(ansatz->numParams());
          MPI_Recv(xparams.data(), ansatz->numParams(), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          ansatz->setParams(xparams);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        cudaSafeCall(cudaMemset(obs.exp_output, 0, expvals.size() * sizeof(ValType)));
        reset_state();
        sim(ansatz);
        cudaDeviceSynchronize();
        cudaSafeCall(cudaMemcpy(expvals.data(), obs.exp_output, expvals.size() * sizeof(ValType),
                                    cudaMemcpyDeviceToHost));
        MPI_Barrier(MPI_COMM_WORLD);
        if (i_proc != 0) {
          MPI_Reduce(expvals.data(), expvals.data(), expvals.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          stat = WAIT;
        } else {
          std::vector<ValType> temp(expvals.size());
          MPI_Reduce(expvals.data(), temp.data(), expvals.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          std::copy(temp.begin(), temp.end(), expvals.begin());
        }
        MPI_Barrier(MPI_COMM_WORLD);              
        
      };
      virtual void process_loop() {
        assert(i_proc != 0);
        while(stat != EXIT_LOOP) {
          MPI_Recv(&stat, 1, MPI_INT, 0, iteration, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          if (stat == CALL_SIMULATOR) {
            call_simulator();
            iteration++;
          }
        }

      }
      virtual void optimize(std::vector<ValType>& parameters, ValType& final_ene) override {
          iteration = 0;
          if (parameters.size() == 0) {
            parameters = std::vector<ValType>(ansatz->numParams(), 0.0);
          }
          if (i_proc == 0) {
            nlopt::result optimization_result = optimizer.optimize(parameters, final_ene);
            
            // std::cout << energy(parameters) << std::endl;iteration = 1;
            // std::cout << zmasks[1376] << std::endl;
            stat = EXIT_LOOP;
            for(IdxType i = 1; i < n_cpus; i++) {
              MPI_Send(&stat, 1, MPI_INT, i, iteration, MPI_COMM_WORLD);
            }

          } else {
            process_loop();
          }
      }
      ~SV_CUDA_MPI_VQE()
        {
            SAFE_FREE_GPU(obs.xmasks);
            SAFE_FREE_GPU(obs.zmasks);
            SAFE_FREE_GPU(obs.x_index_sizes);
            SAFE_FREE_GPU(obs.x_indices);
            SAFE_FREE_GPU(obs.exp_output);
            SAFE_FREE_GPU(ansatz->gates->back().data);
            // SAFE_FREE_GPU(randoms_gpu);
            // SAFE_FREE_GPU(gates_gpu);
            // SAFE_FREE_HOST_CUDA(randoms);
            // SAFE_FREE_HOST_CUDA(results);
        }
    protected:
        IdxType n_cpus;
        STATUS stat; 
   
  };
}; // namespace NWQSim
};
#endif