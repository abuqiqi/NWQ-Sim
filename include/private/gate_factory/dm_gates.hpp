#pragma once

#include <vector>
#include <complex>

#include "device_noise.hpp"
#include "../sim_gate.hpp"
#include "../config.hpp"

namespace NWQSim
{

    std::vector<DMGate> getDMGates(const std::vector<Gate> &gates, const IdxType n_qubits)
    {
        std::vector<DMGate> sim_dm_gates;

        for (const auto &g : gates)
        {

            if (g.op_name == OP::RESET)
            {
                sim_dm_gates.push_back(DMGate(OP::RESET, g.qubit, g.ctrl));
            }
            else if (g.op_name == OP::M)
            {
                if (Config::ENABLE_NOISE)
                {
                    std::complex<double> noisy_operator[4][4] = {};
                    getMeasureSP(noisy_operator[0], g.qubit);

                    DMGate noisy_dm_gate(OP::C2, g.qubit, g.ctrl);
                    noisy_dm_gate.set_gm(noisy_operator[0], 4);

                    sim_dm_gates.push_back(noisy_dm_gate);
                }

                sim_dm_gates.push_back(DMGate(OP::M, g.qubit, g.ctrl));
            }
            else if (g.op_name == OP::MA)
            {
                if (Config::ENABLE_NOISE)
                {
                    for (IdxType i = 0; i < n_qubits; i++)
                    {
                        std::complex<double> noisy_operator[4][4] = {};
                        getMeasureSP(noisy_operator[0], i);

                        DMGate noisy_dm_gate(OP::C2, i, g.ctrl);
                        noisy_dm_gate.set_gm(noisy_operator[0], 4);

                        sim_dm_gates.push_back(noisy_dm_gate);
                    }
                }
                sim_dm_gates.push_back(DMGate(OP::MA, g.repetition, g.ctrl));
            }
            else
            {
                auto gate = generateDMGate(g.op_name, g.qubit, g.ctrl, g.theta);
                if (g.op_name == OP::DELAY)
                    for (int i = 0; i < 4; ++i)
                    {
                        for (int j = 0; j < 4; ++j)
                        {
                            double real_part = gate.gm_real[i * 4 + j];
                            double imag_part = gate.gm_imag[i * 4 + j];

                            std::cout << std::setw(6) << std::fixed << std::setprecision(3) << real_part << "+" << std::setw(5) << imag_part << "j ";
                        }
                        std::cout << std::endl;
                    }

                sim_dm_gates.push_back(gate);
            }
        }
        return sim_dm_gates;
    }
}
