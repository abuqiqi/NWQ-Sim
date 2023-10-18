#pragma once

#include "backendManager.hpp"
#include "state.hpp"
#include "circuit.hpp"
#include "nwq_util.hpp"

#include <iostream>
#include <memory>
#include <vector>
#include <iomanip>

namespace NWQSim
{

    inline std::shared_ptr<Circuit> get_custom_circuit()
    {
        int n_qubits = 3;

        std::shared_ptr<Circuit> circuit = std::make_shared<Circuit>(n_qubits);
        circuit->X(0);

        return circuit;
    }

    // Function to print a square complex matrix as a 2D grid
    void printDM(std::vector<std::complex<ValType>> state)
    {
        int size = state.size();
        int sideLength = static_cast<int>(std::sqrt(size));

        if (sideLength * sideLength != size)
        {
            std::cout << "Invalid dm size." << std::endl;
            return;
        }

        int width = 16;    // Adjust the width as needed for your data
        int precision = 4; // Adjust the number of decimal places as needed

        for (int row = 0; row < sideLength; ++row)
        {
            for (int col = 0; col < sideLength; ++col)
            {
                int index = row * sideLength + col;
                std::cout << std::left << std::fixed << std::setw(width) << std::setprecision(precision) << state[index] << " ";
            }
            std::cout << std::endl;
        }
    }
    inline void post_processing(std::vector<std::complex<ValType>> state)
    {
        printDM(state);
    }

}