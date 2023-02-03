#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include "../src/config.h"

using namespace NWQSim;
using namespace std;

vector<string> qasm_strs;

void push_qasmstr(string filename, string op, IdxType target, IdxType ctrl = -1, IdxType has_param = 0, ValType param = 0, IdxType n_qubits = 0)
{
    stringstream ss;

    if (op == "measure")
    {
        ofstream outFile;

        if (!qasm_strs.empty())
        {
            outFile.open(filename);
            outFile << "OPENQASM 2.0;" << endl
                    << "include \"qelib1.inc\";" << endl
                    << "qreg q[" << n_qubits << "];" << endl
                    << "creg c[" << n_qubits << "];" << endl;
            for (auto s : qasm_strs)
            {
                outFile << s << endl;
            }
            qasm_strs.clear();
        }
        else
        {
            outFile.open(filename, ios_base::app);
        }
        outFile << "measure "
                << "q[" << target << "] -> c[" << target << "];" << endl;
        outFile.close();
    }
    else
    {
        ss << op;
        if (has_param != 0)
            ss << "(" << param << ")";
        ss << " ";
        if (ctrl != -1)
            ss << "q[" << ctrl << "],";
        ss << "q[" << target << "];";

        qasm_strs.push_back(ss.str());
    }
}
