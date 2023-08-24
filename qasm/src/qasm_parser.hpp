#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <chrono>

#include "parser_util.hpp"
#include "lexer.hpp"

#include "nwq_util.hpp"
#include "state.hpp"
#include "circuit.hpp"

using namespace std;
using namespace NWQSim;
using namespace lexertk;

class qasm_parser
{
private:
    /* data */
    map<string, qreg> list_qregs;
    map<string, creg> list_cregs;

    map<string, defined_gate> list_defined_gates;
    vector<qasm_gate> *list_gates = NULL;
    vector<qasm_gate> *list_conditional_gates = NULL;
    vector<qasm_gate> list_buffered_measure;

    IdxType global_qubit_offset = 0;

    vector<token> cur_inst;

    bool dynamic_circuit = false;
    bool skip_if = false;

    /* Lexer Object */
    generator gen;
    helper::symbol_replacer sr;

    /* File Loading Util */
    ifstream qasmFile;
    string line;
    stringstream ss;

    // MID MEASUREMENT
    int measurement_count = 0;
    bool force_measure = false;

    /* Helper Functions */
    void load_instruction();
    void parse_gate_defination();

    void parse_gate(vector<token> &inst, vector<qasm_gate> *gates);
    void parse_native_gate(vector<token> &inst, vector<qasm_gate> *gates);
    void parse_defined_gate(vector<token> &inst, vector<qasm_gate> *gates);

    void classify_measurements();

    bool execute_gate(shared_ptr<QuantumState> state, std::shared_ptr<NWQSim::Circuit> circuit, qasm_gate gate);
    IdxType *sub_execute(shared_ptr<QuantumState> state, IdxType repetition, bool print_metrics);

    void dump_defined_gates();
    void dump_cur_inst();
    void dump_gates();

public:
    qasm_parser(const char *filename);
    IdxType num_qubits();
    map<string, IdxType> *execute(shared_ptr<QuantumState> state, IdxType repetition, bool print_metrics = false, bool _force_measure = false);
    ~qasm_parser();
};

qasm_parser::qasm_parser(const char *filename)
{
    qasmFile.open(filename);
    if (!qasmFile)
        throw runtime_error(string("Could not open qasm file at:") + filename);

    sr.add_replace("pi", "pi", token::e_pi);
    sr.add_replace("sin", "sin", token::e_func);
    sr.add_replace("cos", "cos", token::e_func);

    list_gates = new vector<qasm_gate>;
    list_conditional_gates = new vector<qasm_gate>;

    while (!qasmFile.eof())
    {
        load_instruction();

        // dump_cur_inst();
        if (cur_inst.size() > 0)
        {
            if (cur_inst[INST_NAME].value == OPENQASM)
            // parse OpenQASM version
            {
                // cout << "Executing with OpenQASM " << cur_inst[INST_QASM_VERSION].value << endl;
            }
            else if (cur_inst[INST_NAME].value == QREG)
            // parse qubit registers
            {
                qreg qreg;
                qreg.name = cur_inst[INST_REG_NAME].value;
                qreg.width = stoi(cur_inst[INST_REG_WIDTH].value);
                qreg.offset = global_qubit_offset;

                global_qubit_offset += qreg.width;

                list_qregs.insert({qreg.name, qreg});

                if (global_qubit_offset > 63)
                    skip_if = true;
            }
            else if (cur_inst[INST_NAME].value == CREG)
            // parse classical registers
            {
                creg creg;

                creg.name = cur_inst[INST_REG_NAME].value;
                creg.width = stoi(cur_inst[INST_REG_WIDTH].value);

                creg.qubit_indices.insert(creg.qubit_indices.end(), creg.width, UN_DEF);

                list_cregs.insert({creg.name, creg});
            }
            else if (cur_inst[INST_NAME].value == GATE)
            // parse custom gate definations
            {
                parse_gate_defination();
            }
            else if (cur_inst[INST_NAME].value == IF)
            // parse if statement
            {
                if (!skip_if)
                {
                    qasm_gate cur_gate;
                    cur_gate.name = IF;
                    cur_gate.creg_name = cur_inst[INST_IF_CREG].value;
                    cur_gate.if_creg_val = stoll(cur_inst[INST_IF_VAL].value);
                    cur_gate.conditional_inst = new vector<qasm_gate>;

                    auto c_inst = slices(cur_inst, INST_IF_INST_START, cur_inst.size() - 1);
                    parse_gate(c_inst, cur_gate.conditional_inst);

                    list_gates->push_back(cur_gate);

                    dynamic_circuit = true;
                }
            }
            else
            // parse quantum gates
            {
                parse_gate(cur_inst, list_gates);
            }
        }
    }
    classify_measurements();

    // dump_defined_gates();
    // dump_gates();
}

void qasm_parser::load_instruction()
{
    bool has_eof = false, has_lcurly = false, has_rcurly = false;

    cur_inst.clear();

    getline(qasmFile, line);
    transform(line.begin(), line.end(), line.begin(), ::toupper);

    if (!gen.process(line))
        return;

    if (gen.size() > 0)
    {
        sr.process(gen);

        for (size_t i = 0; i < gen.size(); i++)
        {
            cur_inst.push_back(gen[i]);

            switch (gen[i].type)
            {
            case token::e_eof:
                has_eof = true;
                break;
            case token::e_lcrlbracket:
                has_lcurly = true;
                break;
            case token::e_rcrlbracket:
                has_rcurly = true;
                break;
            default:
                break;
            }
        }

        ss.str(string());
        if (has_rcurly || (has_eof && !has_lcurly))
        {
        }
        else if (has_lcurly)
        {
            while (ss.str().find('}') == string::npos)
            {
                getline(qasmFile, line);
                ss << line;
            }
        }
        else
        {
            while (ss.str().find(';') == string::npos)
            {
                getline(qasmFile, line);
                ss << line;
            }
            if (ss.str().find('{') != string::npos)
            {
                while (ss.str().find('}') == string::npos)
                {
                    getline(qasmFile, line);
                    ss << line;
                }
            }
        }
        line = ss.str();
        transform(line.begin(), line.end(), line.begin(), ::toupper);
        ss.str(string());

        gen.process(line);
        sr.process(gen);

        for (size_t i = 0; i < gen.size(); i++)
        {
            cur_inst.push_back(gen[i]);
        }
    }
}

void qasm_parser::dump_cur_inst()
{
    for (size_t i = 0; i < cur_inst.size(); ++i)
    {
        printf("%s ", cur_inst[i].value.c_str());
    }
    cout << endl;
}

void qasm_parser::parse_gate_defination()
{
    defined_gate defined_gate;
    defined_gate.name = cur_inst[INST_GATE_NAME].value;

    IdxType lcurly_pos = -1;
    for (size_t i = 0; i < cur_inst.size(); i++)
        if (cur_inst[i].type == token::e_lcrlbracket)
        {
            lcurly_pos = i;
            break;
        }

    inst_indicies gate_indices = get_indices(cur_inst, 1, lcurly_pos);
    if (gate_indices.param_start != -1)
    {
        for (auto p : slices(cur_inst, gate_indices.param_start, gate_indices.param_end))
        {
            if (p.type == token::e_comma)
                continue;
            else if (p.type == token::e_symbol)
                defined_gate.params.push_back(p.value);
            else
                cout << "INVALID PARAM FOR GATE DEFINATION " << p.value << endl;
        }
    }

    for (auto q : slices(cur_inst, gate_indices.qubit_start, gate_indices.qubit_end))
    {
        if (q.type == token::e_comma)
            continue;
        else if (q.type == token::e_symbol)
            defined_gate.qubits.push_back(q.value);
        else
            cout << "INVALID PARAM FOR GATE DEFINATION " << q.value << endl;
    }

    IdxType cur_start = lcurly_pos + 1;

    for (size_t i = lcurly_pos + 1; i < cur_inst.size(); i++)
        if (cur_inst[i].type == token::e_eof)
        {
            defined_gate.instructions.push_back(slices(cur_inst, cur_start, i + 1));
            cur_start = i + 1;
        }
    list_defined_gates.insert({defined_gate.name, defined_gate});
}

void qasm_parser::parse_gate(vector<token> &inst, vector<qasm_gate> *gates)
{
    if (inst[INST_NAME].value == MEASURE)
    {
        if (inst.size() == 12)
        {
            qasm_gate cur_gate;
            cur_gate.name = MEASURE;
            cur_gate.measured_qubit_index = list_qregs.at(inst[INST_MEASURE_QREG_NAME].value).offset + stoll(inst[INST_MEASURE_QREG_BIT].value);
            cur_gate.creg_name = inst[INST_MEASURE_CREG_NAME].value;
            cur_gate.creg_index = stoll(inst[INST_MEASURE_CREG_BIT].value);
            gates->push_back(cur_gate);
        }
        else
        {
            qreg qreg = list_qregs.at(inst[INST_MEASURE_QREG_NAME].value);

            for (IdxType i = 0; i < qreg.width; i++)
            {
                qasm_gate cur_gate;
                cur_gate.name = MEASURE;
                cur_gate.measured_qubit_index = qreg.offset + i;
                cur_gate.creg_name = inst[4].value;
                cur_gate.creg_index = i;
                gates->push_back(cur_gate);
            }
        }
    }
    else
    {
        auto it = list_defined_gates.find(inst[INST_NAME].value);

        if (it != list_defined_gates.end())
            parse_defined_gate(inst, gates);
        else if (find(begin(DEFAULT_GATES), end(DEFAULT_GATES), inst[INST_NAME].value) != end(DEFAULT_GATES))
            parse_native_gate(inst, gates);
        else if (inst[INST_NAME].value != BARRIER)
        {
            cout << "Undefined instruction: ";
            for (auto t : inst)
                cout << t.value << " ";
            cout << endl;
        }
    }
}

void qasm_parser::parse_native_gate(vector<token> &inst, vector<qasm_gate> *gates)
{
    inst_indicies indices = get_indices(inst, 0, inst.size());

    auto params = get_params(inst, indices.param_start, indices.param_end);
    auto qubits = get_qubits(inst, indices.qubit_start, indices.qubit_end, list_qregs);

    for (IdxType i = 0; i < qubits.first; i++)
    {
        qasm_gate gate;
        gate.name = inst[INST_NAME].value;

        for (auto p : params)
            gate.params.push_back(p);

        for (size_t j = 0; j < qubits.second.size(); j++)
        {
            if (qubits.second[j].size() == 1)
                gate.qubits.push_back(qubits.second[j][0]);
            else
                gate.qubits.push_back(qubits.second[j][i]);
        }
        gates->push_back(gate);
    }
}

IdxType find_index(vector<string> &vec, string target)
{
    for (size_t i = 0; i < vec.size(); i++)
        if (vec[i] == target)
            return i;

    return -1;
}

void dump_inst(vector<token> &inst)
{
    for (size_t i = 0; i < inst.size(); ++i)
    {
        printf("%s ", inst[i].value.c_str());
    }
    cout << endl;
}

void qasm_parser::parse_defined_gate(vector<token> &inst, vector<qasm_gate> *gates)
{
    auto gate_def = list_defined_gates.at(inst[INST_NAME].value);

    auto indices = get_indices(inst, 0, inst.size());

    auto params = get_params(inst, indices.param_start, indices.param_end);
    auto qubits = get_qubits(inst, indices.qubit_start, indices.qubit_end, list_qregs);

    for (IdxType i = 0; i < qubits.first; i++)
    {
        vector<IdxType> cur_qubits;

        for (size_t j = 0; j < qubits.second.size(); j++)
            if (qubits.second[j].size() == 1)
                cur_qubits.push_back(qubits.second[j][0]);
            else
                cur_qubits.push_back(qubits.second[j][i]);

        for (auto sub_inst : gate_def.instructions)
        {
            vector<token> dup_inst(sub_inst);

            for (auto &t : dup_inst)
            {
                IdxType param_idx = find_index(gate_def.params, t.value);
                IdxType qubit_idx = find_index(gate_def.qubits, t.value);

                if (param_idx != -1 && qubit_idx != -1)
                    throw runtime_error("Can't use same symbol for both parameter and qubits");

                if (param_idx != -1)
                {
                    t.type = token::e_number;
                    t.value = to_string(params[param_idx]);
                }
                else if (qubit_idx != -1)
                {
                    t.type = token::e_number;
                    t.value = to_string(cur_qubits[qubit_idx]);
                }
            }
            parse_gate(dup_inst, gates);
        }
    }
}

void qasm_parser::dump_defined_gates()
{
    for (const auto &entry : list_defined_gates)
    {
        auto gatename = entry.first;
        auto gate = entry.second;
        cout << gatename << endl
             << "Params " << gate.params.size() << ":";

        for (auto p : gate.params)
            cout << p << " ";
        cout << endl
             << "Qubits " << gate.qubits.size() << ":";

        for (auto q : gate.qubits)
            cout << q << " ";
        cout << endl
             << "Insts:\n";

        for (auto vec : gate.instructions)
        {
            for (auto t : vec)
                cout << t.value << " ";
            cout << endl;
        }
        cout << endl;
    }
}

void print_gate(qasm_gate gate, bool indent = false)
{
    if (indent)
        cout << "\t";
    if (gate.name == MEASURE)
    {
        cout << "M " << gate.measured_qubit_index << " -> " << gate.creg_name << "[" << gate.creg_index << "];\n";
    }
    else
    {
        cout << gate.name << " ";

        if (gate.params.size() > 0)
        {
            cout << "\b(";
            for (auto p : gate.params)
                cout << p << ",";
            cout << "\b) ";
        }

        for (auto q : gate.qubits)
            cout << q << ",";

        cout << "\b;\n";
    }
}

void qasm_parser::dump_gates()
{
    for (auto gate : *list_gates)
    {
        if (gate.name == IF)
        {
            cout << gate.name << " " << gate.creg_name << " == " << gate.if_creg_val << ":\n";

            for (auto c_gate : *gate.conditional_inst)
            {
                print_gate(c_gate, true);
            }
        }
        else
        {
            print_gate(gate);
        }
    }
}

IdxType qasm_parser::num_qubits() { return global_qubit_offset; }

void qasm_parser::classify_measurements()
{
    bool final_measurements = true;

    for (IdxType i = list_gates->size() - 1; i >= 0; i--)
        if (list_gates->at(i).name == MEASURE)
        {
            list_gates->at(i).final_measurements = final_measurements;

            if (!final_measurements)
                dynamic_circuit = true;
        }
        else
            final_measurements = false;
}

map<string, IdxType> *qasm_parser::execute(shared_ptr<QuantumState> state, IdxType repetition, bool print_metrics, bool _force_measure)
{
    IdxType *results;

    force_measure = _force_measure;
    if (dynamic_circuit && !force_measure)
    {
        auto start = std::chrono::steady_clock::now();

        results = new IdxType[repetition];

        for (IdxType i = 0; i < repetition; i++)
        {
            // printProgressBar(i, repetition, start);

            IdxType *sub_result = sub_execute(state, 1, print_metrics);

            // MID MEASUREMENT
            if (sub_result == NULL)
            {
                results[i] = -measurement_count; // APPEND (-) num measures succeded if didn't finish
                printf("Trail %d failed at %d measurement\n", i, measurement_count);
            }
            else
            {
                results[i] = sub_result[0];
                printf("%dth trail succeeded!\n", i);
            }
        }
    }
    else
    {
        results = sub_execute(state, repetition, print_metrics);
    }

    map<IdxType, IdxType> result_dict;

    for (IdxType i = 0; i < repetition; i++)
    {
        if (result_dict.find(results[i]) != result_dict.end())
            result_dict[results[i]] += 1;
        else
            result_dict.insert({results[i], 1});
    }

    return convert_dictionary(result_dict, list_cregs);
}

IdxType *qasm_parser::sub_execute(shared_ptr<QuantumState> state, IdxType repetition, bool print_metrics)
{
    state->reset_state();

    measurement_count = 0;

    std::shared_ptr<NWQSim::Circuit> circuit = std::make_shared<Circuit>(num_qubits());

    int n_succeed = 0;

    for (auto gate : *list_gates)
    {
        if (gate.name == IF)
        {
            creg creg = list_cregs.at(gate.creg_name);

            if (creg.val == gate.if_creg_val)
            {
                for (auto c_gate : *gate.conditional_inst)
                    execute_gate(state, circuit, c_gate);
            }
        }
        else
        {
            bool proceed = execute_gate(state, circuit, gate);

            if (!proceed)
            {
                return NULL;
            }
            else
            {
                n_succeed++;
            }
        }
    }

    if (!circuit->is_empty())
    {
        circuit->MA(repetition);

        if (print_metrics)
            circuit->print_metrics();
        state->sim(circuit);
    }

    return state->get_results();
}

bool qasm_parser::execute_gate(shared_ptr<QuantumState> state, std::shared_ptr<NWQSim::Circuit> circuit, qasm_gate gate)
{
    auto gate_name = gate.name;
    auto params = gate.params;
    auto qubits = gate.qubits;

    if (gate.name == MEASURE)
    {
        if (!gate.final_measurements)
        {
            if (force_measure)
            {
                circuit->M(gate.measured_qubit_index);
            }
            else
            {
                measurement_count++;

                if (!circuit->is_empty())
                {
                    state->sim(circuit);
                    circuit->clear();
                }

                // Measure and update creg value for intermediate measurements
                IdxType result = state->measure(gate.measured_qubit_index);

                // MID MEASUREMENT
                if (result != 0)
                    return false;
            }
        }
        else
        {
            // Update creg qubit indices for final measurements
            list_cregs.at(gate.creg_name).qubit_indices[gate.creg_index] = gate.measured_qubit_index;
        }
    }
    else if (gate_name == "U")
        circuit->U(params[0], params[1], params[2], qubits[0]);
    else if (gate_name == "U1")
        circuit->U1(params[0], qubits[0]); // circuit->U1(0, 0, params[0], qubits[0]);
    else if (gate_name == "U2")
        circuit->U2(params[0], params[1], qubits[0]); // circuit->U2(pi / 2, params[0], params[1], qubits[0]);
    else if (gate_name == "U3")
        circuit->U3(params[0], params[1], params[2], qubits[0]);
    else if (gate_name == "X")
        circuit->X(qubits[0]);
    else if (gate_name == "Y")
        circuit->Y(qubits[0]);
    else if (gate_name == "Z")
        circuit->Z(qubits[0]);
    else if (gate_name == "H")
        circuit->H(qubits[0]);
    else if (gate_name == "S")
        circuit->S(qubits[0]);
    else if (gate_name == "SDG")
        circuit->SDG(qubits[0]);
    else if (gate_name == "T")
        circuit->T(qubits[0]);
    else if (gate_name == "TDG")
        circuit->TDG(qubits[0]);
    else if (gate_name == "RX")
        circuit->RX(params[0], qubits[0]);
    else if (gate_name == "RY")
        circuit->RY(params[0], qubits[0]);
    else if (gate_name == "RZ")
        circuit->RZ(params[0], qubits[0]);
    else if (gate_name == "CX")
        circuit->CX(qubits[0], qubits[1]);
    else if (gate_name == "CY")
        circuit->CY(qubits[0], qubits[1]);
    else if (gate_name == "CZ")
        circuit->CZ(qubits[0], qubits[1]);
    else if (gate_name == "CH")
        circuit->CH(qubits[0], qubits[1]);
    else if (gate_name == "CCX")
        circuit->CCX(qubits[0], qubits[1], qubits[2]);
    else if (gate_name == "CRX")
        circuit->CRX(params[0], qubits[0], qubits[1]);
    else if (gate_name == "CRY")
        circuit->CRY(params[0], qubits[0], qubits[1]);
    else if (gate_name == "CRZ")
        circuit->CRZ(params[0], qubits[0], qubits[1]);
    else if (gate_name == "CU")
        circuit->CU(params[0], params[1], params[2], params[3], qubits[0], qubits[1]);
    else if (gate_name == "CU1")
        circuit->CU(0, 0, params[0], 0, qubits[0], qubits[1]);
    else if (gate_name == "CU3")
        circuit->CU(params[0], params[1], params[2], 0, qubits[0], qubits[1]);
    else if (gate_name == "RESET")
        circuit->RESET(qubits[0]);
    else if (gate_name == "SWAP")
        circuit->SWAP(qubits[0], qubits[1]);
    else if (gate_name == "SX")
        circuit->SX(qubits[0]);

    else if (gate_name == "RI")
        circuit->RI(params[0], qubits[0]);
    else if (gate_name == "P")
        circuit->P(params[0], qubits[0]);

    else if (gate_name == "CS")
        circuit->CS(qubits[0], qubits[1]);
    else if (gate_name == "CSDG")
        circuit->CSDG(qubits[0], qubits[1]);
    else if (gate_name == "CT")
        circuit->CT(qubits[0], qubits[1]);
    else if (gate_name == "CTDG")
        circuit->CTDG(qubits[0], qubits[1]);
    else if (gate_name == "CSX")
        circuit->CSX(qubits[0], qubits[1]);
    else if (gate_name == "CP")
        circuit->CP(params[0], qubits[0], qubits[1]);
    else if (gate_name == "CSWAP")
        circuit->CSWAP(qubits[0], qubits[1], qubits[2]);
    else if (gate_name == "ID")
        circuit->ID(qubits[0]);
    else if (gate_name == "RXX")
        circuit->RXX(params[0], qubits[0], qubits[1]);
    else if (gate_name == "RYY")
        circuit->RYY(params[0], qubits[0], qubits[1]);
    else if (gate_name == "RZZ")
        circuit->RZZ(params[0], qubits[0], qubits[1]);
    else
        throw logic_error("Undefined gate is called!");

    // MID MEASUREMENT
    return true;
}

qasm_parser::~qasm_parser()
{
    if (list_gates != NULL)
    {
        for (auto gate : *list_gates)
            if (gate.name == IF)
                delete gate.conditional_inst;
        delete list_gates;
    }
}
