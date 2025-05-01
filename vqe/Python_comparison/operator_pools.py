import openfermion
import numpy as np
import copy as cp
import scipy.sparse.linalg as ssl # Muqing: for Pauli Excitation

from openfermion import *

print("This is a modified operator_pools.py by Muqing Zheng")

class OperatorPool:
    def __init__(self):
        self.n_orb = 0
        self.n_occ_a = 0
        self.n_occ_b = 0
        self.n_vir_a = 0
        self.n_vir_b = 0

        self.n_spin_orb = 0
        self.gradient_print_thresh = 0

    def init(self,n_orb,
            n_occ_a=None,
            n_occ_b=None,
            n_vir_a=None,
            n_vir_b=None):
        self.n_orb = n_orb
        self.n_spin_orb = 2*self.n_orb

        if n_occ_a!=None and n_occ_b!=None:
            assert(n_occ_a == n_occ_b)
            self.n_occ = n_occ_a
            self.n_occ_a = n_occ_a
            self.n_occ_b = n_occ_b
            self.n_vir = n_vir_a
            self.n_vir_a = n_vir_a
            self.n_vir_b = n_vir_b
        self.n_ops = 0

        self.generate_SQ_Operators()

    def generate_SQ_Operators(self):
        print("Virtual: Reimplement")
        exit()

    def generate_SparseMatrix(self):
        self.spmat_ops = []
        print(" Generate Sparse Matrices for operators in pool")
        for op in self.fermi_ops:
            temp_sp = linalg.get_sparse_operator(op, n_qubits = self.n_spin_orb) # Muqing
            if np.allclose(ssl.norm(temp_sp.imag), 0): # Muqing
                temp_sp = temp_sp.real # Muqing
            self.spmat_ops.append(temp_sp) # Muqing
        assert(len(self.spmat_ops) == self.n_ops)
        return

#     def get_string_for_term(self,op):

#         opstring = ""
#         spins = ""
#         for t in op.terms:


#             opstring = "("
#             for ti in t:
#                 opstring += str(int(ti[0]/2))
#                 if ti[1] == 0:
#                     opstring += "  "
#                 elif ti[1] == 1:
#                     opstring += "' "
#                 else:
#                     print("wrong")
#                     exit()
#                 spins += str(ti[0]%2)

# #            if self.fermi_ops[i].terms[t] > 0:
# #                spins = "+"+spins
# #            if self.fermi_ops[i].terms[t] < 0:
# #                spins = "-"+spins
#             opstring += ")"
#             spins += " "
#         opstring = " %18s : %s" %(opstring, spins)
#         return opstring

    def get_string_for_term(self,op, final=False): # Muqing: add parameter to print operator as a whole
        if final:
            return str(op).replace("\n", "")
        if type(op) == QubitOperator: # Muqing
            return str(op) # Muqing
        opstring = ""
        spins = ""
        for t in op.terms:
            opstring = "("
            for ti in t:
                opstring += str(int(ti[0])) # Muqing: show the original orbital index
                # opstring += str(int(ti[0]/2))
                if ti[1] == 0:
                    opstring += "  "
                elif ti[1] == 1:
                    opstring += "' "
                else:
                    print("wrong")
                    exit()
                spins += str(ti[0]%2)
            opstring += ")"
            spins += " "
        opstring = " %18s : %s" %(opstring, spins)
        return opstring



    def compute_gradient_i(self,i,v,sig):
        """
        For a previously optimized state |n>, compute the gradient g(k) of exp(c(k) A(k))|n>
        g(k) = 2Real<HA(k)>

        Note - this assumes A(k) is an antihermitian operator. If this is not the case, the derived class should
        reimplement this function. Of course, also assumes H is hermitian

        v   = current_state
        sig = H*v

        """
        opA = self.spmat_ops[i]
        gi = 2*(sig.transpose().conj().dot(opA.dot(v)))
        assert(gi.shape == (1,1))
        gi = gi[0,0]
        assert(np.isclose(gi.imag,0))
        gi = gi.real

        opstring = self.get_string_for_term(self.fermi_ops[i])

        if abs(gi) >= self.gradient_print_thresh:
            print(" %4i %12.8f %s" %(i, gi, opstring) )
        return gi


class spin_complement_GSD(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        alpha_orbs = [2*i for i in range(self.n_orb)]
        beta_orbs = [2*i+1 for i in range(self.n_orb)]

        ops = []
        #aa
        for p in alpha_orbs:
            for q in alpha_orbs:
                if p>=q:
                    continue
                #if abs(hamiltonian_op.one_body_tensor[p,q]) < 1e-8:
                #    print(" Dropping term %4i %4i" %(p,q), " V= %+6.1e" %hamiltonian_op.one_body_tensor[p,q])
                #    continue
                one_elec = openfermion.FermionOperator(((q,1),(p,0)))-openfermion.FermionOperator(((p,1),(q,0)))
                one_elec += openfermion.FermionOperator(((q+1,1),(p+1,0)))-openfermion.FermionOperator(((p+1,1),(q+1,0)))
                ops.append(one_elec)
        #aa
        pq = 0
        for p in alpha_orbs:
            for q in alpha_orbs:
                if p>q:
                    continue
                rs = 0
                for r in alpha_orbs:
                    for s in alpha_orbs:
                        if r>s:
                            continue
                        if pq<rs:
                            continue
                        #if abs(hamiltonian_op.two_body_tensor[p,r,s,q]) < 1e-8:
                            #print(" Dropping term %4i %4i %4i %4i" %(p,r,s,q), " V= %+6.1e" %hamiltonian_op.two_body_tensor[p,r,s,q])
                            #continue
                        two_elec = openfermion.FermionOperator(((r,1),(p,0),(s,1),(q,0)))-openfermion.FermionOperator(((q,1),(s,0),(p,1),(r,0)))
                        two_elec += openfermion.FermionOperator(((r+1,1),(p+1,0),(s+1,1),(q+1,0)))-openfermion.FermionOperator(((q+1,1),(s+1,0),(p+1,1),(r+1,0)))
                        ops.append(two_elec)
                        rs += 1
                pq += 1


        #ab
        pq = 0
        for p in alpha_orbs:
            for q in beta_orbs:
                rs = 0
                for r in alpha_orbs:
                    for s in beta_orbs:
                        if pq<rs:
                            continue
                        two_elec = openfermion.FermionOperator(((r,1),(p,0),(s,1),(q,0)))-openfermion.FermionOperator(((q,1),(s,0),(p,1),(r,0)))
                        if p>q:
                            continue
                        two_elec += openfermion.FermionOperator(((s-1,1),(q-1,0),(r+1,1),(p+1,0)))-openfermion.FermionOperator(((p+1,1),(r+1,0),(q-1,1),(s-1,0)))
                        ops.append(two_elec)
                        rs += 1
                pq += 1

        self.fermi_ops = ops
        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return
# }}}




class spin_complement_GSD2(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form spin-complemented GSD operators")

        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1

            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1

                termA =  FermionOperator(((pa,1),(qa,0)))
                termA += FermionOperator(((pb,1),(qb,0)))

                termA -= hermitian_conjugated(termA)

                termA = normal_ordered(termA)

                if termA.many_body_order() > 0:
                    self.fermi_ops.append(termA)


        pq = -1
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1

            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1

                pq += 1

                rs = -1
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1

                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1

                        rs += 1

                        if(pq > rs):
                            continue

                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)))
                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)))

                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)))
                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)))

                        termC =  FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)))
                        termC += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)))

#                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)))
#                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)))
#
#                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)))
#                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)))
#
#                        termC =  FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)))
#                        termC += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)))

#                        print()
#                        print(p,q,r,s)
#                        print(termA)
#                        print(termB)
#                        print(termC)
                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)
                        termC -= hermitian_conjugated(termC)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)
                        termC = normal_ordered(termC)

                        if termA.many_body_order() > 0:
                            self.fermi_ops.append(termA)

                        if termB.many_body_order() > 0:
                            self.fermi_ops.append(termB)

                        if termC.many_body_order() > 0:
                            self.fermi_ops.append(termC)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return
# }}}




class singlet_GSD(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form singlet GSD operators")

        self.fermi_ops = []
        single_counter = 0
        double_singlet_counter = 0
        double_triplet_counter = 0
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1

            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1

                termA =  FermionOperator(((pa,1),(qa,0)))
                termA += FermionOperator(((pb,1),(qb,0)))

                termA -= hermitian_conjugated(termA)

                termA = normal_ordered(termA)

                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
                
                print(f"Ansatz {p} {q}")
                if termA.many_body_order() > 0:
                    single_counter += 1
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)
                    print(f"  Single: {p} {q}")


        pq = -1
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1

            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1

                pq += 1

                rs = -1
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1

                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1

                        rs += 1

                        if(pq > rs):
                            continue

#                        oplist = []
#                        oplist.append(FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12)))
#                        oplist.append(FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12)))
#                        oplist.append(FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12)))
#                        oplist.append(FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12)))
#                        oplist.append(FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12)))
#                        oplist.append(FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12)))
#
#                        print(p,q,r,s)
#                        for i in range(len(oplist)):
#                            oplist[i] -= hermitian_conjugated(oplist[i])
#                        for i in range(len(oplist)):
#                            for j in range(i+1,len(oplist)):
#                                opi = oplist[i]
#                                opj = oplist[i]
#                                opij = opi*opj - opj*opi
#                                opij = normal_ordered(opij)
#                                print(opij, end='')
#                        print()
                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))
                        termA_copy = termA

                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)
                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)
                        termB += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                        termB += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)
                        termB_copy = termB

                        termA -= hermitian_conjugated(termA) 
                        termB -= hermitian_conjugated(termB)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)

                        #Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        # print(f"Ansatz {r} {s} {p} {q}")
                        if termA.many_body_order() > 0:
                            double_triplet_counter += 1
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)
                            print(f"  Triplet{len(termA.terms)//2}t: {r} {s} {p} {q}" )
                            print(termA)
                            print()

                        if termB.many_body_order() > 0:
                            double_singlet_counter += 1
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)
                            print(f"  Singlet{len(termB.terms)//2}t: {r} {s} {p} {q}" )
                            print(termB)
                            print()

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        print(f" Single excitations: {single_counter}", f" Double singlet excitations: {double_singlet_counter}", f" Double triplet excitations: {double_triplet_counter}")
        return
# }}}




class singlet_SD(OperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form singlet SD operators")
        self.fermi_ops = []

        assert(self.n_occ_a == self.n_occ_b)
        n_occ = self.n_occ
        n_vir = self.n_vir

        for i in range(0,n_occ):
            ia = 2*i
            ib = 2*i+1

            for a in range(0,n_vir):
                aa = 2*n_occ + 2*a
                ab = 2*n_occ + 2*a+1

                termA =  FermionOperator(((aa,1),(ia,0)), 1/np.sqrt(2))
                termA += FermionOperator(((ab,1),(ib,0)), 1/np.sqrt(2))

                termA -= hermitian_conjugated(termA)

                termA = normal_ordered(termA)

                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)


        for i in range(0,n_occ):
            ia = 2*i
            ib = 2*i+1

            for j in range(i,n_occ):
                ja = 2*j
                jb = 2*j+1

                for a in range(0,n_vir):
                    aa = 2*n_occ + 2*a
                    ab = 2*n_occ + 2*a+1

                    for b in range(a,n_vir):
                        ba = 2*n_occ + 2*b
                        bb = 2*n_occ + 2*b+1

                        termA =  FermionOperator(((aa,1),(ba,1),(ia,0),(ja,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(bb,1),(ib,0),(jb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), 1/np.sqrt(12))

                        termB  = FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/2)
                        termB += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/2)
                        termB += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), -1/2)
                        termB += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), -1/2)

                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)

                        #Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t


                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)

                        if termB.many_body_order() > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return
    # }}}



def unrestricted_SD(n_occ_a, n_occ_b, n_vir_a, n_vir_b):
    print("NYI")
    exit()
    
    
    
    
    
## Muqing: Implement pool for Qubit-ADAPT-VQE
## Reference: https://github.com/JordanovSJ/VQE/blob/78ff043c2487b379ba2d4d89af1347988ebccebd/scripts/iter_vqe/qubit_adapt_vqe.py
##            https://github.com/JordanovSJ/VQE/blob/78ff043c2487b379ba2d4d89af1347988ebccebd/src/ansatz_element_sets.py#L134
import itertools
class pauli_exc(OperatorPool):
    def xyreplace(self, rstr):
        temp_str = rstr.replace('Y', 'A')
        X2Y_str = temp_str.replace('X', 'Y')
        Y2X_str = X2Y_str.replace('A', 'X')
        return Y2X_str
    
    def generate_SQ_Operators(self):
        print(" Form Pauli excitation operators")
        total_strs = []
        self.fermi_ops = []
        ### single_excitation_elements
        ## SQExc(i, j).excitations_generators https://github.com/JordanovSJ/VQE/blob/78ff043c2487b379ba2d4d89af1347988ebccebd/src/ansatz_element_sets.py#L152
        for i, j in itertools.combinations(range(self.n_spin_orb), 2):
            termA  = (  0.5*QubitOperator('X{:d}'.format(i))+0.5j*QubitOperator('Y{:d}'.format(i))  )
            termA *= (  0.5*QubitOperator('X{:d}'.format(j))-0.5j*QubitOperator('Y{:d}'.format(j))  )
            termA -= hermitian_conjugated(termA)
            
            ## To individual Pauli string
            for pt in termA.terms.keys():
                temp_op = 1j*QubitOperator(pt)
                if temp_op.action_strings.count('Y')%2 == 1:## Muqing: only include odd Y's
                    if self.xyreplace( str(temp_op) ) not in total_strs:
                        total_strs.append( str(temp_op) )
                        self.fermi_ops.append( temp_op )
                
        ### Double excitations
        ## DQExc([i, j], [k, l]).excitations_generators https://github.com/JordanovSJ/VQE/blob/78ff043c2487b379ba2d4d89af1347988ebccebd/src/ansatz_element_sets.py#L177
        for i, j, k, l in itertools.combinations(range(self.n_spin_orb), 4):
            termB = QubitOperator('')
            for q1, q2 in zip([i,j], [k,l]):
                termB *= (  0.5*QubitOperator('X{:d}'.format(q1))+0.5j*QubitOperator('Y{:d}'.format(q1))  )
                termB *= (  0.5*QubitOperator('X{:d}'.format(q2))-0.5j*QubitOperator('Y{:d}'.format(q2))  )
            termB -= hermitian_conjugated(termB)
            
            ## To individual Pauli string
            for pt in termB.terms.keys():
                temp_op = 1j*QubitOperator(pt)
                if temp_op.action_strings.count('Y')%2 == 1:## Muqing: only include odd Y's
                    if self.xyreplace( str(temp_op) ) not in total_strs:
                        total_strs.append( str(temp_op) )
                        self.fermi_ops.append( temp_op )

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return
# }}}
    

import numpy.random as nr
class pauli_exc4(OperatorPool):
    def xyreplace(self, rstr):
        temp_str = rstr.replace('Y', 'A')
        X2Y_str = temp_str.replace('X', 'Y')
        Y2X_str = X2Y_str.replace('A', 'X')
        return Y2X_str

    def generate_SQ_Operators(self):
        print(" Form Pauli excitation operators")
        nr.seed(63199)

        total_strs = []
        self.fermi_ops = []
        ### single_excitation_elements
        ## SQExc(i, j).excitations_generators https://github.com/JordanovSJ/VQE/blob/78ff043c2487b379ba2d4d89af1347988ebccebd/src/ansatz_element_sets.py#L152
        for i, j in itertools.combinations(range(self.n_spin_orb), 2):
            termA  = (  0.5*QubitOperator('X{:d}'.format(i))+0.5j*QubitOperator('Y{:d}'.format(i))  )
            termA *= (  0.5*QubitOperator('X{:d}'.format(j))-0.5j*QubitOperator('Y{:d}'.format(j))  )
            termA -= hermitian_conjugated(termA)
            
            ## To individual Pauli string
            for pt in termA.terms.keys():
                temp_op = 1j*QubitOperator(pt)
                if temp_op.action_strings.count('Y')%2 == 1:## Muqing: only include odd Y'
                    if self.xyreplace( str(temp_op) ) not in total_strs:
                        total_strs.append( str(temp_op) )
                        if nr.random() < 0.25:
                            self.fermi_ops.append( temp_op )
                
        ### Double excitations
        ## DQExc([i, j], [k, l]).excitations_generators https://github.com/JordanovSJ/VQE/blob/78ff043c2487b379ba2d4d89af1347988ebccebd/src/ansatz_element_sets.py#L177
        for i, j, k, l in itertools.combinations(range(self.n_spin_orb), 4):
            termB = QubitOperator('')
            for q1, q2 in zip([i,j], [k,l]):
                termB *= (  0.5*QubitOperator('X{:d}'.format(q1))+0.5j*QubitOperator('Y{:d}'.format(q1))  )
                termB *= (  0.5*QubitOperator('X{:d}'.format(q2))-0.5j*QubitOperator('Y{:d}'.format(q2))  )
            termB -= hermitian_conjugated(termB)
            
            ## To individual Pauli string
            for pt in termB.terms.keys():
                temp_op = 1j*QubitOperator(pt)
                if temp_op.action_strings.count('Y')%2 == 1:## Muqing: only include odd Y's
                    if self.xyreplace( str(temp_op) ) not in total_strs:
                        total_strs.append( str(temp_op) )
                        if nr.random() < 0.25: #if nr.random() < 0.25:
                            self.fermi_ops.append( temp_op )

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return
    


    


class pauli_excV(OperatorPool):
    ## (2n-2)-size pool in Section B in https://arxiv.org/pdf/1911.10205.pdf
    def op_gen_helper(self, cn, n_orbs, str_set):
        if cn <= 1 or n_orbs<=1:
            raise Exception('Must have at least 2 orbitals')
        if cn > n_orbs: ## end the recurision
            return str_set
        if cn == 2:
            return self.op_gen_helper(cn+1, n_orbs, str_set)
        tmp_set = []
        newZ = 'Z{:d}'.format(cn-1)
        for kn in str_set:
            tmp_set.append(newZ+' '+kn)
        str_set = tmp_set + ['Y{:d}'.format(cn-1), 'Y{:d}'.format(cn-2)]
        return self.op_gen_helper(cn+1, n_orbs, str_set)

    def generate_SQ_Operators(self):
        print(" Form reduced Pauli excitation operators V")
        str_set = ['Z1 Y0', 'Y1']
        opstr_set = self.op_gen_helper(2, self.n_spin_orb, str_set)
        self.fermi_ops = [1j*QubitOperator(termA) for termA in opstr_set]
        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return
    
class pauli_excG(OperatorPool):
    ## (2n-2)-size pool in Section B in https://arxiv.org/pdf/1911.10205.pdf
    def generate_SQ_Operators(self):
        print(" Form reduced Pauli excitation operators G")
        opstr_set = []
        for kk in range(self.n_spin_orb - 1):
            str1 = 'Z{:d} Y{:d}'.format(kk, kk+1)
            str2 = 'Y{:d}'.format(kk)
            opstr_set.append(str1)
            opstr_set.append(str2)
        self.fermi_ops = [1j*QubitOperator(termA) for termA in opstr_set]
        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return
