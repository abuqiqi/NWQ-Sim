import numpy as np # Muqing
import scipy
import scipy.linalg as sl

from pyscf.tools import molden

import openfermion as of
from openfermion import *

## Origianl ADAPT-VQE code files with some modifications
from tVQE import *
import vqe_methods 
import operator_pools
import pyscf_helper 


class VQEMole():
    def __init__(self):
        self.geometry = None
        self.mol_unit = None

        self.mol_name = None
        self.mol_name_pic = None

        self.n_orb = None
        self.n_a = None
        self.n_b = None
        self.e_nuc = None
        self.reference_ket = None
        self.occupied_list = None
        self.fermi_ham = None
        self.s2 = None
        self.n_frzn_occ = 0
        self.n_act = None

    def initialize(self,charge=0, spin=0, basis='sto-3g'):
        [self.n_orb, self.n_a, self.n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(self.geometry,
                                                                            charge,
                                                                            spin,basis, 
                                                                            n_frzn_occ=self.n_frzn_occ,
                                                                            n_act = self.n_act,
                                                                            unit=self.mol_unit) 
        # Muqing: unit parameter is added by myself

        print(" n_orb: %4i" %self.n_orb, flush=True)
        print(" n_a  : %4i" %self.n_a, flush=True)
        print(" n_b  : %4i" %self.n_b, flush=True)
        print(" Nuclear Repulsion Energy: %12.8f" %(E_nuc), flush=True)

        self.e_nuc = E_nuc

        sq_ham = pyscf_helper.SQ_Hamiltonian()
        sq_ham.init(h, g, C, S)
        print(" HF Energy: %12.8f" %(E_nuc + sq_ham.energy_of_determinant(range(self.n_a),range(self.n_b))))
        ehf = E_nuc + sq_ham.energy_of_determinant(range(self.n_a),range(self.n_b))
        fermi_ham  = sq_ham.export_FermionOperator()

        hamiltonian = of.linalg.get_sparse_operator(fermi_ham)

        self.s2 = vqe_methods.Make_S2(self.n_orb)

        #build reference configuration
        self.occupied_list = []
        for i in range(self.n_a):
            self.occupied_list.append(i*2)
        for i in range(self.n_b):
            self.occupied_list.append(i*2+1)

        print(" Build reference state with %4i alpha and %4i beta electrons" %(self.n_a,self.n_b), self.occupied_list)
        self.reference_ket = scipy.sparse.csc_matrix(of.jw_configuration_state(self.occupied_list, 2*self.n_orb)).transpose()

        [e,v] = scipy.sparse.linalg.eigsh(hamiltonian.real,1,which='SA',v0=self.reference_ket.todense())
        for ei in range(len(e)):
            S2 = v[:,ei].conj().T.dot(self.s2.dot(v[:,ei]))
            print(" State %4i: %12.8f au  <S2>: %12.8f" %(ei,e[ei]+E_nuc,S2))

        fermi_ham += FermionOperator((),E_nuc)
        self.fermi_ham = fermi_ham
        molden.from_mo(mol, "full.molden", sq_ham.C)

class H4S(VQEMole):
    def __init__(self):
        alpha = 0.005
        x_len = 2*np.sin(alpha*np.pi) # Muqing
        y_len = 2*np.cos(alpha*np.pi) # Muqing
        self.geometry = [('H', (0,0,0)), 
                    ('H', (x_len,y_len,0)), 
                    ('H', (2+x_len,y_len,0)), 
                    ('H', (2+2*x_len,0,0))] # Muqing: H4S
        self.mol_unit = 'B' # (B, b, Bohr, bohr, AU, au), (A, a, Angstrom, angstrom, Ang, ang)
        self.mol_name = 'H4S'
        self.mol_name_pic = 'H4 α=0.005'
        self.n_frzn_occ = 0
        self.n_act = None


class H4L(VQEMole):
    def __init__(self):
        # H4 linear     Running time: 1 or 2 minutes
        self.geometry = [('H', (0,0,0)), 
                    ('H', (2,0,0)), 
                    ('H', (4,0,0)), 
                    ('H', (6,0,0))] # Muqing: H4L
        self.mol_unit = 'B'
        self.mol_name = 'H4L'
        self.mol_name_pic = 'H4 α=0.500'
        self.n_frzn_occ = 0
        self.n_act = None


class H4L5(VQEMole):
    def __init__(self):
        # H4 H-H=5.0 Ang
        self.geometry = [('H', (0,0,0)), 
                    ('H', (5,0,0)), 
                    ('H', (10,0,0)), 
                    ('H', (15,0,0))] # Muqing: H4L
        self.mol_unit = 'Angstrom'
        self.mol_name = 'H4L5'
        self.mol_name_pic = 'H4 H-H=5.0 Ang'
        self.n_frzn_occ = 0
        self.n_act = None

class LiH(VQEMole):
    def __init__(self):
        ## LiH 1.62A     Running time: ~8 minutes
        self.geometry = [('Li', (0,0,0)), 
                    ('H',  (0,0,1.62))] # Muqing
        self.mol_unit = 'Angstrom' #  (B, b, Bohr, bohr, AU, au), (A, a, Angstrom, angstrom, Ang, ang)
        self.mol_name = 'LiH'
        self.mol_name_pic = 'LiH'
        self.n_frzn_occ = 0
        self.n_act = None


class BeH2(VQEMole):
    def __init__(self):
        ## BeH2 1.33A     Running time: ~1 hours
        self.geometry = [('H',  (0.0,0.0,-1.33)), 
                    ('Be', (0.0,0.0,0.0)), 
                    ('H',  (0.0,0.0,1.33))] # Muqing
        self.mol_unit = 'Angstrom' # (B, b, Bohr, bohr, AU, au), (A, a, Angstrom, angstrom, Ang, ang)
        self.mol_name = 'BeH2'
        self.mol_name_pic = 'BeH2'
        self.n_frzn_occ = 0
        self.n_act = None

class H635(VQEMole):
    def __init__(self):
        # H6 3.5     Running time: 
        self.geometry = [('H', (0,0,0)), 
                    ('H', (3.5,0,0)), 
                    ('H', (7.0,0,0)), 
                    ('H', (10.5,0,0)),
                    ('H', (14.0,0,0)),
                    ('H', (17.5,0,0))] # Muqing: H6 3.5
        self.mol_unit = 'B'
        self.mol_name = 'H635'
        self.mol_name_pic = 'H6 3.5Bohr'
        self.n_frzn_occ = 0
        self.n_act = None

class H620(VQEMole):
    def __init__(self):
        # # H6 2.0     Running time: 2 hours
        self.geometry = [('H', (0,0,0)), 
                    ('H', (2.0,0,0)), 
                    ('H', (4.0,0,0)), 
                    ('H', (6.0,0,0)),
                    ('H', (8.0,0,0)),
                    ('H', (10.0,0,0))] # Muqing: H6 2.0
        self.mol_unit = 'B'
        self.mol_name = 'H620'
        self.mol_name_pic = 'H6 2.0Bohr'
        self.n_frzn_occ = 0
        self.n_act = None      
        
class H65A(VQEMole):
    def __init__(self):
        # # H6 5.0Ang     Running time: 4 hours 20 minutes
        self.geometry = [('H', (0,0,0)), 
                    ('H', (5.0,0,0)), 
                    ('H', (10.0,0,0)), 
                    ('H', (15.0,0,0)),
                    ('H', (20.0,0,0)),
                    ('H', (25.0,0,0))] # Muqing: H6 5.0 Ang
        self.mol_unit = 'Angstrom'
        self.mol_name = 'H65A'
        self.mol_name_pic = 'H6 5.0Ang'
        self.n_frzn_occ = 0
        self.n_act = None


class H2O(VQEMole):
    def __init__(self, ROH = 0.9894):
        ## https://cccbdb.nist.gov/geom3x.asp?method=1&basis=20
        angle = (104.4776/180 * np.pi)*0.5
        zcoord = 0.1173 - ROH*np.cos(angle)
        ycoord = ROH*np.sin(angle)
        self.geometry = [('O', (0,0,0.1173)), 
                    ('H', (0, ycoord,zcoord)), 
                    ('H', (0,-ycoord,zcoord))] # symmetric stretch of H2O
        self.mol_unit = 'Angstrom'
        self.mol_name = 'H2O'
        self.mol_name_pic = f'H2O {ROH}Ang'
        self.n_frzn_occ = 0
        self.n_act = None


class H2O20(VQEMole):
    def __init__(self):
        ## https://cccbdb.nist.gov/geom3x.asp?method=1&basis=20
        ROH = 2.0
        angle = (104.4776/180 * np.pi)*0.5
        zcoord = 0.1173 - ROH*np.cos(angle)
        ycoord = ROH*np.sin(angle)
        self.geometry = [('O', (0,0,0.1173)), 
                    ('H', (0, ycoord,zcoord)), 
                    ('H', (0,-ycoord,zcoord))] # symmetric stretch of H2O
        self.mol_unit = 'Angstrom'
        self.mol_name = 'H2O20'
        self.mol_name_pic = 'H2O 2.0Ang'
        self.n_frzn_occ = 0
        self.n_act = None

class H2O40(VQEMole):
    def __init__(self):
        ## https://cccbdb.nist.gov/geom3x.asp?method=1&basis=20
        ROH = 4.0
        angle = (104.4776/180 * np.pi)*0.5
        zcoord = 0.1173 - ROH*np.cos(angle)
        ycoord = ROH*np.sin(angle)
        self.geometry = [('O', (0,0,0.1173)), 
                    ('H', (0, ycoord,zcoord)), 
                    ('H', (0,-ycoord,zcoord))] # symmetric stretch of H2O
        self.mol_unit = 'Angstrom'
        self.mol_name = 'H2O40'
        self.mol_name_pic = 'H2O 4.0Ang'
        self.n_frzn_occ = 0
        self.n_act = None


class N235F4(VQEMole):
    def __init__(self):
        ## N2
        RNN = 3.5
        self.geometry = [('N',  (0.0,0.0,0.0)), 
                    ('N',  (0.0,0.0,RNN))] # Muqing
        self.mol_unit = 'Angstrom' # (B, b, Bohr, bohr, AU, au), (A, a, Angstrom, angstrom, Ang, ang)
        self.n_frzn_occ = 4
        self.n_act = 6
        self.mol_name = 'N235F4'
        self.mol_name_pic = 'N2 3.5 F4'




# H4 linear (Unit: Bohr)
# ('H', (0, 0, 0)), 
# ('H', (2, 0, 0)), 
# ('H', (4, 0, 0)), 
# ('H', (6, 0, 0))

# H4 square (Unit: Bohr)
# ('H', (0, 0, 0)), 
# ('H', (0.03141463462364135, 1.9997532649633212, 0)), 
# ('H', (2.0314146346236415, 1.9997532649633212, 0)), 
# ('H', (2.0628292692472825, 0, 0))

# LiH (Unit: Angstrom)
# ('Li', (0, 0, 0)), 
# ('H', (0, 0, 1.62))

# BeH2 (Unit: Angstrom)
# ('H', (0.0, 0.0, -1.33)), 
# ('Be', (0.0, 0.0, 0.0)), 
# ('H', (0.0, 0.0, 1.33))

# H6 2.0  (Unit: Bohr)
# ('H', (0, 0, 0)), 
# ('H', (2.0, 0, 0)), 
# ('H', (4.0, 0, 0)), 
# ('H', (6.0, 0, 0)), 
# ('H', (8.0, 0, 0)), 
# ('H', (10.0, 0, 0))

# H6 3.5  (Unit: Bohr)
# ('H', (0, 0, 0)), 
# ('H', (3.5, 0, 0)), 
# ('H', (7.0, 0, 0)), 
# ('H', (10.5, 0, 0)), 
# ('H', (14.0, 0, 0)), 
# ('H', (17.5, 0, 0))
