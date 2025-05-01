##--------------
## python qflow_adgcm_main.py > Output_Data/ADAPT-GCM/GCM_H620_tp4/printout.txt
## python qflow_adgcm_main.py > Output_Data/ADAPT-GCM/GCM_H620_tp4_Com/printout.txt

## python qflow_adgcm_main.py > Output_Data/ADAPT-GCM/GCM_H620O6DUCC3_tp4/printout.txt

## python qflow_adgcm_main.py > Output_Data/ADAPT-GCM-OPT/GCMOPT_H620_tp4_5_2/printout.txt
## python qflow_adgcm_main.py > Output_Data/ADAPT-GCM-OPT/GCMOPT_H620O6DUCC3_tp4_Com_3_3/printout.txt
## python qflow_adgcm_main.py > Output_Data/ADAPT-GCM/GCM_H620O6DUCC3_tp4_Com/printout.txt

## python qflow_adgcm_main.py > Output_Data/ADAPT-GCM/GCM_H630O6DUCC3_tp4/printout.txt
## python qflow_adgcm_main.py > Output_Data/ADAPT-GCM-OPT/GCMOPT_H630O6DUCC3_tp4_5_2/printout.txt
## python qflow_adgcm_main.py > Output_Data/ADAPT-GCM-OPT/GCMOPT_H630O6DUCC3_tp4_Com_5_2/printout.txt
## python qflow_adgcm_main.py > Output_Data/ADAPT-GCM-OPT/GCMOPT_H630O6DUCC3_tp4_Com_3_3/printout.txt
## python qflow_adgcm_main.py > Output_Data/ADAPT-GCM-OPT/GCMOPT_H630O6DUCC3_tp4_P_5_2/printout.txt


## python qflow_adgcm_main.py > Output_Data/ADAPT-GCM-OPT/GCMOPT_H630O7DUCC3_tp4_5_2/printout.txt


from moles import *
## ADAPT-GCM
from adapt_vqe_gcm import *
from _gcim_utilis import *
from timeit import default_timer as timer


tstart_total = timer()

#########
# mol_name = 'H620'
# file_path = "Input_Data/H6/2au/6-Orbitals/Bare/H6_2au_Bare_6-electrons_6-Orbitals.out-xacc"

# mol_name =  'H620O6DUCC3'
# file_path = "Input_Data/H6/2au/6-Orbitals/DUCC3/H6_2au_DUCC3_6-electrons_6-Orbitals.out-xacc"

# mol_name =  'H630O6DUCC3'
# file_path = "Input_Data/H6/3au/6-Orbitals/DUCC3/H6_3au_DUCC3_6-electrons_6-Orbitals.out-xacc"

# mol_name =  'H630O7DUCC3'
# file_path = "Input_Data/H6/3au/7-Orbitals/DUCC3/H6_3au_DUCC3_6-electrons_7-Orbitals.out-xacc"

# mol_name =  'H630O8DUCC3'
# file_path = "Input_Data/H6/3au/8-Orbitals/DUCC3/H6_3au_DUCC3_6-electrons_8-Orbitals.out-xacc"

mol_name = "H4_09"
file_path = "../example_hamiltonians/H4_4_0.9_xacc.hamil"



# n_orb = 6
# n_orb = 7
n_orb = 4
n_a = 2
n_b = 2
##
adapt_opt_intvl,opt_depth = 0,0
# adapt_opt_intvl,opt_depth = 5,2
# adapt_opt_intvl,opt_depth = 2,4
###########

fermi_ham = parse_hamiltonian(file_path, n_orb, use_interleaved=True, return_type = 'of')
spin_complete_pool = False
pauli_pool = False
make_orth = False
theta = np.pi/4
use_gcmgrad = False
#
occupied_list = []
for i in range(n_a):
    occupied_list.append(i*2)
for i in range(n_b):
    occupied_list.append(i*2+1)
print(" Build reference state with %4i alpha and %4i beta electrons" %(n_a,n_b), occupied_list)
reference_ket = scipy.sparse.csc_matrix(of.jw_configuration_state(occupied_list, 2*n_orb)).transpose()

print("  Hamiltonian shape", of.linalg.get_sparse_operator(fermi_ham).shape)
print("  Reference state shape", reference_ket.shape)



file_prefix = 'Output_Data/GCM_{:s}/'.format(mol_name)
if adapt_opt_intvl > 0:
    file_prefix = 'Output_Data/GCMOPT_{:s}_{:d}_{:d}/'.format(mol_name,adapt_opt_intvl, opt_depth)
    
import os
if not os.path.exists(file_prefix):
    os.makedirs(file_prefix)
# else:
#     raise Warning("Folder already exists, please change the file_prefix")

# file_prefix = 'Output_Data/ADAPT-GCM/GCM_Test/'

print(" Use Spin-complete Pool: ", spin_complete_pool)
print(" Saved in folder: ", file_prefix)

#   Francesco, change this to singlet_GSD() if you want generalized singles and doubles
if spin_complete_pool:
    if pauli_pool: 
        pool = None
    else:
        pool = operator_pools.spin_complement_GSD()
else:
    if pauli_pool: 
        print(" Choose PAULI EXCITATION POOL 100%")
        pool = operator_pools.pauli_exc()
        # print(" Choose PAULI EXCITATION POOL reduced 1/4")
        # pool = operator_pools.pauli_exc4()
    else:
        pool = operator_pools.spin_complement_GSD()
pool.init(n_orb, n_occ_a=n_a, n_occ_b=n_b, n_vir_a=n_orb-n_a, n_vir_b=n_orb-n_b)
sys.stdout.flush()
print("Finish Pool Construction.\n\n")

#----------------------- Modification ------------------#
"""
A few parameters added by Muqing:
file_prefix (str, optional): file path for saving GCM matrices in each iteration END with '/'. 
                            None for not saving matrices. 
                            If path does not exists, it does NOT create the path. 
                            Defaults to None. 
no_printout_level (int, optional): 0 means print out everything, 
                                    1 means no VQE parameter optimization info, grad print thresh*100, 
                                    2 means ALSO no ansatz info in each VQE iteration, no grad info. 
                                    Defaults to 0.
make_orth (bool, optional): If make basis orthgonal when construct matrices for GCM. 
                            Defaults to False. 
theta_thresh is for VQE convergence, origianlly in adapt_vqe()
"""
# file_prefix = None
# [e,v,params,GCM_DIFFS,VQE_DIFFS,GCM_BASIS_SIZE] = adapt_vqe_gcm(fermi_ham, pool, reference_ket,
#                                                                         file_prefix=file_prefix,
#                                                                         theta_thresh=1e-9,
#                                                                         no_printout_level=2, 
#                                                                         make_orth=make_orth)
# temp_ham = of.linalg.get_sparse_operator(fermi_ham)
# true_lowest_ev = scipy.sparse.linalg.eigsh(temp_ham,1,which='SA')[0][0].real
# np.floor(true_lowest_ev)-1
# print(f"!!! Convergence criteria: GCM errors do not change(<1e-{10}) for {20} iterations!!!")
if adapt_opt_intvl == 0:
    VQE_ITERS = None
    [GCM_EVS,GCM_Indices, GCM_DIFFS, GCM_BASIS_SIZE] = adapt_gcm(fermi_ham, pool, reference_ket,
                                    theta = theta,
                                    use_gcmgrad  = use_gcmgrad,
                                    use_qugcm    = False,
                                    file_prefix  = file_prefix,
                                    stop_type    = 'long_flat',
                                    stop_thresh  = (10,1e-6),
                                    adapt_maxiter= 200,
                                    no_printout_level=2, 
                                    make_orth = make_orth,
                                    ev_thresh = 1e-12)
else:
    [GCM_EVS,GCM_Indices, GCM_DIFFS, GCM_BASIS_SIZE, VQE_ITERS] = adapt_gcm_opt(fermi_ham, pool, reference_ket,
                                    theta = theta,
                                    adapt_opt_intvl = adapt_opt_intvl,
                                    opt_depth = opt_depth,
                                    use_gcmgrad  = use_gcmgrad,
                                    use_qugcm    = False,
                                    file_prefix  = file_prefix,
                                    stop_type    = 'long_flat',
                                    stop_thresh  = (10,1e-6),
                                    adapt_maxiter= 200,
                                    no_printout_level=2, 
                                    make_orth = make_orth,
                                    ev_thresh = 1e-12)
tend_total = timer()
print("\n\n\n\n\n>>> Total GCIM time (sec): ", tend_total-tstart_total)
#----------------------- Modification ------------------#
print("Selected indices:", GCM_Indices)
print("Eigenvalues:",GCM_EVS)

## H6_2au_DUCC3_6-electrons_6-Orbitals
## - Spin adapted: (5,2) (10,1e-6)
## - Complete: (3,3) (25,1e-6)

## H6_3au_DUCC3_6-electrons_6-Orbitals
## - Spin adapted: (5,2) (10,1e-6)
## - Complete: (5,2) (10,1e-6)


np.save(file_prefix+'GCM_DIFF.npy', np.array(GCM_DIFFS))
np.save(file_prefix+'GCM_SIZE.npy', np.array(GCM_BASIS_SIZE))
np.save(file_prefix+'GCM_EVS.npy', np.array(GCM_EVS))
np.save(file_prefix+'GCM_Indices.npy', np.array(GCM_Indices))
if VQE_ITERS:
    np.save(file_prefix+'VQE_NITERS.npy', np.array(VQE_ITERS))



## First good iteration
print("\nVerfiy eigenvector correctness")
mole_dict = {'mole_name': mol_name,'iter_num': len(GCM_Indices)-1,'nele': n_a+n_b,'purt': 1e-12}
_,_ = gcm_res_validation(mole_dict , file_prefix)


index_start = 0
index_end = len(GCM_DIFFS) # not included
iter_range = list(range( index_start, index_end ))
# plt.plot(iter_range, np.array(VQE_DIFFS)[iter_range], linewidth=4, label='ADAPT VQE')
plt.plot(iter_range, np.array(GCM_DIFFS)[iter_range], color = 'b', linewidth=4, label='ADAPT GCM')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.yscale('log')
plt.title('Error from ground-state FCI energy')
plt.legend()
plt.tight_layout()
plt.savefig(file_prefix+mol_name+'.png', dpi=300,facecolor='white', edgecolor='white')
plt.clf()
