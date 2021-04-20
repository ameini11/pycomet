import pandas as pd
import numpy as np
import re

from scipy.spatial import KDTree
from pyDOE import lhs

from const      import const
from ode_solver import ODESolver




import matplotlib.pyplot as plt




class DataGenerator(object):

    ###########################################################################
    ### Initialization
    ###########################################################################
    def __init__(self, inputs):

        # Data Quantities -----------------------------------------------------
        self.train_cases    = inputs.train_cases                    # Number of training cases
        self.train_pts      = inputs.train_pts                      # Number of total training pts for each case
        self.valid_pts      = round(inputs.valid*inputs.train_pts)  # Number of validation v for each case
        self.train_pts_dist = inputs.train_pts_dist                 # Training pts distribution

        self.test_cases     = inputs.test_cases                     # Number of testing cases
        self.test_training  = inputs.test_training                  # Test training case [Bool]

        self.dim            = inputs.G_dim                          # Data dimensionality

        self.rtol           = 1.e-5                                 # Rel.Tol for cutting equilibrium

        # Input Space Variables -----------------------------------------------
        inp_var = [inputs.Ttr, inputs.T0, inputs.P0, inputs.XO2]
        self.Ttr, self.T0, self.P0, self.XO2 = list(map(lambda x: np.array(x, dtype=np.float64), inp_var))
        self.T_str = list(map(lambda x: str(int(x)), self.Ttr))     # Temp. in string form

        # Retrieving Molecular data -------------------------------------------
        self.levels_file = inputs.database_path + '/levels/O3_UMN/O2/FromUMN_Sorted.inp'
        self.binMap_file = inputs.database_path + '/grouping/O3_UMN/O2/LevelsMap_' + inputs.grouping + '.csv'
        self.g, self.EeV, self.num_lev = self.read_levels()
        self.bins, self.num_bins = self.read_groups()

        # Retrieving Rates coefficients ---------------------------------------
        self.rates_path = inputs.database_path + '/kinetics/O3_UMN_' + inputs.grouping + '/'
        self.react_file = {'diss': 'Diss_Corrected.dat', 'exch': 'Exch_Type1.dat', 'inel': 'Inel.dat'}

        # Building Master Equation matrices -----------------------------------
        mat_name     = list(inputs.MEq_mat.keys())
        mat_mask     = list(inputs.MEq_mat.values())
        self.ODE_sol = ODESolver(mat_name, mat_mask)
        self.ODE_mat = self.build_matrices()


    ###########################################################################
    ### Master Equation matrices
    ###########################################################################
    def get_rates(self):
        '''Retrieve rates coefficients from file.'''

        # ---------------------------------------------------------------------
        def read_rates(file, reac_name):
            '''Read rates coefficients from file.'''

            shape = (1,self.num_bins) if reac_name == 'diss' else (self.num_bins,self.num_bins)
            rates = np.zeros(shape, dtype=np.float64)

            infile = open(file, 'r')
            for line in infile:
                reac_eq, rate_str = line.split(':')

                rate = float(rate_str.split(',')[0]) * 1.e-6
                bins_idx = re.findall(r"\((.*?)\)",reac_eq)

                if reac_name == 'diss':
                    rates[0,int(bins_idx[0])-1] = rate
                else:
                    rates[int(bins_idx[0])-1,int(bins_idx[1])-1] = rate
            infile.close()
            return rates
        # ---------------------------------------------------------------------

        rates_coeff = dict()
        for T_i in self.T_str:
            rates_coeff[T_i] = dict()
            for reac in self.react_file:
                data_file = self.rates_path + 'T' + T_i + 'K/' + self.react_file[reac]
                rates_coeff[T_i][reac] = read_rates(data_file, reac)
        return rates_coeff


    def build_matrices(self):
        '''Build Master Equation matrices.'''

        mat = dict()
        rates_coeff = self.get_rates()
        for i, T_i in enumerate(self.T_str):
            Q_bins_T = self.compute_partition_func(self.Ttr[i])
            mat[T_i] = self.ODE_sol.get_matrices(rates_coeff[T_i], Q_bins_T, self.Ttr[i])
        return mat


    ###########################################################################
    ### Training/Validation Data
    ###########################################################################
    def training_data(self):

        n = self.train_cases

        # Contructing Design Matrix -------------------------------------------
        X       = lhs(3, samples=n)
        X       = np.hstack((np.random.choice(self.Ttr, size=(n,1)), X))
        X[:,1:] = np.hstack(list(map(lambda i,x: X[:,i:i+1]*(x[1] - x[0]) + x[0], \
                                     range(1,4), [self.T0, self.P0, self.XO2])))

        # Generating Initial Conditions ---------------------------------------
        ic = np.zeros((n,self.dim), dtype=np.float64)
        for i in range(n):

            Ttr, T0, P0, XO2 = [ X[i,j] for j in range(4) ]
            XO  = 1. - XO2
            nd  = P0 / (const.UKb * T0)

            Q_bins = self.compute_partition_func(T0)
            ic[i,:-1] = XO2 * Q_bins / sum(Q_bins) * 2.0 * const.mO * nd    # rho_O2
            ic[i, -1] = XO  * const.mO * nd                                 # rho_O

        self.test_training_data = [ ic ]      # Keep them for underfitting testing

        # Generating Actual Data ----------------------------------------------
        # ---------------------------------------------------------------------
        def get_data(idx):
            '''Extract data from arrays.'''

            m = np.shape(idx)[0]
            T = np.tile(T_i, (m,1))     # NN Input: Translational Temperature
            u = np.tile(ic_i, (m,1))    # NN Input: Initial Condition
            t = t_i[idx,:]              # NN Input: Time
            y = y_i[idx[:],:]           # NN Target
            return T, u, t, y
        # ---------------------------------------------------------------------

        assert self.train_pts > 3
        for i, ic_i in enumerate(ic):
            # Solving the ODE
            T_i = X[i,0]
            t_i, y_i = self.ODE_sol.solve(ic_i, self.ODE_mat[str(int(T_i))])

            # Select meaningful values by sampling t vector
            tree  = KDTree(np.c_[t_i.ravel()])
            y_del = np.abs((y_i[:-1,:] - y_i[1:,:]) / y_i[1:,:])[:,np.argmax(ic_i)]
            idx_min = np.where(y_del > self.rtol)[0][0]         # Dynamics start-point
            idx_max = np.where(y_del > self.rtol*1.e1)[0][-1]   # Dynamics end-point
            t_min, t_max= t_i[idx_min], t_i[idx_max]
            if self.train_pts_dist == 'random':
                d, ii  = tree.query([[t_min],[t_max]], k=1)
                idx    = np.sort(np.random.choice(ii[1]-ii[0], (self.train_pts-1), replace=False)) + ii[0]
            else:
                t_vec  = np.linspace(np.log10(t_min), np.log10(t_max), \
                                     self.train_pts-1, dtype=np.float64)
                h      = (np.random.rand(self.train_pts-1,1)-0.5)*(t_vec[1,0]-t_vec[0,0])
                t_unif = np.power(10, t_vec + h)
                d, idx = tree.query(t_unif, k=1)

            idx_train, idx_test = self.stratified_sampling(idx, self.valid_pts)
            idx_train = np.insert(idx_train, 0, 0)      # Add initial condition
            assert self.train_pts == np.shape(idx_train)[0] + np.shape(idx_test)[0]

            # Extract data values
            data_train_i, data_valid_i = get_data(idx_train), get_data(idx_test)
            if i == 0:
                data_train, data_valid = data_train_i, data_valid_i
            else:
                data_train = list(map(lambda x,y: np.vstack((x,y)), data_train, data_train_i))
                data_valid = list(map(lambda x,y: np.vstack((x,y)), data_valid, data_valid_i))

            self.test_training_data.append((data_valid_i))  # Keep them for underfitting testing
        assert len(self.test_training_data) == n + 1

        return data_train, data_valid


    ###########################################################################
    ### Testing Data
    ###########################################################################
    def testing_data(self):

        n = self.test_cases

        ### Test overfitting --------------------------------------------------
        # Contructing Design Matrix -------------------------------------------
        X       = lhs(3, samples=n)
        X       = np.hstack((np.random.choice(self.Ttr, size=(n,1)), X))
        X[:,1:] = np.hstack(list(map(lambda i,x: X[:,i]*(x[1] - x[0]) + x[0], \
                                     range(1,4), [self.T0, self.P0, self.XO2])))

        # Generating Initial Conditions ---------------------------------------
        ic = np.zeros((n,self.dim), dtype=np.float64)
        for i in range(n):

            Ttr, T0, P0, XO2 = [ X[i,j] for j in range(4) ]
            XO  = 1. - XO2
            nd  = P0 / (const.UKb * T0)

            Q_bins = self.compute_partition_func(T0)
            ic[i,:-1] = XO2 * Q_bins / sum(Q_bins) * 2.0 * const.mO * nd    # rho_O2
            ic[i, -1] = XO  * const.mO * nd                                 # rho_O

            # Solving the ODE
            T_i = str(int(X[i,0]))
            t_i, y_i = self.ODE_sol.solve(ic_i, self.ODE_mat[T_i])

            # Extract data values
            if i == 0:
                data_test = (T_i, (t_i, y_i))
            else:
                data_test = data_test + (T_i, (t_i, y_i))

        ### Test underfitting --------------------------------------------------
        if self.test_training:
            assert self.train_cases > self.test_cases
            ic_train, *data_valid = self.test_training_data
            idx = np.random.choice(ic_train.shape[0], size=n, replace=False)
            ic_train, Ttr_train = ic_train[idx, :], Ttr_train[idx, :]

            for i, ic_i in enumerate(ic_train):
                # Solving the ODE
                T_i = str(int(Ttr_train[i,0]))
                t_i, y_i = self.ODE_sol.solve(ic_i, self.ODE_mat[T_i])

                # Extract data values
                if i == 0:
                    data_test_training = (T_i, (t_i, y_i))
                else:
                    data_test_training = data_test_training + (T_i, (t_i, y_i))
            data = {'Trained': data_test_training, 'Tested': data_test}
        else:
            data = data_test

        return data


    ###########################################################################
    ### Util Functions
    ###########################################################################
    def compute_partition_func(self, T):
        '''Compute bins partition function.'''

        q = self.g * np.exp( - self.EeV * const.Ue / (T * const.UKb) )
        Q_bins = np.zeros(self.num_bins)
        for i_lev in range(self.num_lev):
            Q_bins[self.bins[i_lev]] = Q_bins[self.bins[i_lev]] + q[i_lev]
        return Q_bins

    def read_levels(self):
        '''Retriev all the rot-vib levels'''

        data = pd.read_csv(self.levels_file, header=None, skiprows=15, delim_whitespace=True)
        data = data.apply(pd.to_numeric, errors='coerce')
        vqn  = np.array(data[0])                    # Vibrational Q.N.
        jqn  = np.array(data[1])                    # Rotational  Q.N.
        g    = 1.5*(2.0*jqn+1.0)                    # Degeneracy (Including Electronic One)
        EeV  = np.array(data[2]) * const.Eh_to_eV   # Energy in eV
        EeV  = EeV - EeV[0]
        num  = len(vqn)
        return g, EeV, num

    def read_groups(self):
        '''Retrieve mapping of each rot-vib level to the respective bin'''

        data = pd.read_csv(self.binMap_file, header=None, skiprows=1)
        data = data.apply(pd.to_numeric, errors='coerce')
        idx  = np.array(data[0])                                                                           
        bins = np.array(data[1]) - 1    # Level-to-Bin
        num  = max(bins) + 1
        return bins, num

    def stratified_sampling(self, x, strata):
        '''Stratified sampling for splitting Training/Validation data.'''

        x_split = np.array_split(x, strata)
        x_test  = np.array([ np.random.choice(i) for i in x_split ])
        x_train = np.array([ i for i in x if i not in x_test ])
        assert np.intersect1d(x_test, x_train).size == 0
        return x_train, x_test
