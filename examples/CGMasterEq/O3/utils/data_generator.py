import pandas as pd
import numpy as np
import re

from sklearn.utils import shuffle
from scipy.spatial import KDTree
from pyDOE import lhs

from const      import const
from ode_solver import ODESolver



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
        self.t0, self.t_end = inputs.t0, inputs.t_end               # Time limits for testing

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
            y_del = np.abs((y_i[:-1,:] - y_i[1:,:]) / y_i[1:,:])
            y_del = y_del[:,np.argmax(ic_i[:-1])]
            idx_min = np.where(y_del > self.rtol)[0][0]         # Dynamics start-point
            idx_max = np.where(y_del > self.rtol*5)[0][-1]   # Dynamics end-point
            t_min, t_max= t_i[idx_min], t_i[idx_max]
            if self.train_pts_dist == 'random':
                d, ii  = tree.query([[t_min],[t_max]], k=1)
                idx    = np.sort(np.random.choice(ii[1]-ii[0], (self.train_pts-1), replace=False)) + ii[0]
            else:
                t_vec  = np.linspace(np.log10(t_min), np.log10(t_max), \
                                     self.train_pts-1, dtype=np.float64)
                h      = (t_vec[1,0]-t_vec[0,0])*0.8
                delta  = (np.random.rand(self.train_pts-1,1)-0.5)*h
                t_unif = np.power(10, t_vec + delta)
                d, idx = tree.query(t_unif, k=1)

            idx_train, idx_valid = self.stratified_sampling(idx, self.valid_pts)
            idx_train = np.insert(idx_train, 0, 0)      # Add initial condition
            assert self.train_pts == np.shape(idx_train)[0] + np.shape(idx_valid)[0]

            # Extract data values
            data_train_i, data_valid_i = get_data(idx_train), get_data(idx_valid)
            if i == 0:
                data_train, data_valid = data_train_i, data_valid_i
            else:
                data_train = list(map(lambda x,y: np.vstack((x,y)), data_train, data_train_i))
                data_valid = list(map(lambda x,y: np.vstack((x,y)), data_valid, data_valid_i))

            # Keep data for underfitting test
            if self.test_training:
                t_tr = data_train_i[2]
                if i == 0:
                    self.test_training_data = [(ic_i, t_tr, (data_valid_i))]
                else:
                    self.test_training_data.append((ic_i, t_tr, (data_valid_i)))
        if self.test_training: assert len(self.test_training_data) == n

        data_train = shuffle(*data_train)
        data_valid = shuffle(*data_valid)

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

            # Solving the ODE
            T_i = str(int(X[i,0]))
            t_i, y_i = self.ODE_sol.solve(ic[i,:], self.ODE_mat[T_i])

            # Extract data values
            tree     = KDTree(np.c_[t_i.ravel()])
            d, ii    = tree.query([[self.t0],[self.t_end]], k=1)
            t_i, y_i = t_i[ii[0]:ii[1],:], y_i[ii[0]:ii[1],:]
            u_i      = np.tile(ic[i,:], (t_i.shape[0],1))

            title = 'Tested case (T = '+T_i+' K)'
            if i == 0:
                data_test = [(title, [u_i, t_i], y_i)]
            else:
                data_test.append((title, [u_i, t_i], y_i))

        data = data_test

        ### Test underfitting --------------------------------------------------
        if self.test_training:
            assert self.train_cases > n

            idx = np.random.choice(self.train_cases, size=n, replace=False)
            for i, idx_i in enumerate(idx):
                ic_i, t_tr, data_valid_i   = self.test_training_data[idx_i]
                T_val, u_val, t_val, y_val = data_valid_i

                # Solving the ODE
                T_i = str(int(T_val[0,0]))
                t_i, y_i = self.ODE_sol.solve(ic_i, self.ODE_mat[T_i])

                # Extract data values
                tree     = KDTree(np.c_[t_i.ravel()])
                t_min    = np.min(t_tr[np.nonzero(t_tr)])
                t_max    = np.max(t_tr[np.nonzero(t_tr)])
                d, ii    = tree.query([[t_min],[t_max]], k=1)
                t_i, y_i = t_i[ii[0]:ii[1],:], y_i[ii[0]:ii[1],:]
                u_i      = np.tile(ic_i, (t_i.shape[0],1))
                assert t_min != 0.

                title = 'Trained case (T = '+T_i+' K)'
                if i == 0:
                    data_test_training = [(title, [u_i, t_i], y_i, (t_val, y_val))]
                else:
                    data_test_training.append((title, [u_i, t_i], y_i, (t_val, y_val)))

            data = data + data_test_training

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
        x_train = np.setdiff1d(x, x_test)
        assert np.intersect1d(x_test, x_train).size == 0
        return x_train, x_test
