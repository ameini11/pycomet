import numpy as np
import time
import os
import re
import json
import shutil
from scipy.integrate import ode
from scipy.spatial import KDTree
from pyDOE import lhs
from sklearn.utils import shuffle
import sys
from numpy import double
import pandas as pd 
import random 


from .const      import const
from .ode_solver import ODESolver


class data_generator(object):

    ###########################################################################
    ### Initialization
    ###########################################################################
    def __init__(self, inputs):

        # Input Space Variables -----------------------------------------------
        self.train_cases = train_cases  # Number of training cases
        self.train_pts  = round((1.-valid)*train_pts)       # Number of training points for each case
        self.residual = residual      # Residual evaluation
        self.valid_pts  = train_pts-self.train_pts       # Number of training points for each case

        self.test_cases = test_cases # Number of testing cases

    
        self.num_test_cases = test_cases # Number of testing cases=

        self.dim = dim      # Data dimensionality


        # Input Space Variables -----------------------------------------------
        inp_var = [inputs.Ttr, inputs.T0, inputs.P0, inputs.XO2]
        self.Ttr, self.T0, self.P0, self.X_O2 = list(map(lambda x: np.array(x, dtype=np.float64), inp_var))
        self.T_str = list(map(lambda x: str(int(x)), self.Ttr))     # Temp. in string form
        self.t0, self.t_end  = inputs.t0, inputs.t_end              # Time limits

        # Retrieving Molecular data -------------------------------------------
        self.levels_file = inputs.database_path + '/levels/O3_UMN/O2/FromUMN_Sorted.inp'
        self.binMap_file = inputs.database_path + '/grouping/O3_UMN/O2/LevelsMap_' + inputs.grouping + '.csv'
        self.g, self.EeV, self.num_lev = self.read_levels()
        self.bins, self.num_bins = self.read_groups()

        # Retrieving Rates coefficients ---------------------------------------
        self.rates_path = inputs.database_path + 'kinetics/O3_UMN_' + inputs.grouping + '/'
        self.react_file = {'diss': 'Diss_Corrected.dat', 'exch': 'Exch_Type1.dat', 'inel': 'Inel.dat'}

        # Building Master Equation matrices -----------------------------------
        mat_name     = list(inputs.MEq_mat.keys())
        mat_mask     = list(inputs.MEq_mat.values())
        self.ODE_sol = ODESolver(self.t0, self.t_end, mat_name, mat_mask)
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
            Q_bins_T = self.compute_partition_func(self.Ttr[i]):
            mat[T_i] = self.ODE_sol.get_matrices(rates_coeff[T_i], Q_bins_T, T)
        return mat


    ###########################################################################
    ### Training/Validation Data
    ###########################################################################
    def training_data(self):

        n = self.train_cases

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

        # Generating Actual Data ----------------------------------------------
        # I.C. Points
        t = np.zeros((n,1),    dtype=np.float64)    # NN Input: Time
        T = np.array(X[:,0:1], dtype=np.float64)    # NN Input: Temperature Tr.
        u = np.array(ic,       dtype=np.float64)    # NN Input: Initial rho
        y = np.array(ic,       dtype=np.float64)    # NN Targets

        # Anchor Points
        if self.train_pts > 3:
            for i, ic_i in enumerate(ic):
                # Solving the ODE
                t_i, y_i = self.ODE_sol.solve(ic_i, self.ODE_mat[T[i,0]])
                # Select indexes
                tree = KDTree(np.c_[tTrain.ravel()])
                idx_min = np.where(np.abs(y_i[       :-1,:] - y_i[        1:,:]) \
                                        / y_i[        1:,:] > rtol)[0][ 0]              # Dynamics start-point
                idx_max = np.where(np.abs(y_i[idx_str:-1,:] - y_i[idx_str+1:,:]) \
                                        / y_i[idx_str+1:,:] > rtol)[0][-1] + idx_min    # Dynamics end-point
                # Sample t vector
                t_min = t_i[idx_str]
                t_max = t_i[idx_end]
                if t_min < self.t0 or t_max > self.t_end:
                    print('Warning! Time data exceed time limits.')
                if InputData.random_dist:
                    d, ii  = tree.query([[t_min],[t_max]], k=1)
                    idx    = np.random.choice(ii[1]-ii[0], (self.train_pts-1), replace=False) + ii[0]
                else:
                    t_vec  = np.expand_dims(np.linspace(np.log10(t_min), np.log10(t_max), \
                                            self.train_pts-1, dtype=np.float64), axis=1)
                    h      = t_vec[1,0] - t_vec[0,0]
                    t_unif = np.power(10, t_vec + np.random.rand(self.train_pts-1,1)*h)
                    d, idx = tree.query(t_unif, k=1)
                # Extracting values
                t = np.vstack((t, t_i[idx,:]))
                T = np.vstack((T, np.tile(T[i,0], (self.train_pts-1,1))))
                u = np.vstack((u, np.tile(rho_0, (NTrainPts-1,1))))
                y = np.vstack((y, y_i[idx[:],:]))
        else:
            raise IndexError('At least 4 training points needed.')

        # Randomize Data
        if (InputData.RandDataFlg):
            TData, uData, tData, yData, encod =  shuffle(TData, uData, tData, yData, encod)

        TrainData = (TData, uData, tData, yData, encod)
        # print('Training Data: TrainData = ', TrainData)
        # input('=====================================================')
        return data


    ###########################################################################
    ### Testing Data
    ###########################################################################
    def testing_data(self):

        # ================================================================ Contructing Design Matrix
        X      = lhs(3, samples=NTestCases)
        X      = np.hstack((np.random.choice(Ttr_vec, size=(NTestCases,1)), X))
        X[:,1] = X[:,1] * (T0_vec[1]   - T0_vec[0]  ) + T0_vec[0]
        X[:,2] = X[:,2] * (P0_vec[1]   - P0_vec[0]  ) + P0_vec[0]
        X[:,3] = X[:,3] * (X_O2_vec[1] - X_O2_vec[0]) + X_O2_vec[0]


        # ================================================================ Writing Initial Conditions + Solving ODE
        TestData = ()
        ICTest   = np.zeros((NTestCases,xDim))

        # Generating Initial Conditions ---------------------------------------
        ic = np.zeros((n,self.dim), dtype=np.float64)
        for i in range(n):

            Ttr, T0, P0, XO2 = [ X[i,j] for j in range(4) ]
            XO  = 1. - XO2
            nd  = P0 / (const.UKb * T0)

            Q_bins = self.compute_partition_func(T0)
            ic[i,:-1] = XO2 * Q_bins / sum(Q_bins) * 2.0 * const.mO * nd    # rho_O2
            ic[i, -1] = XO  * const.mO * nd                           # rho_O

            # Solving the ODE
            tTest, xTest = ODE_solver(ICTest[iCase,:], InputData, Ttr)
            uTest = np.tile(ICTest[iCase,:], np.shape(tTest))

            TestData = TestData + (tTest, uTest, xTest)
            # print('Testing Data: TestData = ', TestData)
            # input('=====================================================')


            if (InputData.WriteDataIntFlg >= 1):

                PathToFldr = InputData.PathToDataFld
                try:
                    os.makedirs(PathToFldr)
                except OSError as e:
                    pass
                Data       = tf.concat([tTest, xTest], axis=1)
                PathToData = PathToFldr + '/TestCase' + str(iCase+1) + '.csv.Data' 
                save_data(PathToData, Data)


        return data


    ###########################################################################
    ### Additional Functions
    ###########################################################################
    def compute_partition_func(self, T):
        '''Compute bins partition function.'''

        q     = self.g * np.exp( - self.EeV * const.Ue / (T * const.UKb) )
        Q_bins = np.zeros(self.num_bins)
        for i_lev in range(self.num_lev):
            Q_bins[self.bins[i_lev]] = Q_bins[self.bins[i_lev]] + q[i_lev]
        return Q_bins

    def read_levels(self):
        '''Retriev all the rot-vib levels'''

        data = pd.read_csv(self.levels_file, header=None, skiprows=15, delim_whitespace=True)
        data = data.apply(pd.to_numeric, errors='coerce')
        vqn  = np.array(data[0])                                    # Vibrational Q.N.
        jqn  = np.array(data[1])                                    # Rotational  Q.N.
        g    = double(1.5)*(double(2.0)*double(jqn)+double(1.0))    # Degeneracy (Including Electronic One)
        EeV  = np.array(data[2]) * const.Eh_to_eV                   # Energy in eV
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