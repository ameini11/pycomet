import re
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from sklearn.utils import shuffle
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from pyDOE import lhs

from const      import const
from ode_solver import ODESolver


class DataGenerator(object):

    ###########################################################################
    ### Initialization
    ###########################################################################
    def __init__(self, inputs):

        self.header   = '[PrODE - Data Gen]: '
        self.data_dir = "./data/"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Data Quantities -----------------------------------------------------
        self.scenarios = inputs.train_pts['ics']
        self.valid_pts = {k: round(inputs.valid*inputs.train_pts[k]) if k != 'ics' \
                                else self.scenarios for k in inputs.train_pts }
        self.train_pts = {k: inputs.train_pts[k] - self.valid_pts[k] if k != 'ics' \
                                else self.scenarios for k in inputs.train_pts }
        for k in inputs.train_pts:
            if self.valid_pts[k] == 0 and self.train_pts[k] != 0:
                self.valid_pts[k] = 1

        self.train_pts_dist = inputs.train_pts_dist         # Training pts distribution
        self.train_split    = [0.8, 0.12, 0.08]             # Training pts splitting

        self.test_cases     = inputs.test_cases             # Number of testing cases
        self.test_training  = inputs.test_training          # Test training case [Bool]
        self.t0, self.t_end = inputs.t0, inputs.t_end       # Time limits for testing

        self.inp_dim        = inputs.inp_dim                # Input Space Dimensionality
        self.dim            = inputs.dim                    # Data dimensionality

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
            print(self.header + '  Reading', reac_name, 'rates from file')

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
            if T_i in rates_coeff: continue
            rates_coeff[T_i] = dict()
            for reac in self.react_file:
                data_file = self.rates_path + 'T' + T_i + 'K/' + self.react_file[reac]
                rates_coeff[T_i][reac] = read_rates(data_file, reac)
        return rates_coeff


    def build_matrices(self):
        '''Build Master Equation matrices.'''

        print('\n' + self.header + 'Building Master Equation matrices ...')

        mat = dict()
        rates_coeff = self.get_rates()
        for i, T_i in enumerate(self.T_str):
            Q_bins_T = self.compute_partition_func(self.Ttr[i])
            mat[T_i] = self.ODE_sol.get_matrices(rates_coeff[T_i], Q_bins_T, self.Ttr[i])
        return mat


    ###########################################################################
    ### Initial Conditions
    ###########################################################################
    def initial_cond(self):

        print('\n' + self.header + 'Generating initial conditions ...')

        n = self.scenarios*2 + self.test_cases    # '*2' for validation cases
        unit_vec = np.array([[1.]*self.inp_dim])

        # Contructing Design Matrix -------------------------------------------
        X       = lhs(self.inp_dim, samples=n)
        X[:,:1] = np.random.choice(self.Ttr, size=(n,1))
        X[:,1:] = np.hstack(tuple(map(lambda i,x: X[:,i:i+1]*(x[1] - x[0]) + x[0], \
                                     range(1,4), [self.T0, self.P0, self.XO2])))
        # Sort wrt the norm of the projection
        norm_fn = lambda i,x: (X[:,i:i+1] - x[0])/(x[1] - x[0]) \
                              if x[1] != x[0] else np.ones(X[:,i:i+1].shape)
        X_norm  = np.hstack(tuple(map(norm_fn, range(0,4), [self.Ttr, self.T0, self.P0, self.XO2])))
        proj    = (np.dot(X_norm, unit_vec.transpose()) \
                    / np.dot(unit_vec, unit_vec.transpose())) * unit_vec
        X       = X[np.linalg.norm(proj, axis=1).argsort()]

        # Generating Initial Conditions ---------------------------------------
        ic = np.zeros((n,self.num_bins+1), dtype=np.float64)
        for i in tqdm(range(n)):
            Ttr, T0, P0, XO2 = [ X[i,j] for j in range(4) ]
            XO        = 1. - XO2
            nd        = P0 / (const.UKb * T0)
            Q_bins    = self.compute_partition_func(T0)
            ic[i,:-1] = XO2 * Q_bins / sum(Q_bins) * 2.0 * const.mO * nd
            ic[i, -1] = XO  * const.mO * nd
        ic = np.hstack((X[:,0:1], ic))

        if self.test_cases > 0:
            idx_train, idx_test = self.stratified_sampling(np.arange(n), self.test_cases)
            return [ic[idx_train,:], ic[idx_test,:]]
        else:
            return [ic]


    ###########################################################################
    ### Training/Validation Data
    ###########################################################################
    def training_data(self, ic_train, eval_jac=True, eval_f=False):

        print('\n' + self.header + 'Generating training pts ...')

        data = {}
        for i, ic_i in enumerate(tqdm(ic_train, file=sys.stdout)):

            # Solving the ODE
            mat = self.ODE_mat[str(int(ic_i[0]))]
            sol = self.ODE_sol.solve(ic_i[1:], mat, eval_jac=eval_jac, eval_f=eval_f)
            t_i, y_i = sol[0], sol[1]
            if eval_jac:
                norm_J_i, eig_J_i = sol[2], sol[3]
            if eval_f:
                f_rhs = sol[-1]

            # Define data type
            if (i % 2) == 0:
                data_type, pts = 'train', self.train_pts
            else:
                data_type, pts = 'valid', self.valid_pts

            # Sample data
            n_pts  = sum([ n for k,n in pts.items() if k != 'ics' ])
            idx    = self.get_indices(t_i, norm_J_i, n_pts)
            data_i = self.get_data(ic_i, t_i, y_i, pts, idx)
            data   = self.stack_data(data, data_i, data_type)

        # Save data in a `.csv` file
        for data_type in data:
            for data_id in data[data_type]:
                if data[data_type][data_id] is None:
                    continue
                path = self.data_dir+data_type+"/"
                if not os.path.exists(path):
                    os.makedirs(path)
                unpack = *data[data_type][data_id][0], data[data_type][data_id][1]
                pd.DataFrame(np.hstack(unpack)).to_csv(path+data_id+".csv", \
                    index=False, header=None)

        return data['train'], data['valid']

    def extract_data_scenario(self, idx, ic, t, y):
        m = np.shape(idx)[0]
        T = np.tile(ic[0 ], (m,1))
        u = np.tile(ic[1:], (m,1))
        t, y = t[idx,:], y[idx,:]
        u, y = self.dens2mass_frac(u, y)
        # return [[T, u, t], y]
        return [[u, t], y]

    def get_data(self, ic, t, y, pts, idx):
        data = {}
        for k, n in pts.items():
            if n == 0 or n is None:
                data[k] = None
            else:
                if k == 'ics':
                    idx_k = [0]
                else:
                    if k == 'pts':
                        if n == 1:
                            idx_k = [idx[-1]]
                            idx = np.delete(idx, -1)
                        else:
                            idx, idx_k = self.stratified_sampling(idx, n)
                    else:
                        idx_k = idx
                data[k] = self.extract_data_scenario(idx_k, ic, t, y)
        return data

    def stack_data(self, data, data_new, data_type):
        if data_type not in data:
            data[data_type] = data_new
        else:
            for data_id in data[data_type]:
                if data[data_type][data_id] is None:
                    continue
                for i, xy in enumerate(data[data_type][data_id]):
                    xy_new = data_new[data_id][i]
                    if isinstance(xy, (list,tuple)):
                        xy = list(map(lambda x,y: np.vstack((x,y)), xy, xy_new))
                    else:
                        xy = np.vstack((xy,xy_new))
                    data[data_type][data_id][i] = xy
        return data


    ###########################################################################
    ### Testing Data
    ###########################################################################
    def testing_data(self, ic_test, ic_train=None, eval_jac=False, eval_f=False):

        print('\n' + self.header + 'Generating testing pts ...')

        if self.test_training:      # Test underfitting
            assert self.scenarios >= self.test_cases
            n       = self.test_cases
            idx     = np.random.choice(ic_train.shape[0], size=n, replace=False)
            ic_test = np.vstack((ic_test, ic_train[idx, :]))

        for i, ic_i in enumerate(tqdm(ic_test, file=sys.stdout)):

            # Solving the ODE
            T_i = str(int(ic_i[0]))
            sol = self.ODE_sol.solve(ic_i[1:], self.ODE_mat[T_i], \
                                     eval_jac=eval_jac, eval_f=eval_f)
            t_i, y_i = sol[0], sol[1]
            if eval_jac:
                norm_J_i, eig_J_i = sol[2], sol[3]
            if eval_f:
                f_rhs = sol[-1]

            # Extract data values
            tree     = KDTree(np.c_[t_i.ravel()])
            d, ii    = tree.query([[self.t0],[self.t_end]], k=1)
            t_i, y_i = t_i[ii[0]:ii[1],:], y_i[ii[0]:ii[1],:]
            u_i      = np.tile(ic_i[1:], (t_i.shape[0],1))
            u_i, y_i = self.dens2mass_frac(u_i, y_i)

            title = 'Tested case (T = '+T_i+' K)'
            if self.test_training and i > n-1:
                title = 'Trained case (T = '+T_i+' K)'

            # Collect data values
            data_i = [title, [u_i, t_i], y_i]
            if eval_f:
                data_i.append(f_rhs[ii[0]:ii[1],:])

            if i == 0:
                data_test = [tuple(data_i)]
            else:
                data_test.append(tuple(data_i))

        return data_test


    ###########################################################################
    ### Util Functions
    ###########################################################################
    # -------------------------------------------------------------------------
    # Chemical Kinetics
    # -------------------------------------------------------------------------
    def compute_partition_func(self, T):
        '''Compute bins partition function.'''
        q = self.g * np.exp( - self.EeV * const.Ue / (T * const.UKb) )
        Q_bins = np.zeros(self.num_bins)
        for i_lev in range(self.num_lev):
            Q_bins[self.bins[i_lev]] = Q_bins[self.bins[i_lev]] + q[i_lev]
        return Q_bins

    def read_levels(self):
        '''Retrieve all the ro-vib levels.'''
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
        '''Retrieve mapping of each rot-vib level to the respective bin.'''
        data = pd.read_csv(self.binMap_file, header=None, skiprows=1)
        data = data.apply(pd.to_numeric, errors='coerce')
        idx  = np.array(data[0])                                                                           
        bins = np.array(data[1]) - 1    # Level-to-Bin
        num  = max(bins) + 1
        return bins, num

    def dens2mass_frac(self, u, y):
        '''Transform variables from partial densities to mass fractions.'''
        mass = np.sum(u, axis=1, keepdims=True)
        u = np.hstack((np.divide(u, mass), mass))
        y = np.divide(y, mass)
        return u, y

    # -------------------------------------------------------------------------
    # Sampling methods
    # -------------------------------------------------------------------------
    def get_indices(self, t, QoI, pts):
        '''Get meaningful values by selecting indices.'''

        tree  = KDTree(np.c_[t.ravel()])
        QoI_del = np.amax(np.abs((QoI[:-1,:] - QoI[1:,:]) / QoI[1:,:]), axis=1, keepdims=True)
        idx_min = np.where(QoI_del > 2.e-5)[0][ 0]  # Dynamics start-point
        idx_max = np.where(QoI_del > 1.e-8)[0][-1]  # Dynamics end-point

        pts_dist = list(map(lambda i: round(self.train_split[i]*pts), range(2)))
        pts_dist.append(pts - sum(pts_dist))

        # Sampling inside the strongest nonequilibrium region
        idx_0 = self.sampling(idx_min, idx_max, t, tree, pts_dist[0])

        # Sampling the initial dynamics
        idx_1 = self.sampling(5, idx_min-2, t, tree, pts_dist[1])

        # Sampling the equilibrium
        idx_2 = self.sampling(idx_max+2, idx_max+500, t, tree, pts_dist[2])

        # Merging indices
        idx = np.sort(np.concatenate([idx_0,idx_1,idx_2]))
        _, counts = np.unique(idx, return_counts=True)
        assert pts == np.sum(counts)

        return idx

    def sampling(self, idx_min, idx_max, t, tree, pts):
        '''Get data by sampling t vector.'''

        t_min, t_max= t[idx_min], t[idx_max]
        if self.train_pts_dist == 'random':
            d, ii  = tree.query([[t_min],[t_max]], k=1)
            idx    = np.sort(np.random.choice(ii[1]-ii[0], pts, replace=False)) + ii[0]
        else:
            t_vec  = np.linspace(np.log10(t_min), np.log10(t_max), pts, dtype=np.float64)
            h      = (t_vec[1,0]-t_vec[0,0])*0.9
            delta  = np.vstack(([[0.]], (np.random.rand(pts-2,1)-0.5)*h, [[0.]]))
            t_unif = np.power(10, t_vec + delta)
            d, idx = tree.query(t_unif, k=1)
        return idx

    def stratified_sampling(self, x, strata):
        '''Stratified sampling.'''

        x_split = np.array_split(x, strata)
        x_1 = np.array([ np.random.choice(i) for i in x_split ])
        x_2 = np.setdiff1d(x, x_1)
        assert np.intersect1d(x_1, x_2).size == 0
        return x_2, x_1
