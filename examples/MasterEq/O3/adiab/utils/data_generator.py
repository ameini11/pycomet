import re
import os
import sys
import numpy as np
import pandas as pd
import joblib as jl
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
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
        self.train_split    = [0.75, 0.17, 0.08]            # Training pts splitting
        # self.train_split    = [0.92, 0.08]            # Training pts splitting

        self.test_cases     = inputs.test_cases             # Number of testing cases
        self.test_training  = inputs.test_training          # Test training case [Bool]
        self.t0, self.t_end = inputs.t0, inputs.t_end       # Time limits for testing

        self.dim            = inputs.dim                    # Data dimensionality

        # Input Space Variables -----------------------------------------------
        self.inp_dim = inputs.inp_dim                       # Input Space Dimensionality
        inp_var = [inputs.Ttr, inputs.T0, inputs.P0, inputs.XO2]
        self.Ttr, self.T0, self.P0, self.XO2 = list(map(lambda x: \
            np.array(x, dtype=np.float64), inp_var))
        self.T_str   = list(map(lambda x: str(int(x)), self.Ttr))   # Temp. in string form

        # Retrieving Molecular data -------------------------------------------
        self.levels_file = inputs.database_path + \
            '/levels/O3_UMN/O2/FromUMN_Sorted.inp'
        self.binMap_file = inputs.database_path + \
            '/grouping/O3_UMN/O2/LevelsMap_' + inputs.grouping + '.csv'
        self.g, self.EeV, self.num_lev = self.read_levels()
        self.bins, self.num_bins = self.read_groups()

        # Retrieving Rates coefficients ---------------------------------------
        self.rates_path = inputs.database_path + '/kinetics/O3_UMN_' + \
            inputs.grouping + '/'
        self.reac_file  = {'diss': 'Diss_Corrected.dat', \
            'exch': 'Exch_Type1.dat', 'inel': 'Inel.dat'}

        # Building Master Equation matrices -----------------------------------
        mat_name     = list(inputs.MEq_mat.keys())
        mat_mask     = list(inputs.MEq_mat.values())
        self.ODE_sol = ODESolver(mat_name, mat_mask)
        self.arrhenius_coeff = pd.read_csv( \
            self.rates_path+'arrhenius_coeff.csv',index_col=0)

        # Labeling Data -------------------------------------------------------
        self.groups_label = [ 'Y_'+str(i+1) for i in range(self.num_bins) ]
        data_label_inp = ['T', 'rho'] + self.groups_label + ['Y_O', 't']
        data_label_out = self.groups_label + ['Y_O']
        self.data_label = data_label_inp + data_label_out


    ###########################################################################
    ### Master Equation matrices
    ###########################################################################
    def modified_arrhenius(self, T, A, n, Ea):
        return A * np.power(T,n) * np.exp(-Ea/(const.Ru*T))

    def get_rates(self, T):
        '''Retrieve rates coefficients from dataframe.'''
        rates_coeff = dict()
        for reac_name in self.reac_file:
            shape = (self.num_bins,) if reac_name == 'diss' \
                else (self.num_bins, self.num_bins)
            rates_coeff[reac_name] = np.zeros(shape, dtype=np.float64)

        for index, row in self.arrhenius_coeff.iterrows():
            rate = self.modified_arrhenius(T, row['A'], row['n'], row['Ea'])
            reac_name, *bins_idx = index.split('_')
            if reac_name == 'diss':
                idx = int(bins_idx[0])-1
            else:
                idx = int(bins_idx[0])-1, int(bins_idx[1])-1
            rates_coeff[reac_name][idx] = rate
        return rates_coeff

    def build_matrices(self, T):
        '''Build Master Equation matrices.'''
        Q_bins = self.compute_partition_func(T)
        rates_coeff = self.get_rates(T)
        return self.ODE_sol.get_matrices(rates_coeff, Q_bins, T)


    ###########################################################################
    ### Initial Conditions
    ###########################################################################
    def initial_cond(self):

        print('\n' + self.header + 'Generating initial conditions ...')

        n = self.scenarios*2 + self.test_cases    # '*2' for validation cases
        unit_vec = np.array([[1.]*self.inp_dim])

        # Contructing Design Matrix -------------------------------------------
        self.X_col = ['Ttr', 'T0', 'P0', 'X_O2']
        X = lhs(self.inp_dim, samples=n)
        X = np.hstack(tuple(map(lambda i,x: X[:,i:i+1]*(x[1] - x[0]) + x[0], \
            range(4), [self.Ttr, self.T0, self.P0, self.XO2])))
        # Sort wrt the norm of the projection
        norm_fn = lambda i,x: (X[:,i:i+1] - x[0])/(x[1] - x[0]) \
                              if x[1] != x[0] else np.ones(X[:,i:i+1].shape)
        X_norm = np.hstack(tuple(map(norm_fn, range(0,4), \
            [self.Ttr, self.T0, self.P0, self.XO2])))
        proj = (np.dot(X_norm, unit_vec.transpose()) \
            / np.dot(unit_vec, unit_vec.transpose())) * unit_vec
        X = X[np.linalg.norm(proj, axis=1).argsort()]

        # Generating Initial Conditions ---------------------------------------
        rho_s = np.zeros((n,self.num_bins+1), dtype=np.float64)
        for i in tqdm(range(n)):
            Ttr, T0, P0, XO2 = [ X[i,j] for j in range(4) ]
            XO        = 1. - XO2
            nd        = P0 / (const.UKb * T0)
            Q_bins    = self.compute_partition_func(T0)
            rho_s[i,:-1] = XO2 * Q_bins / sum(Q_bins) * 2.0 * const.mO * nd
            rho_s[i, -1] = XO  * const.mO * nd
        Ttr, rho = X[:,0:1], np.sum(rho_s, axis=1, keepdims=True)
        ic = np.hstack((Ttr, rho, np.divide(rho_s, rho)))

        if self.test_cases > 0:
            idx_train, idx_test = self.stratified_sampling(np.arange(n), self.test_cases)
            self.X_train, self.X_test = X[idx_train], X[idx_test]
            return [ic[idx_train], ic[idx_test]]
        else:
            self.X_train, self.X_test = X, None
            return [ic]


    ###########################################################################
    ### Training/Validation Data
    ###########################################################################
    def training_data(self, ic_train, eval_jac=True, eval_f=False):

        print('\n' + self.header + 'Generating training pts ...')

        data_list = jl.Parallel(n_jobs=-1)(jl.delayed(self.single_train_scenario)( \
            i, ic_i, eval_jac, eval_f) for i, ic_i in enumerate(tqdm(ic_train)))

        data = {}
        for i, data_i in enumerate(data_list):
            # Define data type
            data_type = 'train' if (i % 2) == 0 else 'valid'
            # if (i % 2) == 0:
            #     marker_style = dict(marker='o', lw=0.5, ls='none', \
            #                         fillstyle='full', markersize=2)
            #     plt.xscale('log')
            #     plt.yscale('log')
            #     plt.plot(data_i['res'][0][1], data_i['res'][1], **marker_style)
            #     plt.plot(data_i['pts'][0][1], data_i['pts'][1], c='k', **marker_style)
            #     plt.grid()
            #     plt.show()
            data = self.stack_data(data, data_i, data_type)

        # Save data in a `.csv` file
        for data_type in data:
            for data_id in data[data_type]:
                if data[data_type][data_id] is None:
                    continue
                path = self.data_dir+data_type+"/"
                if not os.path.exists(path):
                    os.makedirs(path)
                unpacked_data = *data[data_type][data_id][0], data[data_type][data_id][1]
                pd.DataFrame(np.hstack(unpacked_data), columns=self.data_label).to_csv( \
                    path+data_id+".csv", index=False, float_format='{:e}'.format)
            # Print input space
            i_0 = 0 if data_type == 'train' else 1
            pd.DataFrame(self.X_train[range(i_0,i,2)], columns=self.X_col).to_csv( \
                path+"input_space.csv", index=False, float_format='{:e}'.format)

        return data['train'], data['valid']

    def single_train_scenario(self, i, ic_i, eval_jac, eval_f):
        # Solving the ODE
        mat = self.build_matrices(ic_i[0])
        rho_0 = ic_i[2:] * ic_i[1]
        sol = self.ODE_sol.solve(rho_0, mat, eval_jac=eval_jac, eval_f=eval_f)
        t_i, y_i = np.delete(sol[0], 0, 0), np.delete(sol[1], 0, 0)/ic_i[1]
        t_i[0,:], ic_i[2:] = 0., y_i[0,:]
        if eval_jac:
            norm_J_i, eig_J_i = np.delete(sol[2], 0, 0), np.delete(sol[3], 0, 0)
        if eval_f:
            f_rhs = np.delete(sol[-1], 0, 0)
        # Define data type
        pts = self.train_pts if (i % 2) == 0 else self.valid_pts
        # Sample data
        n_pts = sum([ n for k,n in pts.items() if k != 'ics' ])
        idx = self.get_indices(t_i, norm_J_i, n_pts)
        data_i = self.get_data(ic_i, t_i, y_i, pts, idx)

        return data_i

    def extract_data_scenario(self, idx, ic, t, y):
        # Number of data pts
        n = np.shape(idx)[0]
        # Input and Target data
        u = np.tile(ic, (n,1))
        t, y = t[idx], y[idx]
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
    def testing_data(self, ic_test, ic_train=None, eval_jac=False, eval_f=False, header=None):

        print('\n' + self.header + 'Generating testing pts ...')

        if isinstance(ic_test, str):
            ic_test = pd.read_csv(ic_test, sep=',', header=header).values
        else:
            # Print input space
            path = self.data_dir+"test/"
            if not os.path.exists(path):
                os.makedirs(path)

            # Test underfitting
            if self.test_training and ic_train is not None:
                assert self.scenarios >= self.test_cases
                idx     = np.random.choice(ic_train.shape[0], size=self.test_cases, replace=False)
                ic_test = np.vstack((ic_test, ic_train[idx]))
                self.X_test = np.vstack((self.X_test, self.X_train[idx]))

            # Print ics
            pd.DataFrame(ic_test, columns=['T','rho'] + self.groups_label + ['Y_O']).to_csv( \
                path+"ics.csv", index=False, float_format='{:e}'.format)

            # Print and plot input space
            path = path+"input_space/"
            if not os.path.exists(path):
                os.makedirs(path)
            pd.DataFrame(self.X_test, columns=self.X_col, \
                index=list(range(1,self.test_cases*2+1))).to_csv( \
                path+"data.csv", float_format='{:e}'.format)
            self.plot_inp_space(path)

        # Generate data
        # data_test = []
        # for i, ic_i in enumerate(tqdm(ic_test)):
        #     data_test.append(self.single_test_scenario(i, ic_i, eval_jac, eval_f))
        data_test = jl.Parallel(n_jobs=-1)(jl.delayed(self.single_test_scenario)( \
            i, ic_i, eval_jac, eval_f) for i, ic_i in enumerate(tqdm(ic_test)))

        return data_test

    def single_test_scenario(self, i, ic_i, eval_jac, eval_f):

        # Solving the ODE
        mat = self.build_matrices(ic_i[0])
        rho_0 = ic_i[2:] * ic_i[1]
        sol = self.ODE_sol.solve(rho_0, mat, eval_jac=eval_jac, eval_f=eval_f)
        t_i, y_i = sol[0], sol[1]/ic_i[1]
        ic_i[2:] = y_i[1]
        if eval_jac:
            norm_J_i, eig_J_i = sol[2], sol[3]
        if eval_f:
            f_rhs = sol[-1]

        # Extract data values
        tree     = KDTree(np.c_[t_i.ravel()])
        d, ii    = tree.query([[self.t0],[self.t_end]], k=1)
        t_i, y_i = t_i[ii[0]:ii[1]], y_i[ii[0]:ii[1]]
        u_i      = np.tile(ic_i, (t_i.shape[0],1))

        title = 'Tested case'
        if self.test_training and i > self.test_cases-1:
            title = 'Trained case'

        # Collect data values
        data_i = [title, [u_i, t_i], y_i]
        if eval_f:
            data_i.append(f_rhs[ii[0]:ii[1],:])

        return data_i

    def plot_inp_space(self, path):
        lbl = [ r'$T_0$', r'$P_0$', r'${X_O}_2$' ]
        X_tr = self.X_train[range(0,len(self.X_train),2)]
        for i, X_i in enumerate(self.X_test):
            fig = plt.figure()
            ax  = plt.axes(projection='3d')
            ax.set_xlabel(lbl[0])
            ax.set_ylabel(lbl[1])
            ax.set_zlabel(lbl[2])

            marker_style = dict(marker='x', lw=0.5, linestyle='none', \
                fillstyle='none', markersize=5)
            ax.plot3D(X_tr[:,1], X_tr[:,2], X_tr[:,3], c='k', **marker_style)
            ax.plot3D(X_i[1], X_i[2], X_i[3], c='r', **marker_style)

            plt.grid()
            fig.savefig(path+'/test_'+str(i+1)+'.pdf')
            plt.close()


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

    def dens2mass_frac(self, u, y=None):
        '''Transform variables from partial densities to mass fractions.'''
        rho = np.sum(u, axis=1, keepdims=True)
        u = np.hstack((np.divide(u, rho), rho))
        if y is not None:
            y = np.divide(y, rho)
            return u, y
        else:
            return u

    # -------------------------------------------------------------------------
    # Sampling methods
    # -------------------------------------------------------------------------
    # def get_indices(self, t, QoI, pts):
    #     '''Get meaningful values by selecting indices.'''

    #     tree  = KDTree(np.c_[t.ravel()])
    #     QoI_del = np.amax(np.abs((QoI[:-1,:] - QoI[1:,:]) / QoI[1:,:]), axis=1, keepdims=True)
    #     idx_max = np.where(QoI_del > 1.e-13)[0][-1]  # Dynamics end-point

    #     pts_0 = round(self.train_split[0]*pts)
    #     pts_1 = pts - pts_0

    #     # Sampling inside the strongest nonequilibrium region
    #     idx_0 = self.sampling(1, idx_max, t, tree, pts_0)

    #     # Sampling the equilibrium
    #     idx_1 = self.sampling(idx_max+2, idx_max+500, t, tree, pts_1)

    #     # Merging indices
    #     idx = np.sort(np.concatenate([idx_0,idx_1]))
    #     _, counts = np.unique(idx, return_counts=True)
    #     assert pts == np.sum(counts)

    #     return idx

    def get_indices(self, t, QoI, pts):
        '''Get meaningful values by selecting indices.'''

        tree  = KDTree(np.c_[t.ravel()])
        QoI_del = np.amax(np.abs((QoI[:-1,:] - QoI[1:,:]) / QoI[1:,:]), axis=1, keepdims=True)

        # Get min and max indices
        idx_min, delta_min = None, 1.e-5
        while idx_min is None:
            try:
                idx_min = np.where(QoI_del > delta_min)[0][ 0]  # Dynamics start-point
            except:
                idx_min = None
                delta_min = delta_min * 0.99
        idx_max = np.where(QoI_del > 1.e-13)[0][-1]  # Dynamics end-point

        # Split the number of pts among the 3 different regions of sampling
        pts_dist = list(map(lambda i: round(self.train_split[i]*pts), range(2)))
        pts_dist.append(pts - sum(pts_dist))

        # Sampling indices
        idx = []

        # Sampling the equilibrium region
        if idx_max+2+pts_dist[2] <= QoI.shape[0]:
            idx.append(self.sampling(idx_max+2, min(idx_max+500, QoI.shape[0]), t, tree, pts_dist[2]))
        else:
            pts_dist[0] = pts_dist[0] + pts_dist[2]

        # Sampling the initial dynamics region
        if idx_min-2 >= 5+pts_dist[1]:
            idx.append(self.sampling(5, idx_min-2, t, tree, pts_dist[1]))
        else:
            pts_dist[0] = pts_dist[0] + pts_dist[1]

        # Sampling inside the strongest nonequilibrium region
        idx.append(self.sampling(idx_min, idx_max, t, tree, pts_dist[0]))

        # Merging indices
        idx = np.sort(np.concatenate(idx))
        _, counts = np.unique(idx, return_counts=True)
        assert pts == np.sum(counts)

        return idx

    def sampling(self, idx_min, idx_max, t, tree, pts):
        '''Get data by sampling t vector.'''

        t_min, t_max= t[idx_min], t[idx_max]
        if self.train_pts_dist == 'random':
            _, ii  = tree.query([[t_min],[t_max]], k=1)
            idx    = np.sort(np.random.choice(ii[1]-ii[0], pts, replace=False)) + ii[0]
        else:
            t_vec  = np.linspace(np.log10(t_min), np.log10(t_max), pts, dtype=np.float64)
            h      = (t_vec[1,0]-t_vec[0,0])*0.9
            delta  = np.vstack(([[0.]], (np.random.rand(pts-2,1)-0.5)*h, [[0.]]))
            t_unif = np.power(10, t_vec + delta)
            _, idx = tree.query(t_unif, k=1)
        return idx

    def stratified_sampling(self, x, strata):
        '''Stratified sampling.'''

        x_split = np.array_split(x, strata)
        x_1 = np.array([ np.random.choice(i) for i in x_split ])
        x_2 = np.setdiff1d(x, x_1)
        assert np.intersect1d(x_1, x_2).size == 0
        return x_2, x_1
