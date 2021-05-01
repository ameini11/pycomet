from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import re
import os

from sklearn.utils import shuffle
from scipy.spatial import KDTree
from pyDOE import lhs

from const      import const
from ode_solver import ODESolver

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



class DataGenerator(object):

    ###########################################################################
    ### Initialization
    ###########################################################################
    def __init__(self, inputs):

        self.inputs         = inputs

        # Data Quantities -----------------------------------------------------
        self.train_cases    = inputs.train_cases                    # Number of training cases
        self.valid_pts      = round(inputs.valid*inputs.train_pts)  # Number of validation v for each case
        self.train_pts      = inputs.train_pts - self.valid_pts     # Number of total training pts for each case
        self.train_pts_dist = inputs.train_pts_dist                 # Training pts distribution
        self.train_split    = [0.7, 0.2, 0.1]                       # Training pts splitting

        self.test_cases     = inputs.test_cases                     # Number of testing cases
        self.test_training  = inputs.test_training                  # Test training case [Bool]
        self.t0, self.t_end = inputs.t0, inputs.t_end               # Time limits for testing

        self.dim            = inputs.G_dim                          # Data dimensionality
        self.mass_frac      = inputs.mass_frac                      # Data in mass fractions [Bool]

        self.rtol           = 1.e-5                                 # Rel.Tol for cutting equilibrium

        self.trajectories   = []

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

        # Generating initial conditions ---------------------------------------
        self.ic_train, self.ic_test = self.initial_cond()


    ###########################################################################
    ### Master Equation matrices
    ###########################################################################
    def get_rates(self):
        '''Retrieve rates coefficients from file.'''

        # ---------------------------------------------------------------------
        def read_rates(file, reac_name):
            '''Read rates coefficients from file.'''
            print('[PrODE]:   Reading', reac_name,'rates from file')

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

        print('\n[PrODE]: Building Master Equation matrices ...')

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

        print('\n[PrODE]: Generating initial conditions ...')

        assert self.train_cases >= self.test_cases
        n = self.train_cases*2 + self.test_cases

        # Contructing Design Matrix -------------------------------------------
        X       = lhs(3, samples=n-2)
        X       = np.hstack((np.random.choice(self.Ttr, size=(n-2,1)), X))
        X[:,1:] = np.hstack(tuple(map(lambda i,x: X[:,i:i+1]*(x[1] - x[0]) + x[0], \
                                     range(1,4), [self.T0, self.P0, self.XO2])))
        X_0   = np.array([self.Ttr[ 0], self.T0[ 0], self.P0[ 0], self.XO2[ 0]])
        X_end = np.array([self.Ttr[-1], self.T0[-1], self.P0[-1], self.XO2[-1]])
        X     = np.vstack((X_0, X, X_end))

        # Generating Initial Conditions ---------------------------------------
        ic = np.zeros((n,self.dim), dtype=np.float64)
        for i in range(n):

            Ttr, T0, P0, XO2 = [ X[i,j] for j in range(4) ]
            XO  = 1. - XO2
            nd  = P0 / (const.UKb * T0)

            Q_bins    = self.compute_partition_func(T0)
            ic[i,:-1] = XO2 * Q_bins / sum(Q_bins) * 2.0 * const.mO * nd
            ic[i, -1] = XO  * const.mO * nd
        ic = np.hstack((X[:,0:1], ic))

        idx_train, idx_test = self.stratified_sampling(np.arange(n), self.test_cases)

        self.X_tr = X[idx_train,:]

        return ic[idx_train,:], ic[idx_test,:]


    ###########################################################################
    ### Training/Validation Data
    ###########################################################################
    def training_data(self):

        print('\n[PrODE]: Generating training pts ...')

        # ---------------------------------------------------------------------
        def get_data(idx):
            '''Extract data from arrays.'''

            m = np.shape(idx)[0]
            T = np.tile(ic_i[0 ], (m,1))    # NN Input: Translational Temperature
            u = np.tile(ic_i[1:], (m,1))    # NN Input: Initial Condition
            t = t_i[idx,:]                  # NN Input: Time
            y = y_i[idx,:]                  # NN Target
            if self.mass_frac: u, y = self.dens2mass_frac(u, y)
            return T, u, t, y
        # ---------------------------------------------------------------------

        data = {}
        for i, ic_i in enumerate(tqdm(self.ic_train, file=sys.stdout)):

            # Solving the ODE
            t_i, y_i = self.ODE_sol.solve(ic_i[1:], self.ODE_mat[str(int(ic_i[0]))])
            assert (y_i > 0.).all()

            # Define data type
            if (i % 2) == 0:
                data_id, pts = 'train', self.train_pts
                self.trajectories.append([ic_i, t_i, y_i])
            else:
                data_id, pts = 'valid', self.valid_pts

            # Get data from indices
            idx = self.get_indices(ic_i[1:], t_i, y_i, pts)
            data_i = get_data(idx)

            if i == 0 or i == 1:
                data[data_id] = data_i
            else:
                data[data_id] = list(map(lambda x,y: np.vstack((x,y)), data[data_id], data_i))

        # # Save data in a text file
        # np.savetxt('train_data.out', np.hstack(tuple(data['train'])), delimiter=',')
        # np.savetxt('valid_data.out', np.hstack(tuple(data['valid'])), delimiter=',')

        data_train = shuffle(*data['train'])
        data_valid = shuffle(*data['valid'])

        # Low manifolds analysis
        self.low_manifolds()
        # self.low_manifolds_time()

        return data_train, data_valid


    ###########################################################################
    ### Testing Data
    ###########################################################################
    def testing_data(self):

        print('\n[PrODE]: Generating testing pts ...')

        ic_test = self.ic_test
        if self.test_training: # Test underfitting
            n       = self.test_cases
            idx     = np.random.choice(self.ic_train.shape[0], size=n, replace=False)
            ic_test = np.vstack((self.ic_test, self.ic_train[idx, :]))

        for i, ic_i in enumerate(tqdm(ic_test, file=sys.stdout)):

            # Solving the ODE
            T_i = str(int(ic_i[0]))
            t_i, y_i = self.ODE_sol.solve(ic_i[1:], self.ODE_mat[T_i])
            assert (y_i >= 0.).all()

            # Extract data values
            tree     = KDTree(np.c_[t_i.ravel()])
            d, ii    = tree.query([[self.t0],[self.t_end]], k=1)
            t_i, y_i = t_i[ii[0]:ii[1],:], y_i[ii[0]:ii[1],:]
            u_i      = np.tile(ic_i[1:], (t_i.shape[0],1))
            if self.mass_frac: u_i, y_i = self.dens2mass_frac(u_i, y_i)

            title = 'Tested case (T = '+T_i+' K)'
            if self.test_training and i > n-1:
                title = 'Trained case (T = '+T_i+' K)'

            if i == 0:
                data_test = [(title, [u_i, t_i], y_i)]
            else:
                data_test.append((title, [u_i, t_i], y_i))

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
        '''Retriev all the rot-vib levels.'''

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
    def get_indices(self, ic, t, y, pts):
        '''Get meaningful values by selecting indices.'''

        tree  = KDTree(np.c_[t.ravel()])
        y_del = np.abs((y[:-1,:] - y[1:,:]) / y[1:,:])
        y_del = y_del[:,np.argmax(ic[:-1])]
        idx_min = np.where(y_del > self.rtol)[0][0]         # Dynamics start-point
        idx_max = np.where(y_del > self.rtol*1.e-1)[0][-1]  # Dynamics end-point

        pts_dist = list(map(lambda i: round(self.train_split[i]*pts), range(2)))
        pts_dist.append(pts - sum(pts_dist))

        # >>> Sampling inside the strongest nonequilibrium region
        idx_0 = self.sampling(idx_min, idx_max, t, tree, pts_dist[0])

        # >>> Sampling the initial dynamics
        idx_1 = self.sampling(10, idx_min-10, t, tree, pts_dist[1]-1)
        idx_1 = np.insert(idx_1, 0, 0)    # Add initial condition

        # >>> Sampling the equilibrium
        idx_2 = self.sampling(idx_max+10, -10, t, tree, pts_dist[2])

        # Merging indices
        idx = np.concatenate([idx_0,idx_1,idx_2])
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

    # -------------------------------------------------------------------------
    # Low manifolds analysis
    # -------------------------------------------------------------------------
    def low_manifolds(self):

        fig = plt.figure()
        ax  = plt.axes(projection='3d')

        ax.set_xlabel(r'$Y_1$')
        ax.set_ylabel(r'$Y_2$')
        ax.set_zlabel(r'$Y_3$')

        path = './low_manifolds/'
        if not os.path.exists(path): os.makedirs(path)

        if self.P0[0] != self.P0[1]:
            plt.title(r'Variable Pressure: $P\in[%.2e,%.2e]$ Pa' % (self.P0[0],self.P0[1]))
            path = path+'P_var.pdf'
        elif self.XO2[0] != self.XO2[1]:
            plt.title(r'Variable Composition: $X_{O_2}\in[%.1f,%.1f]$' % (self.XO2[0],self.XO2[1]))
            path = path+'X_var.pdf'
        else:
            plt.title(r'Variable Temperature: $T\in[%d,%d]$ K' % (self.T0[0],self.T0[1]))
            path = path+'T_var.pdf'

        for i, data_i in enumerate(self.trajectories):
            ic_i, t_i, y_i = data_i
            if self.mass_frac: y_i = y_i / np.sum(ic_i[1:])
            ax.plot3D(y_i[:,0], y_i[:,1], y_i[:,2], 'k')
            ax.scatter3D(y_i[0,0], y_i[0,1], y_i[0,2], c='r', marker='o', lw=2)

        fig.savefig(path)
        plt.show()

    def low_manifolds_time(self):

        fig = plt.figure()
        ax  = plt.axes(projection='3d')

        ax.set_xlabel(r'$Y_1$')
        ax.set_ylabel(r'$Y_2$')
        ax.set_zlabel(r'$Y_3$')

        col = ['r', 'g', 'b', 'tab:purple', 'tab:brown', 'tab:pink', \
               'tab:blue', 'tab:red', 'tab:green', 'tab:orange']

        path = './low_manifolds/time_evol'
        if not os.path.exists(path): os.makedirs(path)

        if self.P0[0] != self.P0[1]:
            plt.title(r'Variable Pressure: $P\in[%.2e,%.2e]$ Pa' % (self.P0[0],self.P0[1]))
            path = path+'P_var.pdf'
            clm  = 2
            dt   = 5.e-8
            lbl  = ' Pa'
        elif self.XO2[0] != self.XO2[1]:
            plt.title(r'Variable Composition: $X_{O_2}\in[%.1f,%.1f]$' % (self.XO2[0],self.XO2[1]))
            path = path+'X_var.pdf'
            clm  = 3
            dt   = 5.e-8
            lbl  = ''
        else:
            plt.title(r'Variable Temperature: $T\in[%d,%d]$ K' % (self.T0[0],self.T0[1]))
            path = path+'T_var.pdf'
            clm  = 1
            dt   = 1.e-6
            lbl  = ' K'

        for i, data_i in enumerate(self.trajectories):
            ic_i, t_i, y_i = data_i
            if self.mass_frac: y_i = y_i / np.sum(ic_i[1:])
            ax.plot3D(y_i[:,0], y_i[:,1], y_i[:,2], 'k')

        for i, data_i in enumerate(self.trajectories):
            ic_i, t_i, y_i = data_i
            if self.mass_frac: y_i = y_i / np.sum(ic_i[1:])

            tree   = KDTree(np.c_[t_i.ravel()])
            t_unif = np.array([[dt*i] for i in range(1,50)])
            d, idx = tree.query(t_unif, k=1)
            ax.scatter3D(y_i[0,0], y_i[0,1], y_i[0,2], c='k', marker='o', lw=2)
            ax.scatter3D(y_i[idx,0], y_i[idx,1], y_i[idx,2], c=col[i], marker='o', \
                         lw=2, label=('%.2e'+lbl) % self.X_tr[2*i, clm])

        plt.legend()
        fig.savefig(path)
        plt.show()