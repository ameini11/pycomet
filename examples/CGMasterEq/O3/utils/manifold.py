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



class Manifold(object):

    ###########################################################################
    ### Initialization
    ###########################################################################
    def __init__(self, inputs):

        self.path = './fig/manifold/'

        # Trajectories --------------------------------------------------------
        self.train_cases    = inputs.train_cases    # Number of cases
        self.mass_frac      = inputs.mass_frac      # Data in mass fractions [Bool]
        self.dim            = inputs.G_dim          # ODE dimensionality

        # Input Space Variables -----------------------------------------------
        inp_var = [inputs.Ttr, inputs.T0, inputs.P0, inputs.XO2]
        self.Ttr, self.T0, self.P0, self.XO2 = list(map(lambda x: np.array(x, dtype=np.float64), inp_var))
        self.T_str = list(map(lambda x: str(int(x)), self.Ttr))     # Temp. in string form

        # Retrieving Molecular data -------------------------------------------
        self.levels_file = inputs.database_path + '/levels/O3_UMN/O2/FromUMN_Sorted.inp'
        self.binMap_file = inputs.database_path + '/grouping/O3_UMN/O2/LevelsMap_' + inputs.grouping + '.csv'
        self.g, self.EeV, self.num_lev = self.read_levels()
        self.bins, self.num_bins       = self.read_groups()

        # Retrieving Rates coefficients ---------------------------------------
        self.rates_path = inputs.database_path + '/kinetics/O3_UMN_' + inputs.grouping + '/'
        self.react_file = {'diss': 'Diss_Corrected.dat', 'exch': 'Exch_Type1.dat', 'inel': 'Inel.dat'}

        # Building Master Equation matrices -----------------------------------
        mat_name     = list(inputs.MEq_mat.keys())
        mat_mask     = list(inputs.MEq_mat.values())
        self.ODE_sol = ODESolver(mat_name, mat_mask)
        self.ODE_mat = self.build_matrices()

        # Generating initial conditions ---------------------------------------
        self.ic, self.X = self.initial_cond()


    ###########################################################################
    ### Master Equation matrices
    ###########################################################################
    def get_rates(self):
        '''Retrieve rates coefficients from file.'''

        # ---------------------------------------------------------------------
        def read_rates(file, reac_name):
            '''Read rates coefficients from file.'''
            print('[PrODE - Manifold]:   Reading', reac_name,'rates from file')

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

        print('\n[PrODE - Manifold]: Building Master Equation matrices ...')

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

        print('\n[PrODE - Manifold]: Generating initial conditions ...')

        n = self.train_cases
        if n > 2: n = n-2

        # Contructing Design Matrix -------------------------------------------
        X       = lhs(3, samples=n)
        X       = np.hstack((np.random.choice(self.Ttr, size=(n,1)), X))
        X[:,1:] = np.hstack(tuple(map(lambda i,x: X[:,i:i+1]*(x[1] - x[0]) + x[0], \
                                     range(1,4), [self.T0, self.P0, self.XO2])))
        if n > 2:
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

        return ic, X


    ###########################################################################
    ### Trajectories
    ###########################################################################
    def get_trajectories(self):

        print('\n[PrODE - Manifold]: Generating trajectories ...')

        trajectories, t_scales = [], []
        for ic_i in tqdm(self.ic, file=sys.stdout):

            # Solving the ODE
            mat         = self.ODE_mat[str(int(ic_i[0]))]
            t, y, eig_J = self.ODE_sol.solve(ic_i[1:], mat, eigen_jac=True)
            if self.mass_frac: y = y / np.sum(ic_i[1:])
            assert (y > 0.).all()
            
            t_scales.append([t, np.reciprocal(np.absolute(eig_J.real))])
            trajectories.append([t, y])

        return trajectories, t_scales


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


    ###########################################################################
    ### Plotting
    ###########################################################################
    def plot_setup(self):

        fig = plt.figure()
        ax  = plt.axes(projection='3d')
        ax.set_xlabel(r'$Y_1$')
        ax.set_ylabel(r'$Y_2$')
        ax.set_zlabel(r'$Y_3$')
        return fig, ax

    def plot_manifold_struc(self, trajectories):

        fig, ax = self.plot_setup()
        path = self.path + '/struc/'
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

        for i, data in enumerate(trajectories):
            t, y = data
            ax.plot3D(y[:,0], y[:,1], y[:,2], 'k')
            ax.scatter3D(y[0,0], y[0,1], y[0,2], c='r', marker='o', lw=2)
        fig.savefig(path)
        plt.show()

    def plot_manifold_time(self, trajectories):

        fig, ax = self.plot_setup()
        path = self.path + '/time/'
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

        col = ['r', 'g', 'b', 'tab:purple', 'tab:brown', 'tab:pink', \
               'tab:blue', 'tab:red', 'tab:green', 'tab:orange']

        for i, data_i in enumerate(trajectories):
            t, y = data
            ax.plot3D(y[:,0], y[:,1], y[:,2], 'k')

            tree   = KDTree(np.c_[t.ravel()])
            t_unif = np.array([[dt*i] for i in range(1,50)])
            d, idx = tree.query(t_unif, k=1)
            ax.scatter3D(y[0,0],   y[0,1],   y[0,2],   c='k',    marker='o', lw=2)
            ax.scatter3D(y[idx,0], y[idx,1], y[idx,2], c=col[i], marker='o', lw=2, \
                         label=('%.2e'+lbl) % self.X[i, clm])
        plt.legend()
        fig.savefig(path)
        plt.show()

    def plot_timescales(self, timescales):
        t, t_scales = timescales
        path = self.path + '/time/'
        if not os.path.exists(path): os.makedirs(path)

        fig = plt.figure()
        plt.xlabel(r'$t\quad[s]$')
        plt.ylabel('Timescales')

        plt.loglog(t, t_scales, lw=1)
        plt.grid()
        fig.savefig(path+'/timescales.pdf')
        plt.close()
