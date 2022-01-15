import os
import itertools
import numpy as np
import pandas as pd
import joblib as jl

from tqdm import tqdm
from scipy.spatial import KDTree
from pyDOE import lhs

from physics.ode_solver import ODESolver
from prode.utilities import utils
from prode.physics.general import const

import prode.physics.database as db
kinetic_db = os.path.dirname(db.__file__) + '/air/kinetics/'
grouping_db = os.path.dirname(db.__file__) + '/air/grouping/'


class DataGenerator(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        path,
        train_cases,
        test_cases,
        time_limit,
        input_space,
        species,
        process_flags,
        groups_graph=None
    ):
        self.path = path + "/data/"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # Data Quantities -----------------------------------------------------
        # Training/Validation pts
        self.train_cases = train_cases

        # Testing train_cases
        self.test_cases = test_cases        # Number of testing cases
        self.t0, self.t_end = time_limit    # Time limits for testing

        # Input Space Variables -----------------------------------------------
        self.input_space = {
            k: np.array(v, dtype=np.float64) for k, v in input_space.items()
        }

        # Retrieving Molecular data -------------------------------------------
        rovib = species['O2']['rovib']
        path = grouping_db + rovib['system'] + '_' + rovib['PES'] + '/O2/'
        self.levels_file = path + '/levels.inp'
        self.binMap_file = path + '/LevelsMap_' + rovib['grouping'] + '.csv'
        self.g, self.E, self.num_lev = self.read_levels()
        self.bins, self.num_bins = self.read_groups()

        # Timescales connectivity ---------------------------------------------
        if groups_graph:
            self.ts_fast_conn = self.ts_to_fastest_connectivity(groups_graph)  # {0: [[0, 1, 2, 3, 4, 5, 6, 7, 8]], 1: [[0, 1, 2], [3, 4, 5], [6, 7, 8]], 2: None}

        # Retrieving Rates coefficients ---------------------------------------
        self.rates_path = kinetic_db + '/O3_UMN/' + rovib['grouping'] + '/'
        self.reac_names = {'diss', 'exch', 'inel'}

        # Building Master Equation matrices -----------------------------------
        self.ODE_sol = ODESolver(process_flags)
        self.arrhenius_coeff = pd.read_csv(
            self.rates_path+'arrhenius_coeff.csv', index_col=0
        )

        # Labeling Data -------------------------------------------------------
        self.groups_label = [ 'Y_'+str(i+1) for i in range(self.num_bins) ]
        self.data_label_inp = ['T', 'rho'] + self.groups_label + ['t']
        self.data_label_out = []
        for ts, indeces in self.ts_fast_conn.items():
            if ts == 0:
                self.data_label_out.append(['Y_O2', 'Y_O'])
            elif indeces is None:
                self.data_label_out.append([
                    'Y_'+str(ts)+str(i+1) for i in range(self.num_bins)
                ])
            else:
                self.data_label_out.append([
                    'Y_'+str(ts)+str(i+1) for i in range(len(indeces))
                ])
        self.data_label = self.data_label_inp
        for lbl in self.data_label_out:
            self.data_label += lbl

    # Master Equation matrices
    ###########################################################################
    def modified_arrhenius(self, T, A, n, Ea):
        return A * np.power(T,n) * np.exp(-Ea/(const.URG*T))

    def get_rates(self, T):
        '''Retrieve rates coefficients from dataframe.'''
        rates_coeff = dict()
        for reac_name in self.reac_names:
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
        Q_bins = self.partition_func(T)
        rates_coeff = self.get_rates(T)
        return self.ODE_sol.get_matrices(rates_coeff, Q_bins, T)

    # Initial Conditions
    ###########################################################################
    def initial_cond(
        self
    ):
        print("Generating initial conditions ...")

        n = self.train_cases*2 + self.test_cases    # '*2' for validation cases
        unit_vec = np.array([[1.]*len(self.input_space)])

        # Contructing Design Matrix -------------------------------------------
        X = lhs(len(self.input_space), samples=n)
        X = np.hstack(tuple(map(
            lambda i,x: X[:,i:i+1] * (x[1] - x[0]) + x[0], range(4), \
                self.input_space.values()
        )))
        # Sort wrt the norm of the projection
        norm_fn = lambda i,x: (X[:,i:i+1] - x[0])/(x[1] - x[0]) \
            if x[1] != x[0] else np.ones(X[:,i:i+1].shape)
        X_norm = np.hstack(tuple(map(
            norm_fn, range(0,4), self.input_space.values()
        )))
        proj = (np.dot(X_norm, unit_vec.transpose()) \
            / np.dot(unit_vec, unit_vec.transpose())) * unit_vec
        X = X[np.linalg.norm(proj, axis=1).argsort()]

        # Generating Initial Conditions ---------------------------------------
        rho_s = np.zeros((n,self.num_bins+1), dtype=np.float64)
        for i in tqdm(range(n)):
            Ttr, T0, P0, XO2 = [ X[i,j] for j in range(4) ]
            XO = 1. - XO2
            nd = P0 / (const.UKB * T0)
            Q_bins = self.partition_func(T0)
            rho_s[i,:-1] = XO2 * Q_bins / sum(Q_bins) * 2.0 * const.mass['O'] * nd
            rho_s[i, -1] = XO  * const.mass['O'] * nd
        Ttr, rho = X[:,0:1], np.sum(rho_s, axis=1, keepdims=True)
        ic = np.hstack((Ttr, rho, np.divide(rho_s, rho)))

        if self.test_cases > 0:
            idx_train, idx_test = self.stratified_sampling(
                np.arange(n), self.test_cases
            )
            self.X_train, self.X_test = X[idx_train], X[idx_test]
            return [ic[idx_train], ic[idx_test]]
        else:
            self.X_train, self.X_test = X, None
            return [ic]

    # Generate Data
    ###########################################################################
    def generate(
        self,
        ic,
        name='test',
        header=None
    ):
        print("Generating " + name + " data ...")

        if isinstance(ic, str):
            ic = pd.read_csv(ic, sep=',', header=header).to_numpy(dtype=np.float64)
        else:
            # Print input space
            path = self.path + name + '/'
            if not os.path.exists(path):
                os.makedirs(path)

            # Print ics
            df = pd.DataFrame(
                ic, columns=['T','rho'] + self.groups_label + ['Y_O']
            )
            df.to_csv(
                path+"ics.csv", index=False, float_format='{:e}'.format
            )

        # Generate data
        data = jl.Parallel(n_jobs=-1)(
            jl.delayed(self.single_scenario)(i, ic_i) \
                for i, ic_i in enumerate(tqdm(ic))
        )

        return data

    def single_scenario(self, i, ic_i):
        # Solving the ODE
        mat = self.build_matrices(ic_i[0])
        rho_0 = ic_i[2:] * ic_i[1]
        sol = self.ODE_sol.solve(rho_0, mat)
        t_i, y_i = sol[0], sol[1]/ic_i[1]
        ic_i[2:] = y_i[1]

        # Extract data values
        tree = KDTree(np.c_[t_i.ravel()])
        d, ii = tree.query([[self.t0],[self.t_end]], k=1)
        t_i, y_i = t_i[ii[0]:ii[1]], y_i[ii[0]:ii[1]]

        y_O2, y_O = np.hsplit(y_i, [-1])
        y_i = []
        for ts, indeces in self.ts_fast_conn.items():
            if indeces:
                y_slow = np.hstack([
                    np.sum(
                        np.take(y_O2, i, axis=1), axis=1, keepdims=True
                    ) for i in indeces
                ])
            else:
                y_slow = y_O2
            if ts == 0:
                y_slow = np.hstack([y_slow, y_O])
            y_i.append(y_slow)

        # Collect and return data values
        return [t_i, y_i]

    # Util Functions
    ###########################################################################
    # Chemical Kinetics
    # -------------------------------------------------------------------------
    def partition_func(self, T):
        '''Compute bins partition function.'''
        q = self.g * np.exp( - self.E / (T * const.UKB) )
        Q_bins = np.zeros(self.num_bins)
        for i_lev in range(self.num_lev):
            Q_bins[self.bins[i_lev]] = Q_bins[self.bins[i_lev]] + q[i_lev]
        return Q_bins

    def read_levels(self):
        '''Retrieve all the ro-vib levels.'''
        data = pd.read_csv(
            self.levels_file, header=None, skiprows=15, delim_whitespace=True
        )
        data = data.apply(pd.to_numeric, errors='coerce')
        vqn = np.array(data[0])                 # Vibrational Q.N.
        jqn = np.array(data[1])                 # Rotational  Q.N.
        g = 1.5*(2.0*jqn+1.0)                   # Degeneracies
        E = np.array(data[2]) * const.EH_to_J   # Energy in eV
        E = E - E[0]
        num = len(vqn)
        return g, E, num

    def read_groups(self):
        '''Retrieve mapping of each rot-vib level to the respective bin.'''
        data = pd.read_csv(self.binMap_file, header=None, skiprows=1)
        data = data.apply(pd.to_numeric, errors='coerce')
        idx  = np.array(data[0])
        bins = np.array(data[1]) - 1            # Level-to-Bin
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

    # Groups connectivity/indexing
    # -------------------------------------------------------------------------
    def ts_to_fastest_connectivity(self, graph, ts={}):
        conn = self.graph_to_nested_list(graph)
        TS = utils.depth(graph)
        for l in range(-1, TS):
            ts[l+1] = self.merge(conn, it_max=l+1)
        return ts

    def graph_to_nested_list(self, graph):
        if isinstance(graph, dict):
            ls = []
            for k, v in graph.items():
                ls.append(self.graph_to_nested_list(v))
            return ls
        else:
            if isinstance(graph, (list, tuple, set)):
                return list(graph)
            else:
                utils.raise_value_err(
                    "The last level of the groups graph has to be a list."
                )

    def merge(self, ls, it=0, it_max=1):
        if any(isinstance(l, list) for l in ls):
            if it < it_max:
                ls = list(itertools.chain.from_iterable(ls))
                return self.merge(ls, it=it+1, it_max=it_max)
            else:
                return self.join(ls)

    def join(self, ls):
        try:
            ls = [list(itertools.chain.from_iterable(l)) for l in ls]
            return self.join(ls)
        except:
            return ls

    # Sampling methods
    # -------------------------------------------------------------------------
    def stratified_sampling(self, x, strata):
        '''Stratified sampling.'''
        x_split = np.array_split(x, strata)
        x_1 = np.array([ np.random.choice(i) for i in x_split ])
        x_2 = np.setdiff1d(x, x_1)
        assert np.intersect1d(x_1, x_2).size == 0
        return x_2, x_1
