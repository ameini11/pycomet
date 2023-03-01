import re
import os
import numpy as np
import pandas as pd
import joblib as jl
import tensorflow as tf

from tqdm import tqdm
from pyDOE import lhs

from src.ode_solver import ODESolver
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
        nb_cases,
        t_eval,
        nb_eval,
        input_space,
        species,
        process_flags,
        nb_jobs=4
    ):
        self.nb_jobs = nb_jobs
        self.nb_cases = nb_cases

        # Input Space Variables -----------------------------------------------
        self.input_space = {
            k: np.array(v, dtype=np.float64) for k, v in input_space.items()
        }

        # Retrieving Molecular data -------------------------------------------
        rovib = species['O2']['rovib']
        path = grouping_db + rovib['system'] + '_' + rovib['PES'] + '/O2/'
        # Levels
        self.g, self.E, self.num_lev = self.read_levels(path + '/levels.inp')
        # Grouping
        self.lev_to_bin, self.nb_bins = self.read_groups(
            path + f"/LevelsMap_{rovib['grouping']}.csv"
        )

        # Retrieving Rates coefficients ---------------------------------------
        self.rates_path = kinetic_db + '/O3_UMN/' + rovib['grouping'] + '/'
        self.reac_names = {'diss', 'exch', 'inel'}

        # Building Master Equation matrices -----------------------------------
        self.ODE_sol = ODESolver(
            process_flags,
            t_eval=t_eval,
            nb_eval=nb_eval
        )
        self.reactions_list = pd.read_csv(
            self.rates_path+'reactions.csv', index_col=0
        )
        self.process_indices = {
            'diss': 2,
            'exch': 5,
            'inel': 6
        }

    # Master Equation matrices
    ###########################################################################
    def arrhenius(self, T, A, beta, Ta):
        return A * np.power(T,beta) * np.exp(-Ta/T)

    def get_rates(self, T):
        rates_coeff = dict()
        for reac_name in self.reac_names:
            shape = (self.nb_bins,) if reac_name == 'diss' \
                else (self.nb_bins, self.nb_bins)
            rates_coeff[reac_name] = np.zeros(shape, dtype=np.float64)

        for index, row in self.reactions_list.iterrows():
            rate = self.arrhenius(T, row['A'], row['beta'], row['Ta'])
            bins_idx = re.findall(r"\((.*?)\)", row['reactStr'])
            if self.process_indices['diss'] == row['reacIndex']:
                idx = int(bins_idx[0])-1
            else:
                idx = int(bins_idx[0])-1, int(bins_idx[1])-1
            reac_name = utils.get_key_from_value(
                self.process_indices, row['reacIndex']
            )
            rates_coeff[reac_name][idx] = rate
        return rates_coeff

    def build_matrices(self, T):
        Q_bins = self.partition_func(T, self.lev_to_bin, self.nb_bins)
        rates_coeff = self.get_rates(T)
        return self.ODE_sol.get_matrices(rates_coeff, Q_bins, T)

    # Initial Conditions
    ###########################################################################
    def initial_cond(
        self
    ):
        print("Generating initial conditions ...")

        n = self.nb_cases
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
        rho_s = np.zeros((n,self.nb_bins+1), dtype=np.float64)
        for i in tqdm(range(n)):
            Ttr, T0, P0, XO2 = [X[i,j] for j in range(4)]
            XO = 1.0 - XO2
            nd = P0 / (const.UKB * T0)
            Q = self.partition_func(T0, self.lev_to_bin, self.nb_bins)
            rho_s[i,:-1] = XO2 * Q / sum(Q) * 2.0 * const.mass['O'] * nd
            rho_s[i, -1] = XO  * const.mass['O'] * nd
        Ttr, rho = X[:,0:1], np.sum(rho_s, axis=1, keepdims=True)
        ic = np.hstack((Ttr, rho, np.divide(rho_s, rho)))
        return ic

    # Generate Data
    ###########################################################################
    def generate(
        self,
        ic,
        header=None
    ):
        print("Generating data ...")
        data = jl.Parallel(self.nb_jobs)(
            jl.delayed(self.single_scenario)(i, ic_i) \
                for i, ic_i in enumerate(tqdm(ic))
        )
        return data

    def single_scenario(self, i, ic_i):
        # Solving the ODE
        T, rho = ic_i[:2]
        mat = self.build_matrices(T)
        rho_0 = ic_i[2:] * rho
        sol = self.ODE_sol.solve(rho_0, mat)
        t_i, y_i = sol[0], sol[1]/rho
        ic_i[2:] = y_i[1]
        # Collect and return data values
        T_i = np.full((len(t_i),1), T)
        sol_i = np.hstack([y_i, T_i])
        return [t_i, sol_i]

    # Util Functions
    ###########################################################################
    # Chemical Kinetics
    # -------------------------------------------------------------------------
    def partition_func(self, T, lev_to_bin, nb_bins):
        q = self.g * np.exp(-self.E/(T*const.UKB))
        Q = np.array([np.sum(q[lev_to_bin == i]) for i in range(nb_bins)])
        return Q

    def read_levels(self, file):
        data = pd.read_csv(
            file, header=None, skiprows=15, delim_whitespace=True
        )
        data = data.apply(pd.to_numeric, errors='coerce')
        vqn = np.array(data[0])                 # Vibrational Q.N.
        jqn = np.array(data[1])                 # Rotational  Q.N.
        g = 1.5*(2.0*jqn+1.0)                   # Degeneracies
        E = np.array(data[2]) * const.EH_to_J   # Energy in eV
        E = E - E[0]
        nb_lev = len(vqn)
        return g, E, nb_lev

    def read_groups(self, file):
        data = pd.read_csv(file, header=None, skiprows=1)
        data = data.apply(pd.to_numeric, errors='coerce')
        idx = np.array(data[0])
        lev_to_bin = np.array(data[1])
        lev_to_bin -= np.amin(lev_to_bin)
        nb_bins = np.amax(lev_to_bin) + 1
        return lev_to_bin, nb_bins
