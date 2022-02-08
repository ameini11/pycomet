import os
import re
import itertools
import numpy as np
import pandas as pd
import joblib as jl
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.utils import shuffle
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
        dim,
        train_pts,
        validation_split,
        distribution,
        test_cases,
        test_training,
        time_limit,
        input_space,
        species,
        processFlags,
        trained_model,
        ts_to_grouping
    ):
        # Data Quantities -----------------------------------------------------
        # Training/Validation pts
        self.scenarios = train_pts['ics']
        self.valid_pts = {
            k: round(validation_split*train_pts[k]) if k != 'ics' \
                else self.scenarios for k in train_pts
        }
        self.train_pts = {
            k: train_pts[k] - self.valid_pts[k] if k != 'ics' \
                else self.scenarios for k in train_pts
        }
        for k in train_pts:
            if self.valid_pts[k] == 0 and self.train_pts[k] != 0:
                self.valid_pts[k] = 1

        self.distribution = distribution    # Training pts distribution

        # Testing scenarios
        self.test_cases = test_cases        # Number of testing cases
        self.test_training = test_training  # Test training case [Bool]
        self.t0, self.t_end = time_limit    # Time limits for testing

        # Input Space Variables -----------------------------------------------
        self.input_space = {
            k: np.array(v, dtype=np.float64) for k, v in input_space.items()
        }

        # Retrieving Molecular data -------------------------------------------
        self.species = species
        rovib = species['O2']['rovib']
        path = grouping_db + rovib['system'] + '_' + rovib['PES'] + '/O2/'
        self.levels_file = path + '/levels.inp'
        self.binMap_file = path + '/LevelsMap_' + rovib['grouping'] + '.csv'
        self.g, self.E, self.num_lev = self.read_levels()
        self.bins, self.num_bins = self.read_groups()

        # Retrieving Rates coefficients ---------------------------------------
        self.rates_path = kinetic_db + '/O3_UMN/' + rovib['grouping'] + '/'
        self.reac_names = {'diss', 'exch', 'inel'}

        # Building Master Equation matrices -----------------------------------
        self.ODE_sol = ODESolver(processFlags)
        self.ODE_sol.t_end = 1.e-13
        self.reactions_list = pd.read_csv(
            self.rates_path+'reactions.csv', index_col=0
        )
        self.process_indices = {
            'diss': 2,
            'exch': 5,
            'inel': 6
        }

        # Model ---------------------------------------------------------------
        self.model = self.get_model(trained_model)

        # Timescale to species ------------------------------------------------
        self.ts_to_grouping = ts_to_grouping
        self.ts_to_species_thermos = self.get_species_thermos(ts_to_grouping)

        # Labeling Data -------------------------------------------------------
        self.groups_label = [ 'Y_'+str(i+1) for i in range(self.num_bins) ]
        self.data_label_inp = ['T', 'rho'] + self.groups_label + ['t']
        self.data_label_out = []
        for i, spTh in enumerate(self.ts_to_species_thermos.values()):
            if i+1 == len(ts_to_grouping):
                continue
            self.data_label_out += [
                f'e_{i+1}{j+1}' for j in range(spTh['O2'].specie.num_bins)
            ]
        self.data_label = self.data_label_inp + self.data_label_out

    # Master Equation matrices
    ###########################################################################
    def arrhenius(self, T, A, beta, Ta):
        return A * np.power(T,beta) * np.exp(-Ta/T)

    def get_rates(self, T):
        '''Retrieve rates coefficients from dataframe.'''
        rates_coeff = dict()
        for reac_name in self.reac_names:
            shape = (self.num_bins,) if reac_name == 'diss' \
                else (self.num_bins, self.num_bins)
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
        '''Build Master Equation matrices.'''
        Q_bins = self.partition_function(T)
        rates_coeff = self.get_rates(T)
        return self.ODE_sol.get_matrices(rates_coeff, Q_bins, T)

    # Trained Model
    ###########################################################################
    def get_model(
        self,
        path
    ):
        # Input module
        sys.path.append(path)
        import inputs
        from inputs.general import flags
        # Network
        net = fun._net(
            input_module=inputs
        )
        # Model
        for k in ('train', 'save'):
            flags[k] = 0
        if flags['load'] not in (1, 2):
            utils.raise_value_err(
                "The network has to be restored from a saved "
                "or a checkpointed one. Modified the 'load' "
                "entry in the `flags` input variable."
            )
        return fun._model(
            flags=flags,
            input_module=inputs,
            net=net
        )

    def get_species_thermos(
        self,
        ts_to_grouping
    ):
        ts_to_species_thermos = {}
        for timescale, grouping in ts_to_grouping.items():
            species = self.species
            # Update O2 with the proper grouping strategy
            species['O2']['rovib']['grouping'] = grouping
            # Initialize species thermodynamics
            spTh = {
                name: specieMdl.specieThermos.SpecieThermos(
                    name,
                    info,
                    thermo='CoupledEnergyModes',
                    EOS='PerfectGas',
                ) for name, info in species.items()
            }
            # Add species thermodynamics to the corresponding timescale
            ts_to_species_thermos[timescale] = spTh
        return ts_to_species_thermos

    # Initial Conditions
    ###########################################################################
    def initial_cond(
        self
    ):
        n = self.scenarios*2 + self.test_cases    # '*2' for validation cases
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
            Q_bins = self.partition_function(T0)
            rho_s[i,:-1] = XO2*Q_bins/sum(Q_bins)*2.*const.mass['O']*nd
            rho_s[i, -1] = XO*const.mass['O']*nd
        Ttr, rho = X[:,0:1], np.sum(rho_s, axis=1, keepdims=True)
        ic = np.hstack((Ttr, rho, np.divide(rho_s, rho)))

        if self.test_cases > 0:
            idx_train, idx_test = self.stratified_sampling(
                np.arange(n), self.test_cases
            )
            return [ic[idx_train], ic[idx_test]]
        else:
            return [ic]

    # Training/Validation Data
    ###########################################################################
    def training_data(
        self,
        ic_train
    ):
        # Generate data
        inputs_list = jl.Parallel(n_jobs=-1)(
            jl.delayed(self.get_inputs)(i, ic_i) \
                for i, ic_i in enumerate(tqdm(ic_train))
        )
        data_list = [self.get_data(x) for x in enumerate(tqdm(inputs_list))]
        # Concatenate data
        data = {}
        for i, data_i in enumerate(data_list):
            # Define data type
            data_type = 'train' if (i % 2) == 0 else 'valid'
            # if (i % 2) == 0:
            #     marker_style = dict(marker='o', lw=0.5, ls='none', \
            #                         fillstyle='full', markersize=2)
            #     plt.xscale('log')
            #     plt.yscale('log')
            #     plt.plot(data_i['res'][0][1], np.hstack(data_i['res'][1]), **marker_style, label=self.data_label_out)
            #     plt.plot(data_i['pts'][0][1], np.hstack(data_i['pts'][1]), c='k', **marker_style)
            #     plt.legend()
            #     plt.grid()
            #     plt.show()
            if data_type not in data.keys():
                data[data_type] = data_i
            else:
                data[data_type] = tf.nest.map_structure(
                    lambda x,y: np.vstack((x,y)), data[data_type], data_i
                )
        # Save data in a `.csv` file
        for data_type in data.keys():
            for obj in data[data_type].keys():
                path = self.data_dir + data_type + "/"
                if not os.path.exists(path):
                    os.makedirs(path)
                df = pd.DataFrame(
                    np.hstack(self.unpack_data(data[data_type][obj])),
                    columns=self.data_label
                )
                df.to_csv(
                    path+obj+".csv", index=False, float_format='{:e}'.format
                )
        # Return data
        return data['train'], data['valid']

    def get_inputs(self, scenario_num, ic):
        # Define data type
        pts = self.train_pts if (scenario_num % 2) == 0 else self.valid_pts
        # Get sampling indices
        indices = {}
        for obj, n in pts.items():
            if obj == 'ics':
                indices[obj] = np.array([0])
            else:
                indices[obj] = self.sampling(1, self.max_num_pts, n)
        # Compose inputs
        idx = np.concatenate(list(indices.values()))
        x = self.compose_inputs(ic, indices=idx)
        return {
            obj: tf.nest.map_structure(lambda x: x[i], x) \
                for obj, i in indices.items()
        }

    def get_data(self, inputs):
        data = {}
        for obj, x in inputs.items():
            y = self.get_targets(x)
            data[obj] = [x, y]
        return data

    # def single_train_scenario(self, scenario_num, ic):
    #     # Define data type
    #     pts = self.train_pts if (scenario_num % 2) == 0 else self.valid_pts
    #     # Get sampling indices
    #     indices = {}
    #     for obj, n in pts.items():
    #         if obj == 'ics':
    #             indices[obj] = np.array([0])
    #         else:
    #             indices[obj] = self.sampling(1, self.max_num_pts, n)
    #     # Compose inputs
    #     idx = np.concatenate(list(indices.values()))
    #     x = self.compose_inputs(ic, indices=idx)
    #     # Get target data
    #     y = self.data_single_scenario(x=[ic, t])
    #     return {
    #         obj: tf.nest.map_structure (lambda x: x[i], [x, y]) \
    #             for obj, i in indices.items()
    #     }

    def unpack_data(self, data):
        d = []
        for xy in data:
            if isinstance(xy, (list,tuple)):
                d.extend(xy)
            else:
                d.append(xy)
        return d

    # Testing Data
    ###########################################################################
    def testing_data(
        self,
        ic_test,
        ic_train=None,
        header=None
    ):
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
                idx = np.random.choice(
                    ic_train.shape[0], size=self.test_cases, replace=False
                )
                ic_test = np.vstack((ic_test, ic_train[idx]))
            # Print ics
            df = pd.DataFrame(
                ic_test, columns=['T','rho'] + self.groups_label + ['Y_O']
            )
            df.to_csv(
                path+"ics.csv", index=False, float_format='{:e}'.format
            )
        # Generate data
        # data_list = jl.Parallel(n_jobs=-1)(
        #     jl.delayed(self.single_test_scenario)(i, ic_i) \
        #         for i, ic_i in enumerate(tqdm(ic_test))
        # )
        data_list = [
            self.single_test_scenario(i, ic_i) \
                for i, ic_i in enumerate(tqdm(ic_test))
        ]
        return data_test

    def single_test_scenario(self, ic):
        # Compose inputs
        x = self.compose_inputs(ic)
        # Get target data
        y = self.data_single_scenario(x=[ic, t])
        return [x, y]

    # Single scenario
    ###########################################################################
    # Handling inputs ---------------------------------------------------------
    def compose_inputs(self, ic, indices=None):
        ic = self.update_ic(ic)
        t = np.logspace(
            np.log10(self.t0), np.log10(self.t_end), num=self.max_num_pts
        )
        if indices:
            ic = np.tile(ic, (len(idx),1))
            t = t[idx]
        else:
            ic = np.tile(ic, (self.max_num_pts,1))
        return ic, t

    def update_ic(self, ic):
        T, rho, Y = ic[0], ic[1], ic[2:]
        # Solving the ODE
        mat = self.build_matrices(T)
        t, y = self.ODE_sol.solve(rho * Y, mat)
        # Update ic (reasonable numerical values + exclude Y_O)
        ic[2:] = y[1,:-1] / rho
        return ic

    # Handling target data ----------------------------------------------------
    def get_targets(
        self,
        x
    ):
        # Get nonequilibrium factor
        y = self.model.net.call(x, training=False, noneq_factor=True)
        # Get temperature
        T = x[0][0:1,0:1]
        # Get noneq. and eq. internal partition functions
        Q_noneq, Q_eq = self.get_partition_functions(noneq_factor=y, T=T)
        # Evaluate differential offset `alpha`
        delta_alpha = self.eval_delta_alpha(
            noneq_factor=y,
            Q_noneq=Q_noneq,
            Q_eq=Q_eq
        )
        # Evaluate and return noneq. level data
        return self.eval_noneq_level(delta_alpha)

    def get_partition_functions(
        self,
        noneq_factor,
        T
    ):
        Q_noneq, Q_eq = {}, {}
        for i, timescale in enumerate(self.ts_to_grouping.keys()):
            if i == 0:
                # Skip dissociation timescale with only 1 group
                continue
            # Update species thermodynamics
            for spTh in self.ts_to_species_thermos[timescale].values():
                spTh.intPF.update(T)
            # Get internal partition function
            Q = self.ts_to_species_thermos[timescale]['O2'].intPF.Q
            # Get split noneq. internal partition function
            Q_noneq_ = tf.split(
                tf.math.exp(noneq_factor[i]) * Q,
                self.out_dim[timescale],
                axis=-1
            )
            Q_noneq[timescale] = [
                tf.reduce_sum(Q_i, axis=-1, keepdims=True) for Q_i in Q_noneq_
            ]
            # Get split eq. internal partition function
            Q_eq_ = tf.split(Q, self.out_dim[timescale], axis=-1)
            Q_eq[timescale] = [
                tf.reduce_sum(Q_i, axis=-1, keepdims=True) for Q_i in Q_eq_
            ]
        return Q_noneq, Q_eq

    def eval_delta_alpha(
        self,
        noneq_factor,
        Q_noneq,
        Q_eq
    ):
        delta_alpha = {}
        for i, timescale in enumerate(self.ts_to_grouping.keys()):
            if i == 0:
                # Skip dissociation timescale with only 1 group
                continue
            y = tf.split(noneq_factor[i], self.out_dim[i], axis=-1)
            delta_alpha[timescale] = []
            for j in range(len(y_)):
                ratio = Q_noneq[timescale][j] / Q_eq[timescale][j]
                delta_alpha.append(tf.math.log(ratio) - y_[j])
        return delta_alpha

    def eval_noneq_level(
        self,
        delta_alpha
    ):
        # Evaluate the L2 norm of the differential offset `alpha`
        dalpha_norm = {}
        for timescale, dalpha_ts in enumerate(delta_alpha.items()):
            dalpha_norm[timescale] = [
                tf.norm(da, ord=2, axis=-1, keepdims=True) for da in dalpha_ts
            ]
        # Handle timescale names and indices
        timescale_names = tuple(dalpha_norm.keys())
        timescale_indeces = list(range(len(timescale_names)))
        timescale_indeces.reverse()
        # Construct the noneq. level from `delta_alpha` norms
        neq_lev = [dalpha_norm[timescale_names[-1]]]
        for i in timescale_indeces[1:]:
            # Get constructed noneq. level from previous timescale
            neq_lev_i = tf.concat(
                [
                    tf.math.reduce_sum(l, axis=-1, keepdims=True) \
                        for l in neq_lev[-1]
                ], axis=-1
            )
            neq_lev_i = tf.split(
                neq_lev_i, self.out_dim[timescale_names[i]], axis=-1
            )
            # Add the noneq. level from previous timescale to the L2 norm
            # of the differential offset `alpha` of the current timescale
            da_i = dalpha_norm[timescale_names[i]]
            neq_lev_i = [ neq_lev_i[j] + da_i[j] for j in range(len(da_i)) ]
            # Append the current timescale noneq. level
            neq_lev.append(neq_lev_i)
        # Concatenate and return noneq. level data
        neq_lev = [tf.concat(d, axis=-1) for d in neq_lev]
        neq_lev.reverse()
        return neq_lev

    # Extra functions
    ###########################################################################
    # Chemical Kinetics
    # -------------------------------------------------------------------------
    def partition_function(self, T):
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

    # Sampling methods
    # -------------------------------------------------------------------------
    def sampling(self, idx_min, idx_max, pts):
        '''Get data by sampling t vector.'''
        diff = idx_max - idx_min
        if self.distribution == 'random':
            idx = np.sort(np.random.choice(diff, pts, replace=False)) + idx_min
        else:
            idx = self.stratified_sampling(np.arange(diff), pts)[1] + idx_min
        return idx

    def stratified_sampling(self, x, strata):
        '''Stratified sampling.'''
        x_split = np.array_split(x, strata)
        x_1 = np.array([ np.random.choice(i) for i in x_split ])
        x_2 = np.setdiff1d(x, x_1)
        assert np.intersect1d(x_1, x_2).size == 0
        return x_2, x_1
