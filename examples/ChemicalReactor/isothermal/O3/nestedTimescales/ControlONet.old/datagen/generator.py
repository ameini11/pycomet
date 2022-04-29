import os
import re
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from pyDOE import lhs

from trained_model.physics.ode_solver import ODESolver
from pycomet.utilities import utils
from pycomet.physics.general import const
from pycomet.apps import functionObjects as fun

from pycomet.physics.thermochemistry import specie as specieMdl
import pycomet.physics.database as db
kinetic_db = os.path.dirname(db.__file__) + '/air/kinetics/'
grouping_db = os.path.dirname(db.__file__) + '/air/grouping/'


class DataGenerator(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
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
        self.max_num_pts = 1000
        self.norm_order = 2

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
        self.trained_model = self.get_model(trained_model)
        self.ts_out_dim = [
            list(ts.out_dim) for ts in self.trained_model.net.timescales
        ]

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
                f'e_{i+1}{j+1}' for j in range(spTh['O2'].specie.n_bins)
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
        # Get current working directory
        cwd = os.getcwd()
        # Input modules
        from trained_model.inputs import engine, general, network
        input_modules = {
            'engine': engine,
            'general': general,
            'network': network
        }
        # Change working directory
        os.chdir(path)
        # Network
        net = fun._net(
            input_modules=input_modules
        )
        # Model
        flags = input_modules['general'].flags
        for k in ('train', 'save'):
            flags[k] = 0
        if flags['load'] not in (1, 2):
            utils.raise_value_err(
                "The network has to be restored from a saved "
                "or a checkpointed one. Modified the 'load' "
                "entry in the `flags` input variable."
            )
        trained_model = fun._model(
            flags=flags,
            input_modules=input_modules,
            net=net
        )
        # Move to the previous working directory
        os.chdir(cwd)
        return trained_model

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
        data_list = [
            self.single_train_scenario(i, ic_i) \
                for i, ic_i in enumerate(tqdm(ic_train))
        ]
        # Concatenate data
        data = {}
        for i, data_i in enumerate(data_list):
            # Define data type
            data_type = 'train' if (i % 2) == 0 else 'valid'
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
                    np.hstack(tf.nest.flatten(data[data_type][obj])),
                    columns=self.data_label
                )
                df.to_csv(
                    path+obj+".csv", index=False, float_format='{:e}'.format
                )

    def single_train_scenario(self, scenario_num, ic):
        # Define data type
        pts = self.train_pts if (scenario_num % 2) == 0 else self.valid_pts
        # Get sampling indices
        indices = {}
        for obj, n in pts.items():
            if obj == 'ics':
                indices[obj] = np.array([0])
            else:
                indices[obj] = self.sampling(1, self.max_num_pts, n)
        # Compose/sample inputs
        idx = np.concatenate(list(indices.values()))
        inputs = self.compose_inputs(ic, indices=idx)
        # Get target data
        data = {}
        for obj, idx in indices.items():
            if obj == 'ics':
                x = tf.nest.map_structure(lambda x: x[idx], inputs)
            else:
                x = tf.nest.map_structure(
                    lambda x: np.delete(x, indices['ics'], axis=0), inputs
                )
            y = self.get_targets(x)
            data[obj] = [x, y]
        return data

    # Testing Data
    ###########################################################################
    def testing_data(
        self,
        ic_test,
        ic_train=None,
        header=None
    ):
        path = self.data_dir+"test/"
        if not os.path.exists(path):
            os.makedirs(path)
        if isinstance(ic_test, str):
            shutil.copy(ic_test, path)
            ic_test = pd.read_csv(ic_test, sep=',', header=header).to_numpy()
        else:
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
        data = [
            self.single_test_scenario(ic_i) for ic_i in tqdm(ic_test)
        ]
        # Save data in a `.csv` file
        path = path+"cases/"
        if not os.path.exists(path):
            os.makedirs(path)
        for i, d in enumerate(data):
            df = pd.DataFrame(
                np.hstack(tf.nest.flatten(d)),
                columns=self.data_label
            )
            df.to_csv(
                path+str(i+1).zfill(4)+".csv",
                index=False,
                float_format='{:e}'.format
            )

    def single_test_scenario(self, ic):
        # Compose inputs
        x = self.compose_inputs(ic)
        # Get target data
        y = self.get_targets(x)
        return [x, y]

    # Single scenario
    ###########################################################################
    # Handling inputs ---------------------------------------------------------
    def compose_inputs(self, ic, indices=None):
        ic = self.update_ic(ic)
        t = np.logspace(
            [np.log10(self.t0)], [np.log10(self.t_end)], num=self.max_num_pts
        )
        if indices is not None:
            ic = np.tile(ic, (len(indices),1))
            t = t[indices]
        else:
            ic = np.tile(ic, (self.max_num_pts,1))
        return ic, t

    def update_ic(self, ic):
        T, rho, Y = ic[0], ic[1], ic[2:]
        # Solving the ODE
        mat = self.build_matrices(T)
        t, y = self.ODE_sol.solve(rho * Y, mat)
        # Update ic (reasonable numerical values)
        ic[2:] = y[1] / rho
        # Return ic (exclude Y_O)
        return ic[:-1]

    # Handling target data ----------------------------------------------------
    def get_targets(
        self,
        x
    ):
        # Get nonequilibrium factor
        y = self.trained_model.net.call(x, training=False, noneq_factor=True)
        # Get temperature
        T = np.take(x[0], [0], axis=1)
        # Get noneq. and eq. internal partition functions
        Q_noneq, Q_eq = self.get_partition_functions(noneq_factor=y, T=T)
        # Evaluate differential offset `alpha`
        delta_alpha = self.eval_delta_alpha(
            noneq_factor=y,
            Q_noneq=Q_noneq,
            Q_eq=Q_eq
        )
        # Evaluate and return noneq. level data
        data = self.eval_noneq_level(delta_alpha)
        for i, d in enumerate(data):
            if len(self.ts_out_dim[i]) > 1:
                data[i] = tf.split(d, self.ts_out_dim[i], axis=1)
        data = tf.nest.map_structure(
            lambda x: x.numpy()/np.amax(x.numpy()), tf.nest.flatten(data)
        )
        return data

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
                self.ts_out_dim[i],
                axis=1
            )
            Q_noneq[timescale] = [
                tf.reduce_sum(Q_i, axis=1, keepdims=True) for Q_i in Q_noneq_
            ]
            # Get split eq. internal partition function
            Q_eq_ = tf.split(Q, self.ts_out_dim[i], axis=1)
            Q_eq[timescale] = [
                tf.reduce_sum(Q_i, axis=1, keepdims=True) for Q_i in Q_eq_
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
            y = tf.split(noneq_factor[i], self.ts_out_dim[i], axis=1)
            delta_alpha[timescale] = []
            for j in range(len(y)):
                ratio = Q_noneq[timescale][j] / Q_eq[timescale][j]
                delta_alpha[timescale].append(tf.math.log(ratio) - y[j])
        return delta_alpha

    def eval_noneq_level(
        self,
        delta_alpha
    ):
        # Evaluate the L2 norm of the differential offset `alpha`
        dalpha_norm = []
        for dalpha_ts in delta_alpha.values():
            norms = [
                tf.math.pow(tf.norm(
                    da, ord=self.norm_order, axis=1, keepdims=True
                ), self.norm_order) for da in dalpha_ts
            ]
            dalpha_norm.append(tf.concat(norms, axis=1))
        # Handle timescale names and indices
        timescale_indeces = list(range(len(dalpha_norm)))
        timescale_indeces.reverse()
        # Construct the noneq. level from `delta_alpha` norms
        neq_lev = [dalpha_norm[-1]]
        for i in timescale_indeces[1:]:
            # Get constructed noneq. level from previous timescale
            if len(self.ts_out_dim[i+1]) > 1:
                neq_lev_i = tf.split(neq_lev[-1], self.ts_out_dim[i+1], axis=1)
            else:
                neq_lev_i = tf.nest.flatten(neq_lev[-1])
            # Reduce dimensions by summing: previous to current timescale
            neq_lev_i = tf.concat([
                tf.math.reduce_sum(l, axis=1, keepdims=True) \
                    for l in neq_lev_i
            ], axis=1)
            # Add the noneq. level from previous timescale to the L2 norm
            # of the differential offset `alpha` of the current timescale
            neq_lev.append(neq_lev_i + dalpha_norm[i])
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
