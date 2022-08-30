import os
import numpy as np
import pandas as pd
import pycomet as pcm
import tensorflow as tf

from tqdm import tqdm
from pyDOE import lhs


class DataGenerator(pcm.data.DataGenerator):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        train_pts,
        validation_split,
        test_cases,
        testing,
        test_training,
        time_limits,
        inp_space,
        params,
        *args,
        **kwargs
    ):
        super(DataGenerator, self).__init__()

        self.data_dir = "./data/"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Data Quantities -----------------------------------------------------
        # Training/Validation pts
        self.scenarios = train_pts['ics']
        self.valid_pts = {
            k: round(validation_split * train_pts[k]) if k != 'ics' \
                else self.scenarios for k in train_pts
        }
        self.train_pts = {
            k: train_pts[k] - self.valid_pts[k] if k != 'ics' \
                else self.scenarios for k in train_pts
        }
        for k in train_pts:
            if self.valid_pts[k] == 0 and self.train_pts[k] != 0:
                self.valid_pts[k] = 1

        # Testing scenarios
        self.testing = testing
        self.test_cases = test_cases            # Number of testing cases
        self.test_training = test_training      # Test training case [Bool]

        # Input Space Variables -----------------------------------------------
        self.t0, self.t_end = time_limits
        self.inp_space = np.vstack(list(inp_space.values())).T

        # Model parameters ----------------------------------------------------
        self.psi = params['c'] / (2.*np.sqrt(params['m'] * params['k']))
        if not 0. <= self.psi <= 1.:
            utils._raise_value_err(
                'The damping ratio "psi" has to be between 0 and 1. '
                'Modify the parameters of the physical model.'
            )
        self.w_natural = np.sqrt(params['k'] * params['m'])
        self.w_damped = self.w_natural * np.sqrt(1. - self.psi**2)

        # Labeling Data -------------------------------------------------------
        self.data_label_inp = ['x [m]', 'v [m/s]', 't [s]']
        self.data_label_out = ['x [m]', 'v [m/s]']
        self.data_label = self.data_label_inp + self.data_label_out

    def get_data_src(
        self
    ):
        ics = self.compute_ics()
        self.data_src.update(self.training_data(ics[0]))
        if self.testing and self.test_cases > 0:
            self.data_src['test'] = self.testing_data(ics[1], ics_train=ics[0])
        return self.data_src

    # Initial conditions
    ###########################################################################
    def compute_ics(
        self
    ):
        dim = self.inp_space.shape[1]
        samples = self.scenarios * 2 + self.test_cases

        # Contructing design matrix -------------------------------------------
        x_norm = lhs(dim, samples)
        x = x_norm * (self.inp_space[1]-self.inp_space[0]) + self.inp_space[0]

        # Sorting w.r.t. the projection norm
        proj = np.dot(x_norm, np.ones((dim,1))).squeeze()
        ics = x[proj.argsort()]

        # Returning initial conditions ----------------------------------------
        if self.test_cases > 0:
            idx_train, idx_test = self.stratified_sampling(
                np.arange(samples), self.test_cases
            )
            return [ics[idx_train], ics[idx_test]]
        else:
            return [ics]

    # Analytical solution
    ###########################################################################
    def solution(self, ic, t):
        # Initial conditions
        x0, v0 = ic
        # Coefficients
        A = np.sqrt(
            (self.w_damped**2*x0**2 + (v0+self.psi*self.w_natural*x0)**2) \
            / self.w_damped**2
        )
        phi = np.arctan(self.w_damped*x0 / (v0+self.psi*self.w_natural*x0))
        # Factors
        f_exp = np.exp(-self.psi*self.w_natural*t)
        f_sin = np.sin(self.w_damped * t + phi)
        f_cos = np.cos(self.w_damped * t + phi)
        # Position
        x = A * f_exp * f_sin
        # Velocity
        v = -self.psi*self.w_natural*x + self.w_damped * A * f_exp * f_cos
        return np.hstack((x, v))

    # Training/Validation data
    ###########################################################################
    def training_data(
        self,
        ics
    ):
        data = {}
        for i, ic in enumerate(ics):
            # Generate data
            data_pts = self.train_pts if (i % 2) == 0 else self.valid_pts
            data_i = self.get_data(ic, data_pts)
            # Collect data
            data_type = 'train' if (i % 2) == 0 else 'valid'
            data = self.stack_data(data, data_i, data_type)
        # Save data in a `.csv` file
        for data_type in data:
            for data_id in data[data_type]:
                if data[data_type][data_id] is None:
                    continue
                path = self.data_dir + data_type + "/"
                if not os.path.exists(path):
                    os.makedirs(path)
                _data = np.hstack(tf.nest.flatten(data[data_type][data_id]))
                pd.DataFrame(
                    _data, columns=self.data_label[:_data.shape[1]]
                ).to_csv(
                    path+data_id+".csv", index=0, float_format='{:e}'.format
                )
        return data

    def get_data(self, ic, pts):
        data = {}
        for k, n in pts.items():
            if n == 0 or n is None:
                data[k] = None
            else:
                if k == 'ics':
                    u, t = ic.reshape(1,-1), np.zeros((1,1))
                    data[k] = [[u, t], u]
                else:
                    u = np.tile(ic, (n,1))
                    t = self.sample_time(n).reshape(-1,1)
                    if k == 'pts':
                        y = self.solution(ic, t)
                        data[k] = [[u, t], y]
                    else:
                        data[k] = [[u, t]]
        return data

    def stack_data(self, data, data_new, data_type):
        if data_type not in data:
            data[data_type] = data_new
        else:
            for data_id in data[data_type]:
                if data[data_type][data_id] is None:
                    continue
                data[data_type][data_id] = tf.nest.map_structure(
                    lambda x,y : np.vstack((x,y)),
                    data[data_type][data_id],
                    data_new[data_id]
                )
        return data

    # Testing data
    ###########################################################################
    def testing_data(
        self,
        ics_test,
        ics_train=None,
        header=None,
        test_pts=100
    ):
        if isinstance(ics_test, str):
            ics_test = pd.read_csv(ics_test, sep=',', header=header).to_numpy()
        else:
            # Print input space
            path = self.data_dir + '/test/'
            if not os.path.exists(path):
                os.makedirs(path)
            # Test underfitting
            if self.test_training and ics_train is not None:
                assert self.scenarios >= self.test_cases
                idx = self.stratified_sampling(
                    np.arange(ics_train.shape[0]), self.test_cases
                )[1]
                ics_test = np.vstack((ics_test, ics_train[idx]))
        # Generate data
        data = {}
        for i, ic in enumerate(ics_test):
            u = np.tile(ic, (test_pts,1))
            t = np.linspace(self.t0, self.t_end, test_pts).reshape(-1,1)
            y = self.solution(ic, t)
            data['case_' + str(i+1).zfill(3)] = [[u, t], y]
        # Save data
        for k, v in data.items():
            pd.DataFrame(
                np.hstack(tf.nest.flatten(v)), columns=self.data_label
            ).to_csv(
                path + k + ".csv", index=0, float_format='{:e}'.format
            )
        return data

    # Util functions
    ###########################################################################
    def sample_time(self, pts):
        t = np.linspace(self.t0, self.t_end, pts+1)[1:]
        h = (t[1]-t[0])*0.9
        deltas = (np.random.rand(pts)-0.5) * h
        return t + deltas

    def stratified_sampling(self, x, strata):
        '''Stratified sampling.'''
        x_split = np.array_split(x, strata)
        x_1 = np.array([np.random.choice(i) for i in x_split])
        x_2 = np.setdiff1d(x, x_1)
        assert np.intersect1d(x_1, x_2).size == 0
        return x_2, x_1
