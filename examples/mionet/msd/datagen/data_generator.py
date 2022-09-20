import os
import numpy as np
import pandas as pd
import pycomet as pcm
import tensorflow as tf

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
        spaces,
        *args,
        **kwargs
    ):
        super(DataGenerator, self).__init__()

        self.data_dir = "./data/"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Data Quantities -----------------------------------------------------
        # Training/Validation pts
        self.scenarios = train_pts['ics'] * train_pts['params']
        self.valid_pts = {
            k: round(validation_split * train_pts[k]) if k != 'ics' \
                else self.scenarios for k in ('ics', 'pts', 'res')
        }
        self.train_pts = {
            k: train_pts[k] - self.valid_pts[k] if k != 'ics' \
                else self.scenarios for k in ('ics', 'pts', 'res')
        }
        for k in ('ics', 'pts', 'res'):
            if self.valid_pts[k] == 0 and self.train_pts[k] != 0:
                self.valid_pts[k] = 1

        # Testing scenarios
        self.testing = testing
        self.test_cases = test_cases            # Number of testing cases
        self.test_training = test_training      # Test training case [Bool]

        # Input Space Variables -----------------------------------------------
        self.t0, self.t_end = spaces['time']
        self.inp_space = np.vstack(
            list(spaces['params'].values()) + list(spaces['ics'].values())
        ).T

        # Labeling Data -------------------------------------------------------
        self.data_label_inp = ['K', 'C', 'x [m]', 'v [m/s]', 't [s]']
        self.data_label_out = ['x [m]', 'v [m/s]']
        self.data_label = self.data_label_inp + self.data_label_out

    def get_data_src(
        self
    ):
        inp_space = self.compute_inp_space()
        self.data_src.update(self.training_data(inp_space[0]))
        if self.testing and self.test_cases > 0:
            self.data_src['test'] = self.testing_data(
                inp_space[1], inp_space_train=inp_space[0]
            )
        return self.data_src

    # Initial conditions/Parameters
    ###########################################################################
    def compute_inp_space(
        self
    ):
        dim = self.inp_space.shape[1]
        samples = self.scenarios * 2 + self.test_cases

        # Contructing design matrix -------------------------------------------
        x_norm = lhs(dim, samples)
        x = x_norm * (self.inp_space[1]-self.inp_space[0]) + self.inp_space[0]

        # Sorting w.r.t. the projection norm
        proj = np.dot(x_norm, np.ones((dim,1))).squeeze()
        inp_space = x[proj.argsort()]

        # Returning initial conditions ----------------------------------------
        if self.test_cases > 0:
            idx_train, idx_test = self.stratified_sampling(
                np.arange(samples), self.test_cases
            )
            return [inp_space[idx_train], inp_space[idx_test]]
        else:
            return [inp_space]

    # Analytical solution
    ###########################################################################
    def solution(self, m, k, c, ic, t):
        # Parameters
        xi, w_natural, w_damped = self.get_derived_params(m, k, c)
        # Initial conditions
        x0, v0 = ic
        # Coefficients
        phi = np.arctan(w_damped*x0 / (v0+xi*w_natural*x0))
        A = x0 / np.sin(phi)
        # Factors
        f_exp = np.exp(-xi*w_natural*t)
        f_sin = np.sin(w_damped * t + phi)
        f_cos = np.cos(w_damped * t + phi)
        # Position
        x = A * f_exp * f_sin
        # Velocity
        v = -xi*w_natural*x + w_damped * A * f_exp * f_cos
        return np.hstack((x, v))

    def get_derived_params(self, m, k, c):
        xi = c / (2.*np.sqrt(m * k))
        if not 0. <= xi <= 1.:
            utils._raise_value_err(
                'The damping ratio "xi" has to be between 0 and 1. '
                'Modify the parameters of the physical model.'
            )
        w_natural = np.sqrt(k / m)
        w_damped = w_natural * np.sqrt(1. - xi**2)
        return xi, w_natural, w_damped

    # Training/Validation data
    ###########################################################################
    def training_data(
        self,
        inp_space
    ):
        data = {}
        for i, inp in enumerate(inp_space):
            # Generate data
            data_pts = self.train_pts if (i % 2) == 0 else self.valid_pts
            data_i = self.get_data(inp, data_pts)
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

    def get_data(self, inp, pts):
        data = {}
        # Parameters space
        m, k, c = inp[:3]
        p = np.array([k/m, c/m]).reshape(1,-1)
        # ICs space
        ic = np.take(inp, [3,4]).reshape(1,-1)
        for key, n in pts.items():
            if n == 0 or n is None:
                data[key] = None
            else:
                if key == 'ics':
                    data[key] = [[p, ic, np.zeros((1,1))], ic]
                else:
                    t = self.sample_time(n).reshape(-1,1)
                    inputs = [np.tile(p, (n,1)), np.tile(ic, (n,1)), t]
                    if key == 'pts':
                        y = self.solution(m, k, c, ic.squeeze(), t)
                        data[key] = [inputs, y]
                    else:
                        data[key] = [inputs]
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
        inp_space_test,
        inp_space_train=None,
        header=None,
        test_pts=100
    ):
        if isinstance(inp_space_test, str):
            inp_space_test = pd.read_csv(
                inp_space_test, sep=',', header=header
            ).to_numpy()
        else:
            # Print input space
            path = self.data_dir + '/test/'
            if not os.path.exists(path):
                os.makedirs(path)
            # Test underfitting
            if self.test_training and inp_space_train is not None:
                assert self.scenarios >= self.test_cases
                idx = self.stratified_sampling(
                    np.arange(inp_space_train.shape[0]), self.test_cases
                )[1]
                inp_space_test = np.vstack(
                    (inp_space_test, inp_space_train[idx])
                )
        # Generate data
        data = {}
        for i, inp in enumerate(inp_space_test):
            # Parameters space
            m, k, c = inp[:3]
            p = np.array([k/m, c/m]).reshape(1,-1)
            u1 = np.tile(p, (test_pts,1))
            # ICs space
            ic = np.take(inp, [3,4]).reshape(1,-1)
            u2 = np.tile(ic, (test_pts,1))
            # Time
            t = np.linspace(self.t0, self.t_end, test_pts).reshape(-1,1)
            # Solution
            y = self.solution(m, k, c, ic.squeeze(), t)
            data['case_' + str(i+1).zfill(3)] = [[u1, u2, t], y]
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
