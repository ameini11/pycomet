import re
import os
import sys
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
        params,
        *args,
        **kwargs
    ):
        self.header = '[PyCOMET - Data Gen]: '
        self.data_dir = "./data/"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

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
        self.distribution = distribution        # Training pts distribution

        # Testing scenarios
        self.test_cases = test_cases            # Number of testing cases
        self.test_training = test_training      # Test training case [Bool]
        self.t0, self.t_end = time_limits       # Time limits for testing

        # Input Space Variables -----------------------------------------------
        self.inp_space = np.vstack(list(inp_space.values())).T

        # Check damping ratio -------------------------------------------------
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

# Input space (limit values)
input_space = {
    'x': [-5., 5.],     # Position [m]
    'v': [-5., 5.],     # Velocity [m/s]
}

    # Initial Conditions
    ###########################################################################
    def compute_ics(self):

        dim = self.inp_space.shape[1]
        samples = self.scenarios * 2 + self.test_cases

        # Contructing design matrix -------------------------------------------
        x_norm = lhs(dim, samples)
        x = x_norm * (inp_space[1] - inp_space[0]) + inp_space[0]

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

    # Training/Validation Data
    ###########################################################################
    def training_data(
        self,
        ics
    ):
        data = {'train': [], 'valid': []}
        for i, ic in enumerate(tqdm(ics)):
            d_pts = self.train_pts if (i % 2) == 0 else self.valid_pts
            d_type = 'train' if (i % 2) == 0 else 'valid'
            data[d_type].append(self.get_data(ic, pts))

        data = tf.nest.map_structure(np.vstack, )

        data = {}
        for i, data_i in enumerate(data_list):
            # Define data type
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
                _data = np.hstack(self.unpack_data(data[data_type][data_id]))
                pd.DataFrame(
                    _data, columns=self.data_label[:_data.shape[1]]
                ).to_csv(
                    path+data_id+".csv", index=False, float_format='{:e}'.format
                )
            
        return data['train'], data['valid']

    def get_data(self, ic, pts):
        data = {}
        for k, n in pts.items():
            if n == 0 or n is None:
                data[k] = None
            else:
                if k == 'ics':
                    u, t = ic.reshape((1,-1)), np.zeros((1,1))
                    data[k] = [[u, t], u]
                else:
                    u = np.tile(ic, (n,1))
                    t = self.sample_time(n)
                    if k == 'pts':
                        y = self.solution(ic, t)
                        data[k] = [[u, t], y]
                    else:
                        data[k] = [[u, t]]
        return data



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



    def get_data(self, ic, index):
        # Define data type
        pts = self.train_pts if (index % 2) == 0 else self.valid_pts
        # Sample data
        data = {}
        for k, n in pts.items():
            if n == 0 or n is None:
                data[k] = None
            else:
                if k == 'ics':
                    u, t = ic, np.zeros((1,1))
                    data[k] = [[u, t], u]
                else:
                    u = np.tile(ic, (n,1))
                    t = self.sampling(0, t.shape[0]-1, t, tree, n)
                    if k == 'pts':
                        y = self.solution(ic, t)
                        data[k] = [[u, t], y]
                    else:
                        data[k] = [[u, t]]
        return data




    def unpack_data(self, data):
        _data = []
        for xy in data:
            if isinstance(xy, (list,tuple)):
                _data.extend(xy)
            else:
                _data.append(xy)
        return _data

    def single_train_scenario(self, i, ic_i):
        # Solving the ODE
        sol = self.ODE_sol.solve(ic_i)
        t_i, y_i = sol[0], sol[1]
        # Define data type
        pts = self.train_pts if (i % 2) == 0 else self.valid_pts
        # Sample data
        data_i = self.get_data(ic_i, t_i, y_i, pts)
        return data_i

    def extract_data_scenario(self, ic, t, y):
        m = np.shape(idx)[0]
        u, t = np.tile(ic, (m,1)), t[idx]
        if y is None:
            return [[u, t]]
        else:
            return [[u, t], y[idx]]

    def get_data(self, ic, t, y, pts):
        data = {}
        for k, n in pts.items():
            if n == 0 or n is None:
                data[k] = None
            else:
                if k == 'ics':
                    u, t = ic.reshape((1,-1)), np.zeros((1,1))
                    data[k] = [[u, t], u]
                else:
                    u = np.tile(ic, (n,1))
                    t = self.sample_time(n)
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
                for i, xy in enumerate(data[data_type][data_id]):
                    xy_new = data_new[data_id][i]
                    if isinstance(xy, (list,tuple)):
                        xy = list(map(lambda x,y: np.vstack((x,y)), xy, xy_new))
                    else:
                        xy = np.vstack((xy,xy_new))
                    data[data_type][data_id][i] = xy
        return data

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
                # idx = np.random.choice(
                #     ic_train.shape[0], size=self.test_cases, replace=False
                # )
                _, idx = self.stratified_sampling(
                    np.arange(ic_train.shape[0]), self.test_cases
                )
                ic_test = np.vstack((ic_test, ic_train[idx]))

            # Print ics
            pd.DataFrame(ic_test, columns=self.data_label_inp[:-1]).to_csv(
                path+"ics.csv", index=False, float_format='{:e}'.format
            )

        # Generate data
        data_test = jl.Parallel(n_jobs=-1)(
            jl.delayed(self.single_test_scenario)(i, ic_i) \
                for i, ic_i in enumerate(tqdm(ic_test))
        )
        return data_test

    def single_test_scenario(self, i, ic_i):

        # Solving the ODE
        sol = self.ODE_sol.solve(ic_i)
        t_i, y_i = sol[0], sol[1]

        # Extract data values
        tree = KDTree(np.c_[t_i.ravel()])
        d, ii = tree.query([[self.t0],[self.t_end]], k=1)
        t_i, y_i = t_i[ii[0]:ii[1]], y_i[ii[0]:ii[1]]
        u_i = np.tile(ic_i, (t_i.shape[0],1))

        # Collect data values
        data_i = [[u_i, t_i], y_i]

        return data_i

    def single_test_scenario(self, i, ic_i):

        # Solving the ODE
        sol = self.ODE_sol.solve(ic_i)
        t_i, y_i = sol[0], sol[1]

        # Extract data values
        tree = KDTree(np.c_[t_i.ravel()])
        d, ii = tree.query([[self.t0],[self.t_end]], k=1)
        t_i, y_i = t_i[ii[0]:ii[1]], y_i[ii[0]:ii[1]]
        u_i = np.tile(ic_i, (t_i.shape[0],1))

        # Collect data values
        data_i = [[u_i, t_i], y_i]

        return data_i

    # Util Functions
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
