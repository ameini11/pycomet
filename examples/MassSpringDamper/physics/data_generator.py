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
        valid,
        distribution,
        test_cases,
        test_training,
        time_limit,
        input_space,
        params,
        *args,
        **kwargs
    ):
        self.header = '[PrODE - Data Gen]: '
        self.data_dir = "./data/"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Data Quantities -----------------------------------------------------
        # Training/Validation pts
        self.scenarios = train_pts['ics']
        self.valid_pts = {
            k: round(valid*train_pts[k]) if k != 'ics' \
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
        self.t0, self.t_end = time_limit        # Time limits for testing

        # Input Space Variables -----------------------------------------------
        self.input_space = {k: np.array(v, dtype=np.float64) \
            for k, v in input_space.items()}

        # Building Master Equation matrices -----------------------------------
        self.ODE_sol = ODESolver(params, *time_limit)

        # Labeling Data -------------------------------------------------------
        self.data_label_inp = ['x [m]', 'v [m/s]', 't [s]']
        self.data_label_out = ['x [m]', 'v [m/s]']
        self.data_label = self.data_label_inp + self.data_label_out

    # Initial Conditions
    ###########################################################################
    def initial_cond(self):

        print('\n' + self.header + 'Generating initial conditions ...')

        d = len(self.input_space)
        n = self.scenarios*2 + self.test_cases      # '*2' for validation cases
        unit_vec = np.array([[1.] * d])

        # Contructing Design Matrix -------------------------------------------
        X = lhs(len(self.input_space), samples=n)
        X = np.hstack(tuple(map(
            lambda i,x: X[:,i:i+1] * (x[1] - x[0]) + x[0], \
                range(d), self.input_space.values()
        )))

        # Sort wrt the norm of the projection
        norm_fn = lambda i,x: (X[:,i:i+1] - x[0])/(x[1] - x[0]) \
            if x[1] != x[0] else np.ones(X[:,i:i+1].shape)
        X_norm = np.hstack(tuple(map(
            norm_fn, range(d), self.input_space.values()
        )))
        proj = (np.dot(X_norm, unit_vec.transpose()) \
            / np.dot(unit_vec, unit_vec.transpose())) * unit_vec
        ic = X[np.linalg.norm(proj, axis=1).argsort()]

        # Returning Initial Conditions ----------------------------------------
        if self.test_cases > 0:
            idx_train, idx_test = self.stratified_sampling(
                np.arange(n), self.test_cases
            )
            return [ic[idx_train,:], ic[idx_test,:]]
        else:
            return [ic]

    # Training/Validation Data
    ###########################################################################
    def training_data(
        self,
        ic_train
    ):
        print('\n' + self.header + 'Generating training pts ...')

        data_list = jl.Parallel(n_jobs=-1)(
            jl.delayed(self.single_train_scenario)(i, ic_i) \
                for i, ic_i in enumerate(tqdm(ic_train))
        )

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

    def extract_data_scenario(self, idx, ic, t, y):
        m = np.shape(idx)[0]
        u, t = np.tile(ic, (m,1)), t[idx]
        if y is None:
            return [[u, t]]
        else:
            return [[u, t], y[idx]]

    def get_data(self, ic, t, y, pts):
        tree = KDTree(np.c_[t.ravel()])
        data = {}
        for k, n in pts.items():
            if n == 0 or n is None:
                data[k] = None
            else:
                if k == 'ics':
                    idx = [0]
                else:
                    idx = self.sampling(0, t.shape[0]-1, t, tree, n)
                if k == 'res':
                    y = None
                data[k] = self.extract_data_scenario(idx, ic, t, y)
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

    # Util Functions
    ###########################################################################
    def sampling(self, idx_min, idx_max, t, tree, pts):
        '''Get data by sampling t vector.'''
        t_min, t_max = t[idx_min], t[idx_max]
        if self.distribution == 'random':
            _, ii = tree.query([[t_min],[t_max]], k=1)
            idx = np.sort(np.random.choice(ii[1]-ii[0], pts, replace=False)) + ii[0]
        else:
            t_vec = np.linspace(t_min, t_max, pts, dtype=np.float64)
            h = (t_vec[1,0]-t_vec[0,0])*0.9
            dlt = np.vstack(([[0.]], (np.random.rand(pts-2,1)-0.5)*h, [[0.]]))
            t_unif = t_vec + dlt
            _, idx = tree.query(t_unif, k=1)
        return idx

    def stratified_sampling(self, x, strata):
        '''Stratified sampling.'''
        x_split = np.array_split(x, strata)
        x_1 = np.array([ np.random.choice(i) for i in x_split ])
        x_2 = np.setdiff1d(x, x_1)
        assert np.intersect1d(x_1, x_2).size == 0
        return x_2, x_1
