import re
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools

from tqdm import tqdm
from sklearn.utils import shuffle
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from pyDOE import lhs

from ode_solver import ODESolver


class DataGenerator(object):

    ###########################################################################
    ### Initialization
    ###########################################################################
    def __init__(self, inputs):

        self.header   = '[PrODE - Data Gen]: '
        self.data_dir = "./data/"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Data Quantities -----------------------------------------------------
        self.scenarios = inputs.train_pts['ics']
        self.valid_pts = {k: round(inputs.valid*inputs.train_pts[k]) for k in inputs.train_pts }
        self.train_pts = {k: inputs.train_pts[k] - self.valid_pts[k]  for k in inputs.train_pts }
        self.valid_pts['ics'] = self.train_pts['ics'] = self.scenarios
        self.train_pts_dist   = inputs.train_pts_dist                 # Training pts distribution

        self.test_cases     = inputs.test_cases                     # Number of testing cases
        self.test_training  = inputs.test_training                  # Test training case [Bool]
        self.t0, self.t_end = inputs.t0, inputs.t_end               # Time limits for testing

        self.dim            = inputs.dim                            # Data dimensionality

        # Input Space Variables -----------------------------------------------
        inp_var = [inputs.pos, inputs.vel]
        self.pos, self.vel = list(map(lambda x: np.array(x, dtype=np.float64), inp_var))

        # Building Master Equation matrices -----------------------------------
        self.params  = inputs.params
        self.ODE_sol = ODESolver()
        self.ODE_mat = self.build_matrices()


    ###########################################################################
    ### ODE matrices
    ###########################################################################
    def build_matrices(self):
        '''Build Master Equation matrices.'''

        print('\n' + self.header + 'Building ODE matrices ...')

        mat = self.ODE_sol.get_matrices(self.params)
        return mat


    ###########################################################################
    ### Initial Conditions
    ###########################################################################
    def initial_cond(self):

        print('\n' + self.header + 'Generating initial conditions ...')

        dim = self.dim["inp"][0]
        n = self.scenarios * 2 + self.test_cases    # '*2' for validation cases
        unit_vec = np.array([[1.] * dim])

        # Contructing Design Matrix -------------------------------------------
        X = lhs(dim, samples=n)

        # Generating Initial Conditions ---------------------------------------
        X = np.hstack(tuple(map(lambda i,x: X[:,i:i+1]*(x[1] - x[0]) + x[0], \
                                range(dim), [self.pos, self.vel])))

        # Sort wrt the norm of the projection
        norm_fn = lambda i,x: (X[:,i:i+1] - x[0])/(x[1] - x[0]) \
                              if x[1] != x[0] else np.ones(X[:,i:i+1].shape)
        X_norm  = np.hstack(tuple(map(norm_fn, range(dim), [self.pos, self.vel])))
        proj    = (np.dot(X_norm, unit_vec.transpose()) \
                    / np.dot(unit_vec, unit_vec.transpose())) * unit_vec
        ic       = X[np.linalg.norm(proj, axis=1).argsort()]

        if self.test_cases > 0:
            idx_train, idx_test = self.stratified_sampling(np.arange(n), self.test_cases)
            return [ic[idx_train,:], ic[idx_test,:]]
        else:
            return [ic]


    ###########################################################################
    ### Training/Validation Data
    ###########################################################################
    def training_data(self, ic_train):

        print('\n' + self.header + 'Generating training pts ...')

        data = {}
        for i, ic_i in enumerate(tqdm(ic_train, file=sys.stdout)):

            # Solving the ODE
            sol = self.ODE_sol.solve(ic_i, self.ODE_mat)
            t_i, y_i = sol[0], sol[1]

            # Define data type
            if (i % 2) == 0:
                data_type, pts = 'train', self.train_pts
            else:
                data_type, pts = 'valid', self.valid_pts

            # Sample data
            data_i = self.get_data(ic_i, t_i, y_i, pts)
            data   = self.stack_data(data, data_i, data_type)

        # Save data in a `.csv` file
        for data_type in data:
            for data_id in data[data_type]:
                path = self.data_dir+data_type+"/"
                if not os.path.exists(path):
                    os.makedirs(path)
                unpack = *data[data_type][data_id][0], data[data_type][data_id][1]
                pd.DataFrame(np.hstack(unpack)).to_csv(path+data_id+".csv", \
                    index=False, header=None)

        return data['train'], data['valid']

    def extract_data_scenario(self, idx, ic, t, y):
        m = np.shape(idx)[0]
        u = np.tile(ic, (m,1))
        return [[u, t[idx,:]], y[idx,:]]

    def get_data(self, ic, t, y, pts):
        tree = KDTree(np.c_[t.ravel()])
        data = {}
        for k, v in pts.items():
            if k == 'ics':
                idx = [0]
            else:
                idx = self.sampling(0, t.shape[0]-1, t, tree, v)
            data[k] = self.extract_data_scenario(idx, ic, t, y)
        return data

    def stack_data(self, data, data_new, data_type):
        if data_type not in data:
            data[data_type] = data_new
        else:
            for data_id in data[data_type]:
                for i, xy in enumerate(data[data_type][data_id]):
                    xy_new = data_new[data_id][i]
                    if isinstance(xy, (list,tuple)):
                        xy = list(map(lambda x,y: np.vstack((x,y)), xy, xy_new))
                    else:
                        xy = np.vstack((xy,xy_new))
                    data[data_type][data_id][i] = xy
        return data


    ###########################################################################
    ### Testing Data
    ###########################################################################
    def testing_data(self, ic_test, ic_train=None):

        print('\n' + self.header + 'Generating testing pts ...')

        if self.test_training:      # Test underfitting
            assert self.num_ics >= self.test_cases
            n       = self.test_cases
            idx     = np.random.choice(ic_train.shape[0], size=n, replace=False)
            ic_test = np.vstack((ic_test, ic_train[idx, :]))

        for i, ic_i in enumerate(tqdm(ic_test, file=sys.stdout)):

            # Solving the ODE
            sol = self.ODE_sol.solve(ic_i, self.ODE_mat)
            t_i, y_i = sol[0], sol[1]

            # Extract data values
            tree     = KDTree(np.c_[t_i.ravel()])
            d, ii    = tree.query([[self.t0],[self.t_end]], k=1)
            t_i, y_i = t_i[ii[0]:ii[1],:], y_i[ii[0]:ii[1],:]
            u_i      = np.tile(ic_i, (t_i.shape[0],1))

            title = 'Tested case'
            if self.test_training and i > n-1:
                title = 'Trained case'

            # Collect data values
            data_i = [title, [u_i, t_i], y_i]

            if i == 0:
                data_test = [tuple(data_i)]
            else:
                data_test.append(tuple(data_i))

        return data_test


    ###########################################################################
    ### Util Functions
    ###########################################################################

    def sampling(self, idx_min, idx_max, t, tree, pts):
        '''Get data by sampling t vector.'''
        t_min, t_max = t[idx_min], t[idx_max]
        if self.train_pts_dist == 'random':
            d, ii  = tree.query([[t_min],[t_max]], k=1)
            idx    = np.sort(np.random.choice(ii[1]-ii[0], pts, replace=False)) + ii[0]
        else:
            t_vec  = np.linspace(t_min, t_max, pts, dtype=np.float64)
            h      = (t_vec[1,0]-t_vec[0,0])*0.9
            t_unif = t_vec + (np.random.rand(pts,1)-0.5)*h
            d, idx = tree.query(t_unif, k=1)
        return idx

    def stratified_sampling(self, x, strata):
        '''Stratified sampling.'''

        x_split = np.array_split(x, strata)
        x_1 = np.array([ np.random.choice(i) for i in x_split ])
        x_2 = np.setdiff1d(x, x_1)
        assert np.intersect1d(x_1, x_2).size == 0
        return x_2, x_1
