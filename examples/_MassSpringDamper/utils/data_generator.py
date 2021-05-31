import re
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

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
        self.train_cases    = inputs.train_cases                    # Number of training cases
        self.valid_pts      = round(inputs.valid*inputs.train_pts)  # Number of validation v for each case
        self.train_pts      = inputs.train_pts - self.valid_pts     # Number of total training pts for each case
        self.train_pts_dist = inputs.train_pts_dist                 # Training pts distribution

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

        dim = self.dim["u_dim"]
        n = self.train_cases*2 + self.test_cases    # '*2' for validation cases
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
    def training_data(self, ic_train, eval_jac=False, eval_f=False):

        print('\n' + self.header + 'Generating training pts ...')

        # ---------------------------------------------------------------------
        def get_data(idx):
            '''Extract data from arrays.'''

            m = np.shape(idx)[0]
            u = np.tile(ic_i, (m,1))    # NN Input: Initial Condition
            t = t_i[idx,:]                  # NN Input: Time
            y = y_i[idx,:]                  # NN Target
            return [u, t, y]
        # ---------------------------------------------------------------------

        data = {}
        for i, ic_i in enumerate(tqdm(ic_train, file=sys.stdout)):

            # Solving the ODE
            sol = self.ODE_sol.solve(ic_i, self.ODE_mat, eval_jac=eval_jac, eval_f=eval_f)
            t_i, y_i = sol[0], sol[1]
            if eval_jac:
                norm_J_i, eig_J_i = sol[2], sol[3]
            if eval_f:
                f_rhs = sol[-1]

            # Define data type
            if (i % 2) == 0:
                data_id, pts = 'train', self.train_pts
            else:
                data_id, pts = 'valid', self.valid_pts

            # Get data from indices
            tree = KDTree(np.c_[t_i.ravel()])
            idx = self.sampling(0, t_i.shape[0]-1, t_i, tree, pts)
            data_i = get_data(idx)
            if eval_f:
                data_i.append(f_rhs)
            if data_id == 'train':
                ohe_i  = tf.one_hot(np.hstack(([0], [-1]*(pts-1))), 1).numpy()
                data_i.append(ohe_i)

            # marker_style = dict(marker='o', lw=0.5, ls='none', \
            #                     fillstyle='full', markersize=2)
            # plt.xscale('log')
            # plt.plot(data_i[2], data_i[3], **marker_style)
            # plt.grid()
            # plt.show()

            if i == 0 or i == 1:
                data[data_id] = data_i
            else:
                data[data_id] = list(map(lambda x,y: np.vstack((x,y)), data[data_id], data_i))

        # Shuffle data
        data_train = shuffle(*data['train'])
        data_valid = shuffle(*data['valid'])

        # Save data in a `.csv` file
        pd.DataFrame(np.hstack(tuple(data_train))).to_csv( \
            self.data_dir+"train_data.csv", index=False, header=None)
        pd.DataFrame(np.hstack(tuple(data_valid))).to_csv( \
            self.data_dir+"valid_data.csv", index=False, header=None)

        return data_train, data_valid


    ###########################################################################
    ### Testing Data
    ###########################################################################
    def testing_data(self, ic_test, ic_train=None, eval_jac=False, eval_f=False):

        print('\n' + self.header + 'Generating testing pts ...')

        if self.test_training:      # Test underfitting
            assert self.train_cases >= self.test_cases
            n       = self.test_cases
            idx     = np.random.choice(ic_train.shape[0], size=n, replace=False)
            ic_test = np.vstack((ic_test, ic_train[idx, :]))

        for i, ic_i in enumerate(tqdm(ic_test, file=sys.stdout)):

            # Solving the ODE
            sol = self.ODE_sol.solve(ic_i, self.ODE_mat, \
                eval_jac=eval_jac, eval_f=eval_f)
            t_i, y_i = sol[0], sol[1]
            if eval_jac:
                norm_J_i, eig_J_i = sol[2], sol[3]
            if eval_f:
                f_rhs = sol[-1]

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
            if eval_f:
                data_i.append(f_rhs[ii[0]:ii[1],:])

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
