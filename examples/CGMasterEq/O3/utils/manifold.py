from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import re
import os

from scipy.spatial import KDTree
from pyDOE import lhs

from const          import const
from ode_solver     import ODESolver
from data_generator import DataGenerator

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



class Manifold(DataGenerator):

    ###########################################################################
    ### Initialization
    ###########################################################################
    def __init__(self, inputs):
        super(Manifold, self).__init__(inputs)

        self.path   = './fig/manifold/'
        self.header = '[PrODE - Manifold]: '

        # Generating initial conditions ---------------------------------------
        self.ic, self.X = self.initial_cond()


    ###########################################################################
    ### Initial Conditions
    ###########################################################################
    def initial_cond(self):

        print('\n' + self.header + 'Generating initial conditions ...')

        n = self.train_cases
        # if n > 2: n = n-2

        # Contructing Design Matrix -------------------------------------------
        X       = lhs(3, samples=n)
        X       = np.hstack((np.random.choice(self.Ttr, size=(n,1)), X))
        X[:,1:] = np.hstack(tuple(map(lambda i,x: X[:,i:i+1]*(x[1] - x[0]) + x[0], \
                                     range(1,4), [self.T0, self.P0, self.XO2])))
        # if n > 2:
        #     X_0   = np.array([self.Ttr[ 0], self.T0[ 0], self.P0[ 0], self.XO2[ 0]])
        #     X_end = np.array([self.Ttr[-1], self.T0[-1], self.P0[-1], self.XO2[-1]])
        #     X     = np.vstack((X_0, X, X_end))

        # Generating Initial Conditions ---------------------------------------
        ic = np.zeros((X.shape[0], self.dim), dtype=np.float64)
        for i, X_i in enumerate(X):

            Ttr, T0, P0, XO2 = [ X_i[j] for j in range(4) ]
            XO  = 1. - XO2
            nd  = P0 / (const.UKb * T0)

            Q_bins    = self.compute_partition_func(T0)
            ic[i,:-1] = XO2 * Q_bins / sum(Q_bins) * 2.0 * const.mO * nd
            ic[i, -1] = XO  * const.mO * nd
        ic = np.hstack((X[:,0:1], ic))

        return ic, X


    ###########################################################################
    ### Trajectories
    ###########################################################################
    def get_trajectories(self, points=False):

        print('\n' + self.header + 'Generating trajectories ...')

        trajectories, t_scales, norm_jac = [], [], []
        for ic_i in tqdm(self.ic, file=sys.stdout):

            # Solving the ODE
            mat = self.ODE_mat[str(int(ic_i[0]))]
            t, y, norm_J, eig_J = self.ODE_sol.solve(ic_i[1:], mat, eval_jac=True)
            if self.mass_frac: y = y / np.sum(ic_i[1:])
            assert (y > 0.).all()

            if points:
                idx = self.get_indices(ic_i[1:], t, norm_J, self.train_pts)
                t, y = t[idx,:], y[idx,:]

            marker_style = dict(marker='o', lw=0.5, ls='none', \
                                fillstyle='full', markersize=2)
            plt.xscale('log')
            plt.plot(t, y, **marker_style)
            plt.grid()
            plt.show()
            
            trajectories.append((t, y))
            norm_jac.append((t, norm_J))
            t_scales.append((t, np.reciprocal(np.absolute(eig_J.real))))

        return trajectories, t_scales, norm_jac


    ###########################################################################
    ### Plotting
    ###########################################################################
    def plot_setup(self):

        fig = plt.figure()
        ax  = plt.axes(projection='3d')
        ax.set_xlabel(r'$Y_1$')
        ax.set_ylabel(r'$Y_2$')
        ax.set_zlabel(r'$Y_3$')
        return fig, ax

    def plot_manifold_struc(self, trajectories, points=False):

        fig, ax = self.plot_setup()
        path = self.path + '/struc/'
        if points: path = path + 'pts/'
        if not os.path.exists(path): os.makedirs(path)

        if self.P0[0] != self.P0[1]:
            plt.title(r'Variable Pressure: $P\in[%.2e,%.2e]$ Pa' % (self.P0[0],self.P0[1]))
            path = path+'P_var.pdf'
        elif self.XO2[0] != self.XO2[1]:
            plt.title(r'Variable Composition: $X_{O_2}\in[%.2f,%.2f]$' % (self.XO2[0],self.XO2[1]))
            path = path+'X_var.pdf'
        else:
            plt.title(r'Variable Temperature: $T\in[%d,%d]$ K' % (self.T0[0],self.T0[1]))
            path = path+'T_var.pdf'

        for i, data in enumerate(trajectories):
            t, y = data
            if points: ax.scatter3D(y[:,0], y[:,1], y[:,2], c='k', marker='o', lw=1)
            else: ax.plot3D(y[:,0], y[:,1], y[:,2], c='k')
            ax.scatter3D(y[0,0], y[0,1], y[0,2], c='r', marker='o', lw=1)
        fig.savefig(path)
        plt.show()
        plt.close()

    def plot_manifold_time(self, trajectories):

        fig, ax = self.plot_setup()
        path = self.path + '/time/'
        if not os.path.exists(path): os.makedirs(path)

        if self.P0[0] != self.P0[1]:
            plt.title(r'Variable Pressure: $P\in[%.2e,%.2e]$ Pa' % (self.P0[0],self.P0[1]))
            path = path+'P_var.pdf'
            clm  = 2
            dt   = 5.e-8
            lbl  = ' Pa'
        elif self.XO2[0] != self.XO2[1]:
            plt.title(r'Variable Composition: $X_{O_2}\in[%.1f,%.1f]$' % (self.XO2[0],self.XO2[1]))
            path = path+'X_var.pdf'
            clm  = 3
            dt   = 5.e-8
            lbl  = ''
        else:
            plt.title(r'Variable Temperature: $T\in[%d,%d]$ K' % (self.T0[0],self.T0[1]))
            path = path+'T_var.pdf'
            clm  = 1
            dt   = 1.e-6
            lbl  = ' K'

        col = ['r', 'g', 'b', 'tab:purple', 'tab:brown', 'tab:pink', \
               'tab:blue', 'tab:red', 'tab:green', 'tab:orange']

        for i, data_i in enumerate(trajectories):
            t, y = data
            ax.plot3D(y[:,0], y[:,1], y[:,2], 'k')

            tree   = KDTree(np.c_[t.ravel()])
            t_unif = np.array([[dt*i] for i in range(1,50)])
            d, idx = tree.query(t_unif, k=1)
            ax.scatter3D(y[0,0],   y[0,1],   y[0,2],   c='k',    marker='o', lw=2)
            ax.scatter3D(y[idx,0], y[idx,1], y[idx,2], c=col[i], marker='o', lw=2, \
                         label=('%.2e'+lbl) % self.X[i, clm])
        plt.legend()
        fig.savefig(path)
        plt.close()

    def plot_timescales(self, timescales):
        t, t_scales = timescales
        path = self.path + '/time/'
        if not os.path.exists(path): os.makedirs(path)

        fig = plt.figure()
        plt.xlabel(r'$t\quad[s]$')
        plt.ylabel('Timescales')

        plt.loglog(t, t_scales, lw=1)
        plt.grid()
        fig.savefig(path+'/timescales.pdf')
        plt.close()

    def plot_norm_jac(self, norm_jac):
        t, norm_J = norm_jac
        path = self.path
        if not os.path.exists(path): os.makedirs(path)

        fig = plt.figure()
        plt.xlabel(r'$t\quad[s]$')
        plt.ylabel(r'$\|\|\/\mathcal{J}_i\/\|\|_\mathcal{F}$')
        plt.title('Frobenius norm of the Jacobian matrix')

        plt.loglog(t, norm_J, lw=1)
        plt.grid()
        fig.savefig(path+'/norm_jac.pdf')
        plt.show()
        plt.close()

    def plot_norm_jac_error(self, norm_jac):
        t, norm_J = norm_jac
        path = self.path
        if not os.path.exists(path): os.makedirs(path)

        fig = plt.figure()
        plt.xlabel(r'$t\quad[s]$')
        plt.ylabel(r'$\|\|\/\mathcal{J}_i\/\|\|_\mathcal{F}$')
        plt.title('Frobenius norm of the Jacobian matrix')

        norm_J_del = np.amax(np.abs((norm_J[:-1,:] - norm_J[1:,:]) / norm_J[1:,:]), axis=1, keepdims=True)
        idx_min = np.where(norm_J_del > 1.e-6)[0][0]    # Dynamics start-point
        idx_max = np.where(norm_J_del > 1.e-12)[0][-1]  # Dynamics end-point

        plt.semilogx(t[:-1,:], norm_J_del, lw=1)
        plt.scatter(t[idx_min,:], norm_J_del[idx_min,:], marker='x')
        plt.scatter(t[idx_max,:], norm_J_del[idx_max,:], marker='x')
        plt.grid()

        fig.savefig(path+'/norm_jac_error.pdf')
        plt.show()
        plt.close()
