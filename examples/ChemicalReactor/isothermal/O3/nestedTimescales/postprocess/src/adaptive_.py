import numpy as np
import tensorflow as tf

from scipy.integrate import ode
from prode.physics.general import const
from prode.utilities import utils


import matplotlib.pyplot as plt


class AdaptiveSolver(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        noneqController,
        chemModels
    ):
        self.noneqController = noneqController
        self.chemModels = chemModels

        self.atol = 1.e-20

    # Solve adaptive ODE
    ###########################################################################
    def fun(self, t, y):
        w = np.zeros_like(y)
        if self.chemModels[-1].processFlags['excit']:
            w = w + np.matmul(self.me_mats[0], y) * y[-1]
        w = w + np.matmul(self.me_mats[1], y) * y[-1]
        w = w + np.squeeze(self.me_mats[2]) * np.power(y[-1], 3)
        return w

    def jac(self, t, y):
        '''Jacobian calculation: jac[i,j] = df[i] / dy[j].'''
        d = np.shape(y)[0]
        J = np.zeros((d,d), dtype=np.float64)
        J[:,:-1] = (self.me_mats[0][:,:-1] + self.me_mats[1][:,:-1]) * y[-1]
        J[:, -1] = np.matmul(self.me_mats[0][:,:-1], y[:-1]) \
            + np.matmul(self.me_mats[1][:,:-1], y[:-1]) \
            + 3*np.squeeze(self.me_mats[2])*np.power(y[-1], 2)
        return J

    def solve(
        self,
        data
    ):
        # Nonequilibrium controller -------------------------------------------
        df_norm = self.noneqController.eval_df_norm(data)[-1]

        # Data scenario -------------------------------------------------------
        (ic, t), Y = data
        T, rho = ic[0][:2]
        Y_O2 = ic[0][2:]
        rho_0 = np.concatenate((Y_O2, 1. - np.sum(Y_O2, keepdims=True))) * rho
        self.t_vec = t.squeeze()

        # Thermochemical models -----------------------------------------------
        for chem in self.chemModels:
            chem.update(utils.reshape_to_2DArray(T))
        # Get rates
        self.me_mats = self.chemModels[-1].me_mats_()
        # Get reconstructed rates
        self.rec_groups = []
        self.rec_rates_mats = self.grouped_reconstructed_rates()

        # Integrator ----------------------------------------------------------
        r = ode(self.fun, self.jac).set_integrator(
            'vode', method='bdf', with_jacobian=True, atol=self.atol
        )
        r.set_initial_value(rho_0, self.t_vec[0])

        # Initialize solution
        y = np.expand_dims(rho_0,0)

        # Itegrate
        it = 1
        while r.successful() and r.t < self.t_vec[-1]:
            self.apply_noneq_control(r.t, df_norm)
            r.integrate(self.t_vec[it])
            y = np.vstack((y, np.expand_dims(r.y,0)))
            it += 1
        self.apply_noneq_control(self.t_vec[-1], df_norm)
        assert (y > 0.).all()

        self.eval_group_res()

        return [t, y/rho]

    # Reconstructed rates
    ###########################################################################
    def grouped_reconstructed_rates(self):
        # Get partition functions of the high-resolved grouping
        Q = self.chemModels[-1].spTh['O2'].intPF.Q
        # Reconstruct rates for the high-resolved grouping from
        # the low-resolved ones
        rec_rates_mats = []
        for i, chem in enumerate(self.chemModels):
            if i == len(self.chemModels)-1:
                rec_rates_mats_i = chem.rates_mats
            else:
                # Get grouped partition functions ratio
                Q_i = chem.spTh['O2'].intPF.Q
                Q_i = tf.split(Q_i, Q_i.shape[1], axis=1)
                Q_i = tf.concat(
                    [tf.gather(
                        Q, self.noneqController.ts_fast_conn[i][j], axis=1
                    ) / Q_i[j] for j in range(len(Q_i))], axis=1
                )
                # Get number of bins ratio
                bins_ratio = int(self.chemModels[-1].n_bins / chem.n_bins)
                # Loop over reaction processes
                rec_rates_mats_i = {}
                for process, rates_mat in chem.rates_mats.items():
                    if process == 'excit':
                        kron_shape = (bins_ratio, bins_ratio)
                        Q_ = Q_i
                    elif process == 'diss':
                        kron_shape = (1, bins_ratio)
                        Q_ =  tf.ones_like(Q_i)
                    elif process == 'recom':
                        kron_shape = (1, bins_ratio)
                        Q_ = Q_i
                    rec_rates_mats_i[process] = tf.experimental.numpy.kron(
                        rates_mat, tf.ones(kron_shape)
                    ) * Q_
            rec_rates_mats.append(rec_rates_mats_i)
        # Convert to numpy.array
        rec_rates_mats = tf.nest.map_structure(
            lambda x: np.array(x), rec_rates_mats
        )
        return rec_rates_mats

    # Nonequilibrium controller
    ###########################################################################
    def apply_noneq_control(self, t, df_norm):
        # Get nonequilibrium distance at specific time instant
        idx = tf.math.argmin(tf.math.abs(self.t_vec - t), axis=0)
        df_norm_ = tf.nest.map_structure(lambda x: x[idx], df_norm)
        # Update rates matrices
        rates_mats = self.update_rates_mats(df_norm_)
        # Update Master Equation matrices
        self.me_mats = self.chemModels[-1].me_mats_(rates_mats=rates_mats)

    def update_rates_mats(self, df_norm):
        # Get grouping mask
        grouping_mask = self.get_grouping_mask(df_norm)
        # Initialize new rates matrices
        rates_mats = self.rec_rates_mats[-1]
        if all(grouping_mask == len(self.chemModels)-1):
            return rates_mats
        else:
            for i, chem in enumerate(self.chemModels):
                if i in grouping_mask:
                    idx = np.where(grouping_mask == i)
                    rates_mats = tf.nest.map_structure(
                        lambda x, y: utils.replace(x, y, idx, axis=1),
                        rates_mats,
                        self.rec_rates_mats[i]
                    )
            return rates_mats

    def get_grouping_mask(self, df_norm):
        # Get mask for deeper level of group splitting
        ts_masks = tf.nest.map_structure(
            lambda x: x > self.noneqController.tol, df_norm
        )
        grouping_mask = np.array([None]*self.chemModels[-1].n_bins)
        # Loop over timescales
        for i, masks_i in enumerate(ts_masks):
            # Loop over groups in each timescale
            for j, masks_ij in enumerate(masks_i):
                if not masks_ij:
                    m1 = grouping_mask == None
                    m2 = np.isin(
                        np.arange(self.chemModels[-1].n_bins),
                        self.noneqController.ts_fast_conn[i][j]
                    )
                    grouping_mask[np.where(np.logical_and(m1, m2))] = i
        grouping_mask[np.where(grouping_mask == None)] = i+1
        self.rec_groups.append(grouping_mask)
        return grouping_mask

    # Groups resolution
    ###########################################################################
    def eval_group_res(self):
        self.group_res = []
        self.rec_groups = np.vstack(self.rec_groups)
        for i in range(len(self.chemModels)):
            self.group_res.append(
                np.sum(self.rec_groups == i, axis=1, keepdims=True)
            )
        self.group_res = np.hstack(self.group_res)

    def plot_group_res(
        self,
        path=None,
        var_names=None,
        fig_name=None,
        labels=[r'$t$ [s]', r'\# of groups'],
        scales=['log', 'linear'],
        title=None,
        xlim=None,
        ylim=None,
        save=True,
        fig_ax=None
    ):
        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax

        # Manipulate variable -------------------------------------------------
        x = self.t_vec.squeeze()
        y = [self.group_res[:,0]]
        for i in range(1,self.group_res.shape[1]):
            y.append(y[-1]+self.group_res[:,i])
        y = tf.nest.map_structure(lambda x: x.astype(self.t_vec.dtype), y)

        # Set up figure -------------------------------------------------------
        if title is not None:
            ax.set_title(title)
            (?# ax.set_title(r'$\epsilon_{tol} = {%s}$' % self.noneqController.tol))
        x_label, y_label = labels
        x_scale, y_scale = scales
        # X axis
        if x_label is not None:
            ax.set_xlabel(x_label)
        if x_scale is not None:
            ax.set_xscale(x_scale)
        if xlim is not None:
            ax.set_xlim(xlim)
        # Y axis
        if y_label is not None:
            ax.set_ylabel(y_label)
        if y_scale is not None:
            ax.set_yscale(y_scale)
        if ylim is not None:
            ax.set_ylim(ylim)

        # Plot figure ---------------------------------------------------------
        n = len(y)
        colors = []
        # Loop over variables
        for i in range(n):
            y1 = y[i]
            if i == 0:
                y2 = np.zeros_like(y1)
            else:
                y2 = y[i-1]
            ax.plot(x, y1, alpha=0.3)
            ax.fill_between(x, y1=y1, y2=y2, alpha=0.3)
            colors.append(plt.gca().lines[-1].get_color())

        # Add legend
        if var_names is not None:
            ax.legend(
                [ax.plot([], [], c=colors[i])[0] for i in range(n)],
                [var_names[i] for i in range(n)],
                fontsize='medium'
            )

        # Save figure ---------------------------------------------------------
        if save:
            fig_path = path + '/' + fig_name + '.pdf'
            fig.savefig(fig_path)
            
        if fig_ax is not None:
            return fig, ax
