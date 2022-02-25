import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from prode.utilities import utils


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

    # Solve adaptive technique
    ###########################################################################
    def solve(
        self,
        x,
        Y,
        Y_pred
    ):
        # Nonequilibrium controller -------------------------------------------
        self.df_norm = self.noneqController.eval_df_norm([x, Y])[-1]

        # Data scenario -------------------------------------------------------
        ic, self.t = x
        T = ic[0][:1]
        Y_pred[0] = np.take(Y_pred[0], [0], axis=1)

        # Thermochemical models -----------------------------------------------
        for chem in self.chemModels:
            chem.update(utils.reshape_to_2DArray(T))
        # Get reconstructed groups
        self.rec_groups = []
        self.Y_rec = self.reconstruct_Y(Y_pred)
        # print('Y_rec', tf.nest.map_structure(lambda x: x.shape, self.Y_rec))

        # Integrate -----------------------------------------------------------
        y = []
        # print('time', tf.nest.map_structure(lambda x: x.shape, self.t))
        for i in range(len(self.t)):
            y.append(self.apply_noneq_control(i))
        y = np.vstack(y)
        # print(y.shape)
        # yO = 1.-np.sum(y, axis=1, keepdims=True)
        # print(yO.shape)
        y = np.concatenate((1.-np.sum(y, axis=1, keepdims=True), y), axis=1)

        self.eval_group_res()

        return [self.t, y]

    # Reconstructed mass fractions
    ###########################################################################
    def reconstruct_Y(self, Y):
        # Get partition functions of the high-resolved grouping
        Q = self.chemModels[-1].spTh['O2'].intPF.Q
        # Reconstruct groups for the high-resolved grouping from
        # the low-resolved ones
        Y_rec = []
        for i, chem in enumerate(self.chemModels):
            if i == len(self.chemModels)-1:
                Y_rec.append(Y[i])
            else:
                # Get grouped partition functions ratio
                Q_i = chem.spTh['O2'].intPF.Q
                Q_i = tf.split(Q_i, Q_i.shape[1], axis=1)
                Q_i = [tf.gather(
                        Q, self.noneqController.ts_fast_conn[i][j], axis=1
                    ) / Q_i[j] for j in range(len(Q_i))]
                # Reconstruct groups
                Y_rec.append(tf.concat(
                    tf.nest.map_structure(
                        lambda x, y: x*y, Q_i, tf.split(Y[i], Y[i].shape[1], 1)
                    ), axis=1
                ))
        # Convert to numpy.array
        Y_rec = tf.nest.map_structure(lambda x: np.array(x), Y_rec)
        return Y_rec

    # Nonequilibrium controller
    ###########################################################################
    def apply_noneq_control(self, idx):
        # Get nonequilibrium distance at specific time instant
        df_norm_ = tf.nest.map_structure(lambda x: x[idx], self.df_norm)
        # print('df_norm', tf.nest.map_structure(lambda x: x.shape, df_norm_))
        # Update O2 mass fractions
        Y_ = self.update_Y(df_norm_, idx)
        # print('Y updated', tf.nest.map_structure(lambda x: x.shape, Y_))
        return Y_

    def update_Y(self, df_norm, idx):
        # Get reconstructed mass fractions at specific time instant
        Y_rec_ = tf.nest.map_structure(
            lambda x: x[idx].reshape(1,-1), self.Y_rec
        )
        # Get grouping mask
        grouping_mask = self.get_grouping_mask(df_norm)
        # Initialize new mass fractions vector
        Y = Y_rec_[-1]
        if all(grouping_mask == len(self.chemModels)-1):
            return Y
        else:
            for i, chem in enumerate(self.chemModels):
                if i in grouping_mask:
                    idx = np.where(grouping_mask == i)
                    Y = tf.nest.map_structure(
                        lambda x, y: utils.replace(x, y, idx, axis=1),
                        Y,
                        Y_rec_[i]
                    )
            return Y

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
            bins_ratio = int(
                self.chemModels[-1].n_bins / self.chemModels[i].n_bins
            )
            self.group_res.append(
                np.sum(self.rec_groups == i, axis=1, keepdims=True) / bins_ratio
            )
        self.group_res = np.hstack(self.group_res)

    def plot_group_res(
        self,
        path=None,
        var_names=['RVE1', 'RVE3', 'RVE9', 'RVE27', 'Total'],
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

        # # Manipulate variable -------------------------------------------------
        # x = self.t
        # y = [self.group_res[:,0]]
        # for i in range(1,self.group_res.shape[1]):
        #     y.append(y[-1]+self.group_res[:,i])
        # y = tf.nest.map_structure(lambda x: x.astype(self.t_vec.dtype), y)

        # Set up figure -------------------------------------------------------
        if title is not None:
            ax.set_title(title)
            # ax.set_title(r'$\epsilon_{tol} = {%s}$' % self.noneqController.tol))
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

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Plot figure ---------------------------------------------------------
        style = dict(
            marker='^',
            fillstyle='none',
            markersize=5,
            markevery=self.t.shape[0]//20,
            ls='--',
            lw=1.
        )
        n = self.group_res.shape[1]
        colors = []
        # Loop over variables
        for i in range(n):
            ax.plot(self.t, self.group_res[:,i], **style)
            colors.append(plt.gca().lines[-1].get_color())
        # Total number of groups
        ax.plot(self.t, np.sum(self.group_res, axis=1), c='k', lw=1.)
        colors.append(plt.gca().lines[-1].get_color())

        # Add legend
        if var_names is not None:
            ax.legend(
                [ax.plot([], [], c=colors[i])[0] for i in range(n+1)],
                [var_names[i] for i in range(n+1)],
                fontsize=12
            )

        # Save figure ---------------------------------------------------------
        if save:
            fig_path = path + '/' + fig_name + '.pdf'
            fig.savefig(fig_path)

        if fig_ax is not None:
            return fig, ax
