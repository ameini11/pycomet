import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from pycomet.utilities import utils


class AdaptiveSolver(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        model,
        noneqController,
        chemModels
    ):
        self.model = model
        self.noneqController = noneqController
        self.chemModels = chemModels

    # Solve adaptive technique
    ###########################################################################
    def solve(
        self,
        x,
        Y,
        exact=False
    ):
        # Data scenario -------------------------------------------------------
        ic, self.t = x

        # Nonequilibrium controller -------------------------------------------
        self.df_norm = self.noneqController.eval_df_norm([x, Y])[-1]
        mask = tf.nest.map_structure(
            lambda x: x > self.noneqController.tol, self.df_norm
        )
        
        # Groups resolution ---------------------------------------------------
        self.group_res = self.eval_group_res()
        
        # Get mass fractions evolution ----------------------------------------
        y = self.model.net.call(x, neq_mask=mask)
        y = tf.concat([y[0][:,1:2], y[-1]], 1)

        return self.t, y

    # Groups resolution
    ###########################################################################
    def get_rec_groups(self):
        rec_groups = []
        # Get mask for deeper level of group splitting
        ts_masks = tf.nest.map_structure(
            lambda x: x > self.noneqController.tol, self.df_norm
        )
        # Initialize grouping mask array
        n = utils.get_cardinality(self.df_norm)
        for r in range(n):
            grouping_mask = np.array([None]*self.chemModels[-1].n_bins)
            ts_masks_r = tf.nest.map_structure(lambda x: x[r], ts_masks)
            # Loop over timescales
            for i, masks_i in enumerate(ts_masks_r):
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
            rec_groups.append(grouping_mask)
        return np.vstack(rec_groups)
    
    def eval_group_res(self):
        group_res = []
        rec_groups = self.get_rec_groups()
        for i in range(len(self.chemModels)):
            bins_ratio = int(
                self.chemModels[-1].n_bins / self.chemModels[i].n_bins
            )
            group_res.append(
                np.sum(rec_groups == i, axis=1, keepdims=True) / bins_ratio
            )
        return np.hstack(group_res)

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

        # Set up figure -------------------------------------------------------
        if title is not None:
            ax.set_title(title)
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
