import numpy as np
import tensorflow as tf

from pycomet import utils
matplotlib, plt = utils.import_matplotlib()


# Plotting Sampling Quantities
###########################################################################
def plot_pairwise(
    X,
    names,
    scales=None,
    figname='pairwise.pdf',
    figsize=(8.0, 8.0),
    show=False
):
    X = tf.nest.flatten(X)
    n = len(X[0])-1
    fig, axs = plt.subplots(
        n,
        n,
        sharex='col',
        sharey='row',
        figsize=figsize
    )
    axs = np.array(axs).reshape(n,n)
    for i in range(n):
        for j in range(n):
            if i < j:
                axs[i, j].axis('off')
            else:
                if j == 0:
                    axs[i,j].set_ylabel(names[i+1], fontsize=12)
                if i == n-1:
                    axs[i,j].set_xlabel(names[j], fontsize=12)
                if scales is not None:
                    axs[i,j].set_xscale(scales[j])
                    axs[i,j].set_yscale(scales[i+1])
                else:
                    axs[i,j].ticklabel_format(
                        axis='both',
                        style='sci',
                        scilimits=(0,0)
                    )
                for x in X:
                    axs[i,j].plot(
                        x[j],
                        x[i+1],
                        marker='o',
                        markersize=1,
                        ls='',
                        fillstyle='full'
                    )
    plt.tight_layout()
    fig.savefig(figname)
    if show:
        plt.show()
    plt.close()

def plot_hist_1d(
    x,
    xlabel,
    figname='hist_1d.pdf',
    nb_bins=40,
    show=False
):
    fig, ax = plt.subplots(tight_layout=True)
    ax.hist(x, bins=nb_bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Num. samples')
    if show:
        plt.show()
    plt.savefig(figname)
    plt.close()
    
def plot_hist_2d(
    x,
    y,
    labels,
    figname='hist_2d.pdf',
    nb_bins=40,
    show=False,
    cmap='jet'
):
    fig, ax = plt.subplots(tight_layout=True)
    ax.hist2d(x, y, bins=nb_bins, cmap=cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.savefig(figname)
    if show:
        plt.show()
    plt.close()
    
def plot_hist_3d(
    x,
    y,
    labels,
    figname='hist_3d.pdf',
    nb_bins=40,
    show=False,
    cmap='jet'
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(x, y, bins=(nb_bins,nb_bins))
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)
    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()
    cmap = cm.get_cmap(cmap)
    max_height = np.max(dz)
    min_height = np.min(dz)
    rgba = [cmap((k-min_height)/max_height) for k in dz]
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.savefig(figname)
    if show:
        plt.show()
    plt.close()