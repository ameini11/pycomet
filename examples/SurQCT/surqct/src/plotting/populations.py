import os
import numpy as np
import pandas as pd
import tensorflow as tf

from surqct.src.utilities import utils

# Visualization libraries
matplotlib = utils.import_matplotlib()
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
COLORS = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']


def find_nearest(array, value):
    i = 0
    iStep=[]
    for ivalue in value:
        array = np.asarray(array)
        idx = (np.abs(array - ivalue)).argmin()
        iStep.append(idx)
        i = i+1
    return iStep

def plotting(
    Path,
    System,
    Data,
    TTranVec,
    TimeShots,
    subplots_shape=(1,1),
    figsize_factor=1.5
):
    for iTemp in range(len(TempVec)):
        Temp = int(TempVec[iTemp])

        for k in Data.keys():
            Data[k]['Steps'] = find_nearest(Data[k][Temp]['box'].Time, TimeShots[Temp])

        fig, ax = plt.subplots(*subplots_shape, sharex=True, figsize=np.array([6.4,4.8])*figsize_factor)
        ax = np.array(ax).reshape(-1)

        for iStep in range(len(TimeShots[Temp])):
            for iMol, Mol in enumerate(System.Molecule):
                x = Mol.DiatData['EInt'] - Mol.DissE
                for i, k in enumerate(Data.keys()):
                    y = Data[k][Temp]['pop'][Mol.Name][Data[k]['Steps'][iStep],:]/Mol.DiatData['g']
                    ax[iStep].scatter(x, y, c=COLORS[i], s=2, label=k, alpha=0.25)
            ax[iStep].set_yscale('log')
            ax[iStep].set_title('$t$ = %.1e s' % TimeShots[int(Temp)][iStep])

        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('$\epsilon_i$ [eV]', labelpad=30)
        plt.ylabel('$N_i/g_i$ [m$^{-3}$]', labelpad=40)
        
        lines, labels  = ax[0].get_legend_handles_labels()
        plt.legend(lines, labels, frameon=False, markerscale=2.0, ncol=1, bbox_to_anchor=(1., 1.), handletextpad=0.1)
        plt.savefig(Path + f'/Pop_T{Temp}K.pdf', bbox_inches="tight")
