import numpy as np
import pandas as pd

from surqct.src.utilities import utils

# Visualization libraries
matplotlib = utils.import_matplotlib()
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
COLORS = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']


def energies(
    Path,
    System,
    Data,
    TTranVec,
    xlim=[1.e-9, 1.e-4]
):
    iTemp = 0
    fig = plt.figure()
    plt.plot([], [],'--k', label='$E_{Rot}$')
    plt.plot([], [], '-k', label='$E_{Vib}$')
    for i, k in enumerate(Data.keys()):
        plt.plot([], [], label=k, c=COLORS[i])

    for iTemp in range(len(TTranVec)):

        Temp = int(TTranVec[iTemp])
        EMed = []
        for iMol, Mol in enumerate(System.Molecules):
            for i, (k, d) in enumerate(Data.items()):
                t    = d[Temp]['box'].Time
                ERot = d[Temp]['E'][Mol.Name]['ERot']
                EVib = d[Temp]['E'][Mol.Name]['EVib']
                EMed.append((ERot[-1] + EVib[-1]) / 2)
                plt.plot(t, d[Temp]['E'][Mol.Name]['ERot'], '--', c=COLORS[i])
                plt.plot(t, d[Temp]['E'][Mol.Name]['EVib'],  '-', c=COLORS[i])
            
            # Temperature annotation
            plt.annotate(
                'T = %d K' % Temp,
                xy=(1, max(EMed)),
                xytext=(8, 0),
                xycoords=('axes fraction', 'data'),
                textcoords='offset points'
            )

        plt.legend()
        plt.xscale('log')
        plt.xlabel('$t$ [s]')
        plt.ylabel('$E$ [eV]')
        plt.xlim(xlim)

    plt.savefig(Path + f'/EVib_ERot.pdf', bbox_inches="tight")


def taus(
    Path,
    System,
    Data,
    TTranVec
):
    fig = plt.figure()
    plt.plot([], [], 'ks', label='$\\tau_{Rot}$')
    plt.plot([], [], 'k^', label='$\\tau_{Vib}$')
    for i, k in enumerate(Data.keys()):
        plt.plot([], [], label=k, c=COLORS[i])

    for iTemp in range(len(TTranVec)):

        Temp = int(TTranVec[iTemp])
        x    = Temp**(-1/3) * 1.e2
        for iMol, Mol in enumerate(System.Molecules):
            for i, (k, d) in enumerate(Data.items()):
                plt.plot(x, d[Temp]['TausP'][Mol.Name]['ERot'], ls='', marker='s', c=COLORS[i])
                plt.plot(x, d[Temp]['TausP'][Mol.Name]['EVib'], ls='', marker='^', c=COLORS[i])
            
        plt.legend()
        plt.yscale('log')
        plt.xlabel('$T^{-1/3}$ [K$^{-1/3}$] $\\times 10^2$')
        plt.ylabel('$p\cdot\\tau$ [atm$\cdot$s]')
        
    plt.savefig(Path + f'/Taus.pdf', bbox_inches="tight")
    

def mole_fraction(
    Path,
    System,
    Data,
    TTranVec,
    xlim=[1.e-9, 1.e-1],
    ylim=[0.,1.]
):
    fig = plt.figure()
    for i, k in enumerate(Data.keys()):
        plt.plot([], [], label=k, c=COLORS[i])
    
    for iTemp in range(len(TTranVec)):

        Temp = int(TTranVec[iTemp])
        for iMol, Mol in enumerate(System.Molecules):
            for i, (k, d) in enumerate(Data.items()):
                t = d[Temp]['box'].Time
                X = d[Temp]['box'][Mol.Name]
                plt.plot(t, X, c=COLORS[i])

        plt.legend()
        plt.xscale('log')
        plt.xlabel('$t$ [s]')
        plt.ylabel('$X$')
        plt.xlim(xlim)
        
    plt.savefig(Path + f'/MoleFrac.pdf', bbox_inches="tight")
    
    
def mole_depletion_delays(
    Path,
    System,
    Data,
    TTranVec
):
    fig = plt.figure()
    for i, k in enumerate(Data.keys()):
        if k == 'QCT':
            continue
        plt.plot([], [], label=k, c=COLORS[i])
    
    for iTemp in range(len(TTranVec)):

        Temp = int(TTranVec[iTemp])
        for iMol, Mol in enumerate(System.Molecules):
            DeplQCT = Data['QCT'][Temp]['XDepl'][Mol.Name]
            for i, (k, d) in enumerate(Data.items()):
                if k == 'QCT':
                    continue
                Depl  = Data[k][Temp]['XDepl'][Mol.Name]
                Delay = (DeplQCT - Depl)/DeplQCT*100
                plt.plot(Temp, Delay, ls='', marker='^', c=COLORS[i])

        plt.legend()
        plt.ticklabel_format(axis='x', style='sci', scilimits=(3,3))
        plt.xlabel('$T$ [K]')
        plt.ylabel('% Delay')
        
    plt.savefig(Path + f'/Delays.pdf', bbox_inches="tight")
    
    
    
# Population plot times
def find_nearest(array, value):
    i = 0
    iStep=[]
    for ivalue in value:
        array = np.asarray(array)
        idx = (np.abs(array - ivalue)).argmin()
        iStep.append(idx)
        i = i+1
    return iStep

def populations(
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
        