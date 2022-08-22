import os
import numpy as np
import pandas as pd

from surqct.src.utilities import utils

# Visualization libraries
mpl = utils.import_matplotlib()
from matplotlib import pyplot as plt
from matplotlib import pylab as pl
COLORS = mpl.rcParams['axes.prop_cycle'].by_key()['color']


def get_colorbar(colors, labels):
    ncolors = len(colors)
    cmap = mpl.colors.ListedColormap(colors)
    mappable = mpl.cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(
        mappable,
        orientation='horizontal',
        location='top',
        aspect=50
    )
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(labels)
    colorbar.ax.tick_params(labelsize=20)
    return colorbar


def energies(
    Path,
    System,
    Data,
    TTranVec,
    xlim=[1.e-10, 1.e-3]
):
    for iMol, Mol in enumerate(System.Molecules):
        fig = plt.figure()
        plt.plot([], [], '--k', label='$E_{Rot}$')
        plt.plot([], [],  '-k', label='$E_{Vib}$')
        
        colors = pl.cm.coolwarm(np.linspace(0,1,len(TTranVec)))
        # colors = ['#173F5F', '#20639B', '#3CAEA3', '#F6D55C', '#ED553B']
        # colors = ['#581845', '#900C3F', '#C70039', '#FF5733', '#FFC300']
        
        for iTemp in range(len(TTranVec)):

            Temp = int(TTranVec[iTemp])
            EMed = []
            for k, d in Data.items():
                t    = d[Temp][0]
                ERot = d[Temp][1][Mol.Name]['ERot']
                EVib = d[Temp][1][Mol.Name]['EVib']
                EMed.append((ERot[-1] + EVib[-1]) / 2)
                c = 'k' if k == 'QCT' else colors[iTemp]
                plt.plot(t, ERot, '--', c=c)
                plt.plot(t, EVib,  '-', c=c)

            plt.legend(loc='best')
            plt.xscale('log')
            plt.xlabel('$t$ [s]')
            plt.ylabel('$E$ [eV]')
            plt.xlim(xlim)
        
        # Colorbar
        labels = [str(T)+' K' for T in TTranVec] + ['QCT']
        cbar = get_colorbar(list(colors) + ['k'], labels)

        Path = Path + '/' + Mol.Name
        if not os.path.exists(Path):
            os.makedirs(Path)
        plt.savefig(Path + '/EVib_ERot.pdf')


def taus(
    Path,
    System,
    Data
):
    for iMol, Mol in enumerate(System.Molecules):
        fig = plt.figure()
        plt.scatter([], [], marker='s', c=COLORS[0], label='$p\\tau_{Rot}$')
        plt.scatter([], [], marker='^', c=COLORS[0], label='$p\\tau_{Vib}$')
        for i, k in enumerate(Data.keys()):
            plt.plot([], [], label=k, c=COLORS[i])

        for i, (k, d) in enumerate(Data.items()):
            X = d[Mol.Name]['T']**(-1/3)
            plt.plot(X, d[Mol.Name]['pTauRot'], ls='--', marker='s', c=COLORS[i], lw=2.)
            plt.plot(X, d[Mol.Name]['pTauVib'], ls='--', marker='^', c=COLORS[i], lw=2.)

        plt.legend()
        plt.ticklabel_format(axis='x', style='sci', scilimits=(-3,3))
        plt.yscale('log')
        plt.xlabel('$T^{-1/3}$ [K$^{-1/3}$]')
        plt.ylabel('$p\\tau$ [atm s]')

        Path = Path + '/' + Mol.Name
        if not os.path.exists(Path):
            os.makedirs(Path)
        plt.savefig(Path + '/Taus.pdf')


def mole_fraction(
    Path,
    System,
    Data,
    TTranVec,
    xlim=[1.e-9, 1.e1],
    ylim=[0.,1.]
):
    for iMol, Mol in enumerate(System.Molecules):
        fig = plt.figure()
        colors = pl.cm.coolwarm(np.linspace(0,1,len(TTranVec)))
        for iTemp in range(len(TTranVec)):

            Temp = int(TTranVec[iTemp])
            for k, d in Data.items():
                t = d[Temp][0].reshape(-1)
                X = d[Temp][1][f'X_{Mol.Name}'].reshape(-1)
                c = 'k' if k == 'QCT' else colors[iTemp]
                plt.plot(np.append(t, 1.e3), np.append(X, X[-1]), c=c)
                
            plt.xscale('log')
            plt.xlabel('$t$ [s]')
            plt.ylabel('$X_{\\text{O}_2}$')
            plt.xlim(xlim)
        
        # Colorbar
        labels = [str(T)+' K' for T in TTranVec] + ['QCT']
        cbar = get_colorbar(list(colors) + ['k'], labels)

        Path = Path + '/' + Mol.Name
        if not os.path.exists(Path):
            os.makedirs(Path)
        plt.savefig(Path + '/MoleFrac.pdf')

    
def mole_depletion_delays(
    Path,
    System,
    Data,
    TTranVec,
    xlim=[1.e3, 2.1e4],
    ylim=[None, None]
):
    for iMol, Mol in enumerate(System.Molecules):
        fig = plt.figure()
        Delay = {}
        for iTemp in range(len(TTranVec)):
            Temp = int(TTranVec[iTemp])
            DeplQCT = Data['QCT'][Temp][Mol.Name]
            for i, (k, d) in enumerate(Data.items()):
                if k == 'QCT':
                    continue
                if k not in Delay:
                    Delay[k] = {}
                Depl  = Data[k][Temp][Mol.Name]
                Delay[k][Temp] = -100*(Depl - DeplQCT)/DeplQCT
        
        for i, (k, d) in enumerate(Data.items()):
            if k == 'QCT':
                continue
            x = list(Delay[k].keys())
            y = list(Delay[k].values())
            plt.plot(x, y, ls='--', marker='^', c=COLORS[0], lw=2.)

        plt.hlines(0., xlim[0], xlim[1], colors='grey', lw=0.5)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(3,3))
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('$T$ [K]')
        plt.ylabel('Depletion Delay [\%]')

        Path = Path + '/' + Mol.Name
        if not os.path.exists(Path):
            os.makedirs(Path)
        plt.savefig(Path + '/Delays.pdf')


def KD_ThQSS(
    Path,
    System,
    Data
):
    for iMol, Mol in enumerate(System.Molecules):
        fig = plt.figure()
        plt.scatter([], [], marker='s', c=COLORS[0], label='$K^D_{Th}$')
        plt.scatter([], [], marker='^', c=COLORS[0], label='$K^D_{QSS}$')
        for i, k in enumerate(Data.keys()):
            plt.plot([], [], label=k, c=COLORS[i])

        for i, (k, d) in enumerate(Data.items()):
            X = 10000/d[Mol.Name]['T']
            idx = np.where(X < 3)[0]
            plt.plot(X.iloc[idx], d[Mol.Name]['Diss_Eq'].iloc[idx], ls='--', marker='s', c=COLORS[i], lw=2.)
            plt.plot(X.iloc[idx], d[Mol.Name]['Diss_QSS'].iloc[idx], ls='--', marker='^', c=COLORS[i], lw=2.)
            
        plt.legend()
        plt.yscale('log')
        plt.xlabel('$10000/T$ [K$^{-1}$]')
        plt.ylabel('$K^D$ [cm$^3$/s]')

        Path = Path + '/' + Mol.Name
        if not os.path.exists(Path):
            os.makedirs(Path)
        plt.savefig(Path + '/KGlobal.pdf')

    
# Population plot times
def find_nearest(array, value):
    i = 0
    iStep = []
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
    FigName=None
):
    for iMol, Mol in enumerate(System.Molecules):

        fig, ax = plt.subplots(2, 1, sharex=True)
        ax = np.array(ax).reshape(-1)

        x = Mol.DiatData['EInt'] - Mol.DissE
        for iTemp in range(len(TTranVec)):
            Temp = int(TTranVec[iTemp])
            for i, (k, d) in enumerate(Data.items()):
                t   = d[Temp][0]
                Idx = find_nearest(t, TimeShots[Temp])
                Ni  = d[Temp][1][Mol.Name]['Ni'][Idx] / Mol.g.reshape(-1)
                
                if 'QCT' in k:
                    kwargs = dict(label='QCT', c=COLORS[0], alpha=0.)
                else:
                    kwargs = dict(label=k, c=COLORS[i], alpha=0.5)
                    
                ax[iTemp].scatter([], [], c=kwargs['c'], label=kwargs['label'])
                ax[iTemp].scatter(x, Ni, c=kwargs['c'], s=10, alpha=kwargs['alpha'])
            ax[iTemp].set_yscale('log')

        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('$\epsilon_{Int}$ [eV]', labelpad=30)
        plt.ylabel('$N_i/g_i$ [m$^{-3}$]', labelpad=40)

        lines, labels  = ax[0].get_legend_handles_labels()
        plt.legend(lines, labels, frameon=False, markerscale=2.0, ncol=1, bbox_to_anchor=(1., 1.), handletextpad=0.1)

        Path = Path + '/' + Mol.Name
        if not os.path.exists(Path):
            os.makedirs(Path)
        if FigName is None:
            FigName = 'Pop'
        plt.savefig(Path + '/' + FigName + '.pdf')
        

# def populations(
#     Path,
#     System,
#     Data,
#     TTranVec,
#     TimeShots,
#     subplots_shape=(1,1),
#     figsize_factor=1.5
# ):
#     for iTemp in range(len(TempVec)):
#         Temp = int(TempVec[iTemp])

#         for k in Data.keys():
#             Data[k]['Steps'] = find_nearest(Data[k][Temp]['box'].Time, TimeShots[Temp])

#         fig, ax = plt.subplots(*subplots_shape, sharex=True, figsize=np.array([6.4,4.8])*figsize_factor)
#         ax = np.array(ax).reshape(-1)

#         for iStep in range(len(TimeShots[Temp])):
#             for iMol, Mol in enumerate(System.Molecule):
#                 x = Mol.DiatData['EInt'] - Mol.DissE
#                 for i, k in enumerate(Data.keys()):
#                     y = Data[k][Temp]['pop'][Mol.Name][Data[k]['Steps'][iStep],:]/Mol.DiatData['g']
#                     ax[iStep].scatter(x, y, c=COLORS[i], s=2, label=k, alpha=0.25)
#             ax[iStep].set_yscale('log')
#             ax[iStep].set_title('$t$ = %.1e s' % TimeShots[int(Temp)][iStep])

#         fig.add_subplot(111, frame_on=False)
#         plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
#         plt.xticks([])
#         plt.yticks([])
#         plt.xlabel('$\epsilon_i$ [eV]', labelpad=30)
#         plt.ylabel('$N_i/g_i$ [m$^{-3}$]', labelpad=40)
        
#         lines, labels  = ax[0].get_legend_handles_labels()
#         plt.legend(lines, labels, frameon=False, markerscale=2.0, ncol=1, bbox_to_anchor=(1., 1.), handletextpad=0.1)
#         plt.savefig(Path + f'/Pop_T{Temp}K.pdf')#, bbox_inches="tight")
