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


def plotting(
    Systems,
    TTranVec,
    Path,
    xlim=[1.e-9, 1.e-4]
):
    energies = Energies(Systems, TTranVec, Path, xlim)
    taus = Taus(Systems, TTranVec, Path)
    mole_fracs = MoleFractions(Systems, TTranVec, Path, xlim)
    mole_depl_delays = MoleDepletionDelays(Systems, TTranVec, Path)
    kd_th_qss = KDThermalQSS(Systems, TTranVec, Path)


# Energies
###############################################################################
class Energies(object):
    
    def __init__(
        self,
        Systems,
        TTranVec,
        Path,
        xlim=[1.e-9, 1.e-4]
    ):
        self.Systems   = Systems
        self.TTranVec  = TTranVec
        self.Path      = Path
        self.xlim      = xlim
        
        self.Molecules = list(self.Systems.values())[0].Molecules
        
        self.Data      = self.extract_data()
        for ProcCode, Data in self.Data.items():
            self.plotting(self.Path + '/' + ProcCode, Data)

    def extract_data(
        self
    ):
        Data = {}
        for Identifier, System in self.Systems.items():
            for ProcCode in System.MasterEquation.Data.keys():
                if ProcCode not in Data:
                    Data[ProcCode] = {}
                Data[ProcCode][Identifier] = {}
                for TTran in self.TTranVec:
                    Time     = System.MasterEquation.Data[ProcCode][str(TTran)]['t']
                    Energies = System.MasterEquation.Data[ProcCode][str(TTran)]['Energies']
                    Data[ProcCode][Identifier][TTran] = (Time, Energies)
        return Data
            
    def plotting(
        self,
        Path,
        Data
    ):
        for iMol, Mol in enumerate(self.Molecules):
            fig = plt.figure()
            plt.plot([], [],'--k', label='$E_{Rot}$')
            plt.plot([], [], '-k', label='$E_{Vib}$')
            for i, k in enumerate(Data.keys()):
                plt.plot([], [], label=k, c=COLORS[i])

            for iTemp in range(len(self.TTranVec)):

                Temp = int(self.TTranVec[iTemp])
                EMed = []
                for i, (k, d) in enumerate(Data.items()):
                    t    = d[Temp][0]
                    ERot = d[Temp][1][Mol.Name]['ERot']
                    EVib = d[Temp][1][Mol.Name]['EVib']
                    EMed.append((ERot[-1] + EVib[-1]) / 2)
                    plt.plot(t, ERot, '--', c=COLORS[i])
                    plt.plot(t, EVib,  '-', c=COLORS[i])

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
                plt.self.xlim(self.xlim)

            Path = Path + '/' + Mol.Name
            if not os.path.exists(Path):
                os.makedirs(Path)
            plt.savefig(Path + '/EVib_ERot.pdf', bbox_inches="tight")


# Taus
###############################################################################
class Taus(object):
    
    def __init__(
        self,
        Systems,
        TTranVec,
        Path
    ):
        self.Systems   = Systems
        self.TTranVec  = TTranVec
        self.Path      = Path
        
        self.Molecules = list(self.Systems.values())[0].Molecules
        
        self.Data      = self.extract_data()
        for ProcCode, Data in self.Data.items():
            self.plotting(self.Path + '/' + ProcCode, Data)

    def extract_data(
        self
    ):
        Data = {}
        for Identifier, System in self.Systems.items():
            for ProcCode in System.MasterEquation.Data.keys():
                if ProcCode.startswith('0'):
                    if ProcCode not in Data:
                        Data[ProcCode] = {}
                    Data[ProcCode][Identifier] = System.MasterEquation.Data[ProcCode]['Taus']
        Data = tf.nest.map_structure(pd.DataFrame.from_records, Data)
        return Data

    def plotting(
        self,
        Path,
        Data
    ):
        for iMol, Mol in enumerate(self.Molecules):
            fig = plt.figure()
            plt.plot([], [], 'ks', label='$\\tau_{Rot}$')
            plt.plot([], [], 'k^', label='$\\tau_{Vib}$')
            for i, k in enumerate(Data.keys()):
                plt.plot([], [], label=k, c=COLORS[i])

            for i, (k, d) in enumerate(Data.items()):
                X = d[Mol.Name]['T']**(-1/3) * 1.e2
                plt.plot(X, d[Mol.Name]['pTauRot'], ls='', marker='s', c=COLORS[i])
                plt.plot(X, d[Mol.Name]['pTauVib'], ls='', marker='^', c=COLORS[i])

            plt.legend()
            plt.yscale('log')
            plt.xlabel('$T^{-1/3}$ [K$^{-1/3}$] $\\times 10^2$')
            plt.ylabel('$p\cdot\\tau$ [atm$\cdot$s]')

            Path = Path + '/' + Mol.Name
            if not os.path.exists(Path):
                os.makedirs(Path)
            plt.savefig(Path + '/Taus.pdf', bbox_inches="tight")


# Mole Fractions
###############################################################################
class MoleFractions(object):
    
    def __init__(
        self,
        Systems,
        TTranVec,
        Path,
        xlim=[1.e-9, 1.e-4]
    ):
        self.Systems   = Systems
        self.TTranVec  = TTranVec
        self.Path      = Path
        self.xlim      = xlim
        
        self.Molecules = list(self.Systems.values())[0].Molecules
        
        self.Data      = self.extract_data()
        for ProcCode, Data in self.Data.items():
            self.plotting(self.Path + '/' + ProcCode, Data)

    def extract_data(
        self
    ):
        Data = {}
        for Identifier, System in Systems.items():
            for ProcCode in System.MasterEquation.Data.keys():
                if not ProcCode.startswith('0'):
                    if ProcCode not in Data:
                        Data[ProcCode] = {}
                    Data[ProcCode][Identifier] = {}
                    for TTran in TTranVec:
                        DSet = System.MasterEquation.Data[ProcCode][str(TTran)]
                        X    = {k: v for k, v in DSet.items() if 'X_' in k}
                        Data[ProcCode][Identifier][TTran] = (DSet['t'], X)
        return Data

    def plotting(
        self,
        Path,
        Data
    ):
        for iMol, Mol in enumerate(self.Molecules):
            fig = plt.figure()
            for i, k in enumerate(Data.keys()):
                plt.plot([], [], label=k, c=COLORS[i])

            for iTemp in range(len(self.TTranVec)):

                Temp = int(self.TTranVec[iTemp])
                for i, (k, d) in enumerate(Data.items()):
                    t = d[Temp][0]
                    X = d[Temp][1][f'X_{Mol.Name}']
                    plt.plot(t, X, c=COLORS[i])

                plt.legend()
                plt.xscale('log')
                plt.xlabel('$t$ [s]')
                plt.ylabel('$X$')
                plt.self.xlim(self.xlim)

            Path = Path + '/' + Mol.Name
            if not os.path.exists(Path):
                os.makedirs(Path)
            plt.savefig(Path + '/MoleFrac.pdf', bbox_inches="tight")


# Mole Depletion Delays
###############################################################################
class MoleDepletionDelays(object):
    
    def __init__(
        self,
        Systems,
        TTranVec,
        Path
    ):
        self.Systems   = Systems
        self.TTranVec  = TTranVec
        self.Path      = Path
        
        self.Molecules = list(self.Systems.values())[0].Molecules
        
        self.Data      = self.extract_data()
        for ProcCode, Data in self.Data.items():
            self.plotting(self.Path + '/' + ProcCode, Data)
            
    def extract_data(
        self
    ):
        Data = {}
        for Identifier, System in Systems.items():
            for ProcCode in System.MasterEquation.Data.keys():
                if not ProcCode.startswith('0'):
                    if ProcCode not in Data:
                        Data[ProcCode] = {}
                    Data[ProcCode][Identifier] = {}
                    for TTran in TTranVec:
                        Data[ProcCode][Identifier][TTran] = System.MasterEquation.Data[ProcCode][str(TTran)]['MassDepTime']
        return Data

    def plotting(
        self,
        Path,
        Data
    ):
        for iMol, Mol in enumerate(self.Molecules):
            fig = plt.figure()
            for i, k in enumerate(Data.keys()):
                if k == 'QCT':
                    continue
                plt.plot([], [], label=k, c=COLORS[i])

            for iTemp in range(len(self.TTranVec)):

                Temp = int(self.TTranVec[iTemp])
                DeplQCT = Data['QCT'][Temp][Mol.Name]
                for i, (k, d) in enumerate(Data.items()):
                    if k == 'QCT':
                        continue
                    Depl  = Data[k][Temp][Mol.Name]
                    Delay = (DeplQCT - Depl)/DeplQCT*100
                    plt.plot(Temp, Delay, ls='', marker='^', c=COLORS[i])

                plt.legend()
                plt.ticklabel_format(axis='x', style='sci', scilimits=(3,3))
                plt.xlabel('$T$ [K]')
                plt.ylabel('% Delay')

            Path = Path + '/' + Mol.Name
            if not os.path.exists(Path):
                os.makedirs(Path)
            plt.savefig(Path + '/Delays.pdf', bbox_inches="tight")


# Thermal & QSS Dissociation Rates
###############################################################################
class KDThermalQSS(object):
    
    def __init__(
        self,
        Systems,
        TTranVec,
        Path
    ):
        self.Systems   = Systems
        self.TTranVec  = TTranVec
        self.Path      = Path
        
        self.Molecules = list(self.Systems.values())[0].Molecules
        
        self.Data      = self.extract_data()
        for ProcCode, Data in self.Data.items():
            self.plotting(self.Path + '/' + ProcCode, Data)
            
    def extract_data(
        self
    ):
        Data = {}
        for Identifier, System in Systems.items():
            for ProcCode in System.MasterEquation.Data.keys():
                if not ProcCode.startswith('0'):
                    if ProcCode not in Data:
                        Data[ProcCode] = {}
                    Data[ProcCode][Identifier] = System.MasterEquation.Data[ProcCode]['KGlobal']
        Data = tf.nest.map_structure(pd.DataFrame.from_records, Data)
        return Data

    def plotting(
        self,
        Path,
        Data
    ):
        for iMol, Mol in enumerate(self.Molecules):
            fig = plt.figure()
            plt.plot([], [], 'ks', label='$k^D_{Th}$')
            plt.plot([], [], 'k^', label='$k^D_{QSS}$')
            for i, k in enumerate(Data.keys()):
                plt.plot([], [], label=k, c=COLORS[i])

            for i, (k, d) in enumerate(Data.items()):
                X = 10000./d[Mol.Name]['T']
                idx = np.where(X < 3)[0]
                plt.plot(X.iloc[idx], d[Mol.Name]['Diss_Eq'].iloc[idx], ls='', marker='s', c=COLORS[i])
                plt.plot(X.iloc[idx], d[Mol.Name]['Diss_QSS'].iloc[idx], ls='', marker='^', c=COLORS[i])

            plt.legend()
            plt.yscale('log')
            plt.xlabel('$10000/T$ [K$^{-1}$]')
            plt.ylabel('$k^D$ [cm$^3$/s]')

            Path = Path + '/' + Mol.Name
            if not os.path.exists(Path):
                os.makedirs(Path)
            plt.savefig(Path + '/KGlobal.pdf', bbox_inches="tight")
