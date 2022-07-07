import os
import h5py
import numpy as np
import pandas as pd

from surqct.src.utilities import utils
from surqct.src.utilities import const
from surqct import config as cfg

from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve


# Chemical Components
###############################################################################
class Atom():

    def __init__(self):

        self.Name       = ''
        self.Mass       = 0.
        self.NLevels    = 1

        self.Qe         = 1.
        self.EInt       = 0.

        self.Ef         = 0.


class Molecule():

    def __init__(self):

        self.Name               = ''
        self.Mass               = 0.
        self.NLevels            = 1
        self.PathToDiatFile     = None

        self.DegeneracyFactor   = [1., 1.]
        self.Qe                 = 1.

        self.Ef                 = 0.

    def read_diatdata(self):

        self.DiatData           = pd.read_csv(self.PathToDiatFile, delimiter=r"\s+", skiprows=15, header=None)
        self.DiatData.columns   = ['vqn','jqn','EInt','EGam','rMin','rMax','VMin','VMax','Tau','ri','ro']
        self.DiatData.sort_values(by=['EInt'])

        self.NLevels            = len(self.DiatData)
        self.DissE              = np.amin(self.DiatData['VMax']) * const.HA_to_EV
        self.DiatData['g']      = self.compute_degeneracy()

        for k in ('EInt','VMin','VMax'):
            self.DiatData[k]    = self.DiatData[k] * const.HA_to_EV
        self.DiatData['ECB']    = self.DiatData['VMax'] - self.DiatData['EInt']
        self.DiatData['EVib'], self.DiatData['ERot'] = self.compute_vibrotenergy()

        for k in ('EInt','EVib','ERot'):
            self.DiatData[k+'0']  = self.DiatData[k] - np.amin(self.DiatData[k])

    def compute_partition_fn(self, TTranVec):
        self.PFData = pd.DataFrame()
        for TTran in TTranVec:
            self.PFData[str(int(TTran))] = self.DiatData['g'] * np.exp( - self.DiatData['EInt0'] * const.EV_to_J / (const.UKB*TTran) )
        return self.PFData

    def compute_degeneracy(self):
        jqn = self.DiatData['jqn'].to_numpy().astype(float)
        g   = 2*jqn + 1
        for i in range(len(jqn)):
            if (jqn[i] % 2 != 0):
                g[i] = self.DegeneracyFactor[0]*g[i]
            else:
                g[i] = self.DegeneracyFactor[1]*g[i]
        return self.Qe * g

    def compute_vibrotenergy(self):
        jqn     = self.DiatData['jqn'].to_numpy().astype(int)
        vqn     = self.DiatData['vqn'].to_numpy().astype(int)
        EInt    = self.DiatData['EInt'].to_numpy()
        EVib    = np.zeros(self.NLevels)
        ERot    = np.zeros(self.NLevels)
        ETemp   = np.zeros(np.amax(vqn)+1)
        for iLevel in range(self.NLevels):
            if (jqn[iLevel] == 0):
                ETemp[vqn[iLevel]] = EInt[iLevel]
            EVib[iLevel] = ETemp[vqn[iLevel]]
            ERot[iLevel] = EInt[iLevel] - EVib[iLevel]
        return EVib, ERot


# Thermochemical Processes
###############################################################################
class ThermochemicalProc():

    def __init__(self, System, ProcBool, PathToHDF5File=None, CompInOut=None):

        self.System          = System
        self.ProcBool        = ProcBool
        self.PathToHDF5File  = PathToHDF5File
        self.CompInOut       = CompInOut

        if self.CompInOut is None:
            self.CompInOut   = {
                'Diss': {
                    'In':  {
                        'Molecules':    [self.System.Molecules[0]],
                        'Atoms':        [self.System.Atoms[0]]
                    },
                    'Out':  {
                        'Molecules':    None,
                        'Atoms':        self.System.Atoms
                    }
                },
                'Inel': {
                    'In':  {
                        'Molecules':    [self.System.Molecules[0]],
                        'Atoms':        [self.System.Atoms[0]]
                    },
                    'Out':  {
                        'Molecules':    [self.System.Molecules[0]],
                        'Atoms':        [self.System.Atoms[0]]
                    }
                },
                'Exch_1': {
                    'In':  {
                        'Molecules':    [self.System.Molecules[0]],
                        'Atoms':        [self.System.Atoms[0]]
                    },
                    'Out':  {
                        'Molecules':    [self.System.Molecules[0]],
                        'Atoms':        [self.System.Atoms[0]]
                    }
                },
                'Exch_2': {
                    'In':  {
                        'Molecules':    [self.System.Molecules[0]],
                        'Atoms':        [self.System.Atoms[0]]
                    },
                    'Out':  {
                        'Molecules':    [self.System.Molecules[0]],
                        'Atoms':        [self.System.Atoms[0]]
                    }
                }
            }

        self.ProcNames       = np.array(['Diss', 'Inel', 'Exch_1', 'Exch_2'])

    def init_processes(self):
        self.Proc            = {k: None for k in self.ProcNames}
        for i, k in enumerate(self.Proc.keys()):
            if self.ProcBool[i]:
                self.Proc[k] = Process(Name=k, Dtb=self.PathToHDF5File, CompInOut=self.CompInOut[k])

class Process():

    def __init__(self, Name, Dtb, CompInOut):

        self.Name       = Name
        self.Dtb        = Dtb
        self.CompInOut  = CompInOut

    def read_rates(self, TTran, TInt, Dtb=None):

        if Dtb is None:
            Dtb = self.Dtb

        if os.path.exists(Dtb):
            f = h5py.File(Dtb, 'r')
        else:
            utils.raise_value_err(f"File '{Dtb}' not found.")

        TStr    = f'/T_{int(TTran)}_{int(TInt)}/Rates/'
        Rates   = f[TStr][self.Name][()]

        if self.Name == 'Diss':
            Rates = Rates[:,0:1]

        f.close()

        self.Rates = Rates


# 0D Master Equation Simulation
###############################################################################
class MasterEquation():

    def __init__(self, System, MolFracNames=None):

        if MolFracNames is None:
            MolFracNames = [System.Atoms[0].Name, System.Molecules[0].Name]

        self.System     = System
        self.BoxHeader  = ['Time'] + list(MolFracNames) + ['Temp','Density','Pressure','NumberDen','Energy']
        self.MEPath     = cfg.MERuns + f'/run/'

    def read_box(self, PathToCase):
        BoxData         = pd.read_csv(self.MEPath + PathToCase + '/box.dat', delimiter='  ', skiprows=1, engine='python')
        BoxData.columns = self.BoxHeader
        return BoxData

    def read_popevolution(self, PathToCase, TimeInstants):
        Ni = {}
        for iMol, Mol in enumerate(self.System.Molecules):
            if Mol.Name not in Ni:
                FileName     = self.MEPath + PathToCase + f'/pop_{Mol.Name}.dat'
                Ni_ov_g      = np.abs(np.loadtxt(FileName, comments='&', skiprows=2, usecols=(1)))
                Ni[Mol.Name] = Ni_ov_g.reshape((TimeInstants, Mol.NLevels)) * Mol.DiatData['g'].to_numpy().reshape(1,-1)
        return Ni

    def compute_energies(self, Ni):
        E       = {}
        EKeys   = ('EInt', 'EVib', 'ERot')
        for iMol, Mol in enumerate(self.System.Molecules):
            if Mol.Name not in E:
                E[Mol.Name] = {}
                NiTot       = np.sum(Ni[Mol.Name], axis=1)
                for k in EKeys:
                    Ei              = Mol.DiatData[k+'0'].to_numpy().reshape(1,-1)
                    E[Mol.Name][k]  = np.sum(Ni[Mol.Name] * Ei, axis=1) / NiTot
        return E

    def compute_mass_depletion(self, BoxData, DepPerc=0.5):
        # Delay = (t_SurQCT - t_QCT)/t_QCT
        DepTime = {}
        Time    = BoxData['Time'].to_numpy()
        for iMol, Mol in enumerate(self.System.Molecules):
            if Mol.Name not in DepTime:
                # Molar fraction
                MoleFrac = BoxData[Mol.Name].to_numpy()
                # Percentage molar fraction
                MolePerc = np.abs((MoleFrac - MoleFrac[0])/(MoleFrac[0] - MoleFrac[-1]))
                TimeIdx  = np.argmin(np.abs(MolePerc - DepPerc))
                # Deltas
                DeltaT   = Time[TimeIdx] - Time[TimeIdx-1]
                DeltaM   = np.abs(MoleFrac[TimeIdx] - MoleFrac[TimeIdx-1])
                # Depletion time
                TempM    = DepPerc * np.abs(MoleFrac[0] - MoleFrac[-1])
                TempM    = TempM - np.abs(MoleFrac[0] - MoleFrac[TimeIdx-1])
                DepTime[Mol.Name] = Time[TimeIdx-1] + TempM * DeltaT / DeltaM
        return DepTime

    def compute_energy_depletion(self, BoxData, DepPerc=0.5):
        raise NotImplementedError

    def compute_taus(self, BoxData, Energies):
        TausP   = {}
        EKeys   = ('EInt', 'EVib', 'ERot')
        Time    = BoxData['Time'].to_numpy()
        for iMol, Mol in enumerate(self.System.Molecules):
            if Mol.Name not in TausP:
                TausP[Mol.Name] = {}
                Press           = BoxData['O'].to_numpy()[-1] * BoxData['Pressure'].to_numpy()[-1] / const.ATM_to_PA
                for k in EKeys:
                    Ek      = Energies[Mol.Name][k]
                    # Energy limit
                    EkLim   = (Ek[-1] - Ek[0]) * 0.632 + Ek[0]
                    Idx     = np.argmin(np.abs(Ek - EkLim))
                    # Curve fitting
                    # Tau0    = (Time[Idx] + Time[Idx-1]) / 2.
                    Tau0    = Time[Idx]
                    EkFit   = CubicSpline(x=Time[1:-1], y=(Ek[1:-1] - EkLim))
                    TausP[Mol.Name][k] = fsolve(EkFit, [Tau0]) * Press
        return TausP

    def compute_QSS_instant(self, BoxData, DissGlobal):

        Time    = BoxData['Time'].to_numpy()
        yy      = DissGlobal + 1e-100

        NSteps  = len(Time) - 1
        InpPerc = 0.1
        ExitFlg = False

        ###
        it = NSteps
        while (np.abs(np.log10(yy[it]) - np.log10(yy[-1])) < 1.e-2) and (it > 0):
            it = it - 1
        itFinal    = it
        TimeFinal  = Time[itFinal]
        if itFinal < 1:
            ExitFlg = True
        
        ###
        while (np.abs(np.log10(yy[it]) - np.log10(yy[itFinal])) < 1.e-1) and (it > 0):
            it = it - 1;
        itEnd    = it
        TimeEnd  = Time[itEnd]

        ###
        while (np.abs(np.log10(yy[it]) - np.log10(yy[it+1])) > 5.e-4) and (it >= 0):
            it = it - 1
        if (it < 0):
            ExitFlg = True
            itQSS   = NSteps
        else:
            itQSS   = it
        TimeQSS    = Time[itQSS]

        ###
        if not ExitFlg:
            it = itQSS
            while (np.abs(np.log10(yy[it]) - np.log10(yy[itQSS])) < 1.e-1):
                it = it + 1
            itEnd    = it
            TimeEnd  = Time[itEnd]

        ###
        if not ExitFlg:
            it = itQSS
            while (np.abs(np.log10(yy[it]) - np.log10(yy[itQSS])) < 1.e-1) and (it > 0):
                it = it - 1
            if (it <= 0):
                ExitFlg = True
                itStart = NSteps
            else:
                itStart = it
            TimeStart  = Time[itStart]

        ###
        if not ExitFlg:
            while (np.abs(np.log10(yy[it]) - np.log10(yy[itStart])) < 1.e-1):
                it = it - 1
            itIni    = it
            TimeIni  = Time[itIni]
            if (itIni >= itFinal):
                ExitFlg = True

        ###
        if not ExitFlg:

            # QSS starting point
            QSS['Start']    = {}
            yyy_start       = yy[itIni:itFinal] - yy[itQSS]*(1.0 - InpPerc)
            yyyFit          = CubicSpline(x=Time[itIni:itFinal], y=yyy_start) # Normilze data

            QSS['Start']['Time']    = fsolve(yyyFit, [Time[itStart]])
            QSS['Start']['Idx']     = np.argmin(np.abs(Time - QSS['Start']['Time']))

            if (QSS['Start']['Idx'] < 0) or (QSS['Start']['Idx'] > NSteps):
                ExitFlg = True

            # QSS ending point
            QSS['End']      = {}
            if (yy[itQSS] > yy[-1]):
                yyy_end     = yy[itStart:itEnd] - yy[itQSS]*(1.0-InpPerc)
            else:
                yyy_end     = yy[itStart:itEnd] - yy[itQSS]*(1.0+InpPerc)
            yyyFit          = CubicSpline(x=Time[itStart:itEnd], y=yyy_end)

            QSS['End']['Time']      = fsolve(yyyFit, [Time[itEnd]])
            QSS['End']['Idx']       = np.argmin(np.abs(Time - QSS['End']['Time']))

            # In QSS point
            QSS['In']       = {}
            TimeQSS         = np.power(10., (np.log10(QSS['Start']['Time']) + np.log10(QSS['End']['Time']))/2)

            QSS['In']['Idx']        = np.argmin(np.abs(Time - TimeQSS))
            QSS['In']['Time']       = Time[QSS['In']['Idx']]

        ###
        if ExitFlg:
            QSS['Start']['Time']    = Time[-1]
            QSS['Start']['Idx']     = NSteps

            QSS['In']['Time']       = Time[-1]
            QSS['In']['Idx']        = NSteps

            QSS['End']['Time']      = Time[-1]
            QSS['End']['Idx']       = NSteps
            
        return QSS


    def compute_rates_global(self, Ni):

        Proc = self.System.ThermochemicalProc.Proc

        if (self.System.NAtoms == 3):

            DissGlobal = {}
            ExchGlobal = {}

            for iMol, Mol in enumerate(self.System.Molecules):
                if Mol.Name not in DissGlobal:
                    NiTot                   = np.sum(Ni[Mol.Name], axis=1)
                    DissGlobal[Mol.Name]    = np.sum(Proc['Diss'].Rates.reshape(1,-1) * Ni / NiTot, axis=1)

                    # for iStep = 1:Kin.T(Temp.iT).NSteps
                    #     Rates.T(Temp.iT).Molecule(iMol).DissGlobal(iStep)           = sum( Rates.T(Temp.iT).Molecule(iMol).Overall(:,1)       .* Kin.T(Temp.iT).Molecule(iMol).DF(iStep,:)' ); 
                    #     for iExch = 1:Syst.NProc-2
                    #         Rates.T(Temp.iT).Molecule(iMol).ExchGlobal(iStep,iExch) = sum( Rates.T(Temp.iT).Molecule(iMol).Overall(:,2+iExch) .* Kin.T(Temp.iT).Molecule(iMol).DF(iStep,:)' ); 
                    #     end
                    # end
        else:
            raise NotImplementedError

        return DissGlobal, ExchGlobal


    def compute_rates_thermal(self, TTran=10000):

        Proc = self.System.ThermochemicalProc.Proc

        if (self.System.NAtoms == 3):

            DissTh = {}
            ExchTh = {}

            for iMol, Mol in enumerate(self.System.Molecules):
                if Mol.Name not in DissTh:
                    q                   = Mol.PFData[str(int(TTran))]
                    DissTh[Mol.Name]    = np.sum(Proc['Diss'].Rates * q / np.sum(q))

                    # Rates.T(Temp.iT).Molecule(iMol).DissTh = 0.0;
                    # Rates.T(Temp.iT).Molecule(iMol).ExchTh = zeros(1,NProc-2);
                    # for iBin = 1:Syst.Molecule(iMol).EqNStatesIn
                    #     Rates.T(Temp.iT).Molecule(iMol).DissTh              = sum( Rates.T(Temp.iT).Molecule(iMol).Overall(:,1)       .* iQTemp(:) ); 
                    #     for iExch = 1:NProc-2
                    #         Rates.T(Temp.iT).Molecule(iMol).ExchTh(1,iExch) = sum( Rates.T(Temp.iT).Molecule(iMol).Overall(:,2+iExch) .* iQTemp(:) ); 
                    #     end
                    # end
        else:
            raise NotImplementedError

        return DissTh, ExchTh

    def compute_rates_QSS(self, QSS, DissGlobal, ExchGlobal):

        Proc = self.System.ThermochemicalProc.Proc

        if (self.System.NAtoms == 3):

            DissQSS = {}
            ExchQSS = {}

            for iMol, Mol in enumerate(self.System.Molecules):
                if Mol.Name not in DissTh:
                    DissQSS[Mol.Name]   = DissGlobal[Mol.Name][QSS['In']['Idx']]

                    # Rates.T(Temp.iT).Molecule(iMol).DissTh = 0.0;
                    # Rates.T(Temp.iT).Molecule(iMol).ExchTh = zeros(1,NProc-2);
                    # for iBin = 1:Syst.Molecule(iMol).EqNStatesIn
                    #     Rates.T(Temp.iT).Molecule(iMol).DissTh              = sum( Rates.T(Temp.iT).Molecule(iMol).Overall(:,1)       .* iQTemp(:) ); 
                    #     for iExch = 1:NProc-2
                    #         Rates.T(Temp.iT).Molecule(iMol).ExchTh(1,iExch) = sum( Rates.T(Temp.iT).Molecule(iMol).Overall(:,2+iExch) .* iQTemp(:) ); 
                    #     end
                    # end
        else:
            raise NotImplementedError

        return DissQSS, ExchQSS



# Chemical System
###############################################################################
class System():

    def __init__(self, NameLong, Name, NAtoms, NMolecules):

        # Identifier
        self.NameLong           = NameLong
        self.Name               = Name

        # Atoms
        self.NAtoms             = NAtoms
        self.Atoms              = [Atom() for iA in range(NAtoms)]

        # Molecules
        self.NMolecules         = NMolecules
        self.Molecules          = [Molecule() for iMol in range(NMolecules)]
