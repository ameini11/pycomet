import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from surqct.src.utilities import utils
from surqct.src.utilities import const
from surqct import config as cfg

from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve


# 0D Master Equation Simulation
###############################################################################
class MasterEquation():

    def __init__(self, System, CompNames=None):

        if CompNames is None:
            CompNames = [System.Atoms[0].Name, System.Molecules[0].Name]
        CompNames = list(CompNames)

        self.System     = System
        self.BoxHeader  = ['t'] + ['X_' + k for k in CompNames] + ['T','rho','p','nd','E']

    def postprocess_data(self, PathToCase):
        self.MEPath = cfg.MERuns + '/' + PathToCase + '/' + self.System.NameLong
        self.Data   = {}
        for ProcCode in os.scandir(self.MEPath):
            ProcBool            = [int(i) for i in ProcCode.split('_')]
            self.Data[ProcCode] = {}
            for TTran in os.scandir(self.MEPath + '/' + ProcCode):
                # Initialize master equation run
                T    = TTran[1:-1]
                Path = self.MEPath + '/' + ProcCode + '/' + TTran + '/'
                self.Data[ProcCode][T] = {}
                # Read box data resulting from Plato simulation
                self.Data[ProcCode][T].update(
                    self.readbox(Path=Path)
                )
                # Read populations data resulting from Plato simulation
                self.Data[ProcCode][T].update(
                    self.read_pop(Path=Path, Time=self.Data[ProcCode][TTran]['t'])
                )
                # Compute energies
                self.Data[ProcCode][T].update(
                    self.compute_energies(Ni=self.Data[ProcCode][TTran]['Ni'])
                )
                # Compute mass depletion
                if ProcBool[0] > 0:
                    self.Data[ProcCode][T].update(
                        self.compute_mass_depletion(Data=self.Data[ProcCode][TTran])
                    )
                # Read energy depletion
                self.Data[ProcCode][T].update(
                    self.read_energy_depletion(Data=self.Data[ProcCode][TTran])
                )
                # Read taus
                self.Data[ProcCode][T].update(
                    self.read_taus(Data=self.Data[ProcCode][TTran])
                )

    # Reading
    # -------------------------------------------------------------------------
    def read_box(self, Path):
        BoxData         = pd.read_csv(Path + '/box.dat', delimiter='  ', skiprows=1, engine='python')
        BoxData.columns = self.BoxHeader
        BoxData         = BoxData.to_dict('list')
        BoxData         = {k: np.array(v) for k, v in BoxData.items()}
        return BoxData

    def read_pop(self, Path, Time):
        Ni = {}
        for iMol, Mol in enumerate(self.System.Molecules):
            if Mol.Name not in Ni:
                FileName     = Path + f'/pop_{Mol.Name}.dat'
                pop_ov_g     = np.abs(np.loadtxt(FileName, comments='&', skiprows=2, usecols=(1)))
                Ni[Mol.Name] = Ni_ov_g.reshape((len(Time), Mol.NLevels)) * Mol.g.reshape(1,-1)
        return {'Ni': Ni}

    # Computing
    # -------------------------------------------------------------------------
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

    def compute_mass_depletion(self, Data, DepPerc=0.5):
        # Delay = (t_SurQCT - t_QCT)/t_QCT
        DepTime = {}
        Time    = Data['t']
        for iMol, Mol in enumerate(self.System.Molecules):
            if Mol.Name not in DepTime:
                # Molar fraction
                MoleFrac = Data[Mol.Name]
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
        return {'MassDepTime': DepTime}

    # def compute_energy_depletion(self, BoxData, DepPerc=0.5):
    #     raise NotImplementedError

    # def compute_taus(self, BoxData, Energies):
    #     TausP   = {}
    #     EKeys   = ('EInt', 'EVib', 'ERot')
    #     Time    = BoxData['Time'].to_numpy()
    #     for iMol, Mol in enumerate(self.System.Molecules):
    #         if Mol.Name not in TausP:
    #             TausP[Mol.Name] = {}
    #             Press           = BoxData['O'].to_numpy()[-1] * BoxData['Pressure'].to_numpy()[-1] / const.ATM_to_PA
    #             for k in EKeys:
    #                 Ek      = Energies[Mol.Name][k]
    #                 # Energy limit
    #                 EkLim   = (Ek[-1] - Ek[0]) * 0.632 + Ek[0]
    #                 Idx     = np.argmin(np.abs(Ek - EkLim))
    #                 # Curve fitting
    #                 # Tau0    = (Time[Idx] + Time[Idx-1]) / 2.
    #                 Tau0    = Time[Idx]
    #                 EkFit   = CubicSpline(x=Time[1:-1], y=(Ek[1:-1] - EkLim))
    #                 TausP[Mol.Name][k] = fsolve(EkFit, [Tau0]) * Press
    #     return TausP

    # def compute_QSS_instant(self, BoxData, DissGlobal):

    #     Time    = BoxData['Time'].to_numpy()
    #     yy      = DissGlobal + 1e-100

    #     NSteps  = len(Time) - 1
    #     InpPerc = 0.1
    #     ExitFlg = False

    #     ###
    #     it = NSteps
    #     while (np.abs(np.log10(yy[it]) - np.log10(yy[-1])) < 1.e-2) and (it > 0):
    #         it = it - 1
    #     itFinal    = it
    #     TimeFinal  = Time[itFinal]
    #     if itFinal < 1:
    #         ExitFlg = True
        
    #     ###
    #     while (np.abs(np.log10(yy[it]) - np.log10(yy[itFinal])) < 1.e-1) and (it > 0):
    #         it = it - 1;
    #     itEnd    = it
    #     TimeEnd  = Time[itEnd]

    #     ###
    #     while (np.abs(np.log10(yy[it]) - np.log10(yy[it+1])) > 5.e-4) and (it >= 0):
    #         it = it - 1
    #     if (it < 0):
    #         ExitFlg = True
    #         itQSS   = NSteps
    #     else:
    #         itQSS   = it
    #     TimeQSS    = Time[itQSS]

    #     ###
    #     if not ExitFlg:
    #         it = itQSS
    #         while (np.abs(np.log10(yy[it]) - np.log10(yy[itQSS])) < 1.e-1):
    #             it = it + 1
    #         itEnd    = it
    #         TimeEnd  = Time[itEnd]

    #     ###
    #     if not ExitFlg:
    #         it = itQSS
    #         while (np.abs(np.log10(yy[it]) - np.log10(yy[itQSS])) < 1.e-1) and (it > 0):
    #             it = it - 1
    #         if (it <= 0):
    #             ExitFlg = True
    #             itStart = NSteps
    #         else:
    #             itStart = it
    #         TimeStart  = Time[itStart]

    #     ###
    #     if not ExitFlg:
    #         while (np.abs(np.log10(yy[it]) - np.log10(yy[itStart])) < 1.e-1):
    #             it = it - 1
    #         itIni    = it
    #         TimeIni  = Time[itIni]
    #         if (itIni >= itFinal):
    #             ExitFlg = True

    #     ###
    #     if not ExitFlg:

    #         # QSS starting point
    #         QSS['Start']    = {}
    #         yyy_start       = yy[itIni:itFinal] - yy[itQSS]*(1.0 - InpPerc)
    #         yyyFit          = CubicSpline(x=Time[itIni:itFinal], y=yyy_start) # Normilze data

    #         QSS['Start']['Time']    = fsolve(yyyFit, [Time[itStart]])
    #         QSS['Start']['Idx']     = np.argmin(np.abs(Time - QSS['Start']['Time']))

    #         if (QSS['Start']['Idx'] < 0) or (QSS['Start']['Idx'] > NSteps):
    #             ExitFlg = True

    #         # QSS ending point
    #         QSS['End']      = {}
    #         if (yy[itQSS] > yy[-1]):
    #             yyy_end     = yy[itStart:itEnd] - yy[itQSS]*(1.0-InpPerc)
    #         else:
    #             yyy_end     = yy[itStart:itEnd] - yy[itQSS]*(1.0+InpPerc)
    #         yyyFit          = CubicSpline(x=Time[itStart:itEnd], y=yyy_end)

    #         QSS['End']['Time']      = fsolve(yyyFit, [Time[itEnd]])
    #         QSS['End']['Idx']       = np.argmin(np.abs(Time - QSS['End']['Time']))

    #         # In QSS point
    #         QSS['In']       = {}
    #         TimeQSS         = np.power(10., (np.log10(QSS['Start']['Time']) + np.log10(QSS['End']['Time']))/2)

    #         QSS['In']['Idx']        = np.argmin(np.abs(Time - TimeQSS))
    #         QSS['In']['Time']       = Time[QSS['In']['Idx']]

    #     ###
    #     if ExitFlg:
    #         QSS['Start']['Time']    = Time[-1]
    #         QSS['Start']['Idx']     = NSteps

    #         QSS['In']['Time']       = Time[-1]
    #         QSS['In']['Idx']        = NSteps

    #         QSS['End']['Time']      = Time[-1]
    #         QSS['End']['Idx']       = NSteps
            
    #     return QSS


    # def compute_rates_global(self, Ni):

    #     Proc = self.System.ThermochemicalProc.Proc

    #     if (self.System.NAtoms == 3):

    #         DissGlobal = {}
    #         ExchGlobal = {}

    #         for iMol, Mol in enumerate(self.System.Molecules):
    #             if Mol.Name not in DissGlobal:
    #                 NiTot                   = np.sum(Ni[Mol.Name], axis=1)
    #                 DissGlobal[Mol.Name]    = np.sum(Proc['Diss'].Rates.reshape(1,-1) * Ni / NiTot, axis=1)

    #                 # for iStep = 1:Kin.T(Temp.iT).NSteps
    #                 #     Rates.T(Temp.iT).Molecule(iMol).DissGlobal(iStep)           = sum( Rates.T(Temp.iT).Molecule(iMol).Overall(:,1)       .* Kin.T(Temp.iT).Molecule(iMol).DF(iStep,:)' ); 
    #                 #     for iExch = 1:Syst.NProc-2
    #                 #         Rates.T(Temp.iT).Molecule(iMol).ExchGlobal(iStep,iExch) = sum( Rates.T(Temp.iT).Molecule(iMol).Overall(:,2+iExch) .* Kin.T(Temp.iT).Molecule(iMol).DF(iStep,:)' ); 
    #                 #     end
    #                 # end
    #     else:
    #         raise NotImplementedError

    #     return DissGlobal, ExchGlobal


    # def compute_rates_thermal(self, TTran=10000):

    #     Proc = self.System.ThermochemicalProc.Proc

    #     if (self.System.NAtoms == 3):

    #         DissTh = {}
    #         ExchTh = {}

    #         for iMol, Mol in enumerate(self.System.Molecules):
    #             if Mol.Name not in DissTh:
    #                 q                   = Mol.PFData[str(int(TTran))]
    #                 DissTh[Mol.Name]    = np.sum(Proc['Diss'].Rates * q / np.sum(q))

    #                 # Rates.T(Temp.iT).Molecule(iMol).DissTh = 0.0;
    #                 # Rates.T(Temp.iT).Molecule(iMol).ExchTh = zeros(1,NProc-2);
    #                 # for iBin = 1:Syst.Molecule(iMol).EqNStatesIn
    #                 #     Rates.T(Temp.iT).Molecule(iMol).DissTh              = sum( Rates.T(Temp.iT).Molecule(iMol).Overall(:,1)       .* iQTemp(:) ); 
    #                 #     for iExch = 1:NProc-2
    #                 #         Rates.T(Temp.iT).Molecule(iMol).ExchTh(1,iExch) = sum( Rates.T(Temp.iT).Molecule(iMol).Overall(:,2+iExch) .* iQTemp(:) ); 
    #                 #     end
    #                 # end
    #     else:
    #         raise NotImplementedError

    #     return DissTh, ExchTh

    # def compute_rates_QSS(self, QSS, DissGlobal, ExchGlobal):

    #     Proc = self.System.ThermochemicalProc.Proc

    #     if (self.System.NAtoms == 3):

    #         DissQSS = {}
    #         ExchQSS = {}

    #         for iMol, Mol in enumerate(self.System.Molecules):
    #             if Mol.Name not in DissTh:
    #                 DissQSS[Mol.Name]   = DissGlobal[Mol.Name][QSS['In']['Idx']]

    #                 # Rates.T(Temp.iT).Molecule(iMol).DissTh = 0.0;
    #                 # Rates.T(Temp.iT).Molecule(iMol).ExchTh = zeros(1,NProc-2);
    #                 # for iBin = 1:Syst.Molecule(iMol).EqNStatesIn
    #                 #     Rates.T(Temp.iT).Molecule(iMol).DissTh              = sum( Rates.T(Temp.iT).Molecule(iMol).Overall(:,1)       .* iQTemp(:) ); 
    #                 #     for iExch = 1:NProc-2
    #                 #         Rates.T(Temp.iT).Molecule(iMol).ExchTh(1,iExch) = sum( Rates.T(Temp.iT).Molecule(iMol).Overall(:,2+iExch) .* iQTemp(:) ); 
    #                 #     end
    #                 # end
    #     else:
    #         raise NotImplementedError

    #     return DissQSS, ExchQSS
