import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from silx.io.dictdump import dicttoh5, h5todict
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

        self.ProcNames  = np.array(['Diss', 'Inel', 'Exch_1', 'Exch_2'])

    def postprocess_data(self, CaseSuffix=''):
        self.MEPath = f'{cfg.MERuns}/run/{self.System.KinSource}/{self.System.NameLong}/{CaseSuffix}/'
        self.Data   = {}
        for ProcCode in os.scandir(self.MEPath):
            ProcCode            = ProcCode.name
            if any(ProcCode.startswith(k) for k in ('0','1','2')):
                ProcBool            = [int(i) for i in ProcCode.split('_')]
                self.Data[ProcCode] = {}
                utils.print_submain(f"Processes Code '{ProcCode}'")
                for TTran in os.scandir(self.MEPath + '/' + ProcCode):
                    TTran = TTran.name
                    if TTran.startswith('T') and TTran.endswith('K'):
                        T     = TTran[1:-1]
                        utils.print_submain(f'Temperature T = {T} K', delimiter=' '*4)
                        # Initialize master equation run
                        Path  = self.MEPath + '/' + ProcCode + '/' + TTran + '/'
                        self.Data[ProcCode][T] = {}
                        # Read box data resulting from Plato simulation
                        self.Data[ProcCode][T].update(
                            self.read_box(Path=Path)
                        )
                        # Read populations data resulting from Plato simulation
                        self.Data[ProcCode][T].update(
                            self.read_pop(Path=Path, Time=self.Data[ProcCode][T]['t'])
                        )
                        # Compute energies
                        self.Data[ProcCode][T].update(
                            self.compute_energies(Ni=self.Data[ProcCode][T]['Ni'])
                        )
                        # Compute mass depletion
                        if ProcBool[0] > 0:
                            self.Data[ProcCode][T].update(
                                self.compute_mass_depletion(Data=self.Data[ProcCode][T])
                            )
                # From MCoarseAir ---------------------------------------------
                # Read energy depletion
                self.Data[ProcCode].update(
                    self.read_energy_depletion(ProcCode=ProcCode)
                )
                # Read taus
                self.Data[ProcCode].update(
                    self.read_taus(ProcCode=ProcCode)
                )
                # Read QSS timing
                if ProcBool[0] > 0:
                    self.Data[ProcCode].update(
                        self.read_QSS(ProcCode=ProcCode)
                    )
                # Read global rates
                self.Data[ProcCode].update(
                    self.read_globalrates(ProcCode=ProcCode, ProcBool=ProcBool)
                )
                # Read thermal rates
                self.Data[ProcCode].update(
                    self.read_thermalrates(ProcCode=ProcCode, ProcBool=ProcBool)
                )

    def save_data(self):
        file = self.MEPath+'/MEData.hdf5'
        if os.path.exists(file):
            os.remove(file)
        dicttoh5(
            treedict=self.Data,
            h5file=file,
            overwrite_data=True,
            create_dataset_args={'compression': "gzip"}
        )
        
    def read_data(self, CaseSuffix=''):
        self.MEPath = f'{cfg.MERuns}/run/{self.System.KinSource}/{self.System.NameLong}/{CaseSuffix}/'
        self.Data   = h5todict(self.MEPath+'/MEData.hdf5')

    # Reading
    # -------------------------------------------------------------------------
    def read_box(self, Path):
        BoxData         = pd.read_csv(Path + '/output_box/box.dat', delimiter='  ', skiprows=1, engine='python')
        BoxData.columns = self.BoxHeader
        BoxData         = BoxData.to_dict('list')
        BoxData         = {k: np.array(v) for k, v in BoxData.items()}
        return BoxData

    def read_pop(self, Path, Time):
        Ni = {}
        for iMol, Mol in enumerate(self.System.Molecules):
            if Mol.Name not in Ni:
                FileName     = Path + f'/output_box/pop_{Mol.Name}.dat'
                Ni_ov_g      = np.abs(np.loadtxt(FileName, comments='&', skiprows=2, usecols=(1)))
                Ni[Mol.Name] = Ni_ov_g.reshape((len(Time), Mol.NLevels)) * Mol.g.reshape(1,-1)
        return {'Ni': Ni}

    def read_energy_depletion(self, ProcCode):
        EDCoeff = {}
        Columns = ['T', 'IntEq', 'RotEq', 'VibEq', 'IntQSS', 'RotQSS', 'VibQSS']
        for iMol, Mol in enumerate(self.System.Molecules):
            if Mol.Name not in EDCoeff:
                FileName          = f'{self.MEPath}/{ProcCode}/Data/EDCoeffs_{Mol.Name}_{ProcCode}.csv'
                EDCoeff[Mol.Name] = pd.read_csv(FileName, delimiter=',', skiprows=2, header=None, engine='python')
                EDCoeff[Mol.Name].columns = Columns
                EDCoeff[Mol.Name] = EDCoeff[Mol.Name].to_records(index=False)
        return {'EnergyDepCoeff': EDCoeff}

    def read_taus(self, ProcCode):
        Taus    = {}
        Columns = ['T', 'p', 'pTauInt', 'pTauRot', 'pTauVib']
        for iMol, Mol in enumerate(self.System.Molecules):
            if Mol.Name not in Taus:
                FileName       = f'{self.MEPath}/{ProcCode}/Data/Taus_{Mol.Name}_{ProcCode}.csv'
                Taus[Mol.Name] = pd.read_csv(FileName, delimiter=',', skiprows=1, header=None, engine='python')
                Taus[Mol.Name].columns = Columns
                Taus[Mol.Name] = Taus[Mol.Name].to_records(index=False)
        return {'Taus': Taus}

    def read_QSS(self, ProcCode):
        Columns     = ['T', 'Start', 'Mid', 'End']
        FileName    = f'{self.MEPath}/{ProcCode}/Data/QSS_{ProcCode}.csv'
        QSS         = pd.read_csv(FileName, delimiter=',', skiprows=2, header=None, engine='python')
        QSS.columns = Columns
        QSS         = QSS.to_records(index=False)
        return {'QSS': QSS}

    def read_globalrates(self, ProcCode, ProcBool):
        KGlobal = {}
        Columns = ['T']
        for j in ('Eq', 'QSS'):
            Columns = Columns + [f'Diss_{j}']
            for i, k in enumerate(self.ProcNames):
                if i > 1 and ProcBool[i]:
                    Columns = Columns + [f'{k}_{j}']
        for iMol, Mol in enumerate(self.System.Molecules):
            if Mol.Name not in KGlobal:
                FileName          = f'{self.MEPath}/{ProcCode}/Data/KGlobal_{Mol.Name}_{ProcCode}.csv'
                KGlobal[Mol.Name] = pd.read_csv(FileName, delimiter=',', skiprows=2, header=None, engine='python')
                KGlobal[Mol.Name].columns = Columns
                KGlobal[Mol.Name] = KGlobal[Mol.Name].to_records(index=False)
        return {'KGlobal': KGlobal}

    def read_thermalrates(self, ProcCode, ProcBool):
        KTh     = {}
        Columns = ['T', 'Diss']
        for i, k in enumerate(self.ProcNames):
            if i > 1 and ProcBool[i]:
                Columns = Columns + [k]
        for iMol, Mol in enumerate(self.System.Molecules):
            if Mol.Name not in KTh:
                FileName      = f'{self.MEPath}/{ProcCode}/Data/KTh_{Mol.Name}_{ProcCode}.csv'
                KTh[Mol.Name] = pd.read_csv(FileName, delimiter=',', skiprows=2, header=None, engine='python')
                KTh[Mol.Name].columns = Columns
                KTh[Mol.Name] = KTh[Mol.Name].to_records(index=False)
        return {'KTh': KTh}

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
        return {'Energies': E}

    def compute_mass_depletion(self, Data, DepPerc=0.5):
        # Delay = (t_SurQCT - t_QCT)/t_QCT
        DepTime = {}
        Time    = Data['t']
        for iMol, Mol in enumerate(self.System.Molecules):
            if Mol.Name not in DepTime:
                # Molar fraction
                MoleFrac = Data[f'X_{Mol.Name}']
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
