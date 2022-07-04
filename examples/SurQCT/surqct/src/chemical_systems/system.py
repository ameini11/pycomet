import numpy as np
import pandas as pd

from surqct.src.utilities import utils
from surqct.src.utilities import const
from surqct import config as cfg


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
        self.DiatData['EInt0']  = self.DiatData['EInt'] - np.amin(self.DiatData['EInt'])

        self.DiatData['EVib'], self.DiatData['ERot'] = self.compute_vibrotenergy()
        self.DiatData['EVib0']  = self.DiatData['EVib'] - np.amin(self.DiatData['EVib'])
        self.DiatData['ERot0']  = self.DiatData['ERot'] - np.amin(self.DiatData['ERot'])

    def compute_partition_fn(self, TTranVec):
        PFData = pd.DataFrame()
        for TTran in TTranVec:
            PFData['q_'+str(int(TTran))] = self.DiatData['g'] * np.exp( - self.DiatData['EInt0'] / (const.UKB*TTran) )
        return PFData

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

        self.Names           = ('Diss', 'Inel', 'Exch_1', 'Exch_2')

        self.Proc            = {k: None for k in self.Names}
        for i, k in enumerate(self.Proc.keys()):
            if self.ProcBool[i]:
                self.Proc[k] = Process(Name=k, Dtb=self.PathToHDF5File, CompInOut=self.CompInOut[k])

class Process():

    def __init__(self, Name, Dtb, CompInOut):

        self.Name       = Name
        self.Dtb        = Dtb
        self.CompInOut  = CompInOut
        self.Rates      = None

    def read_rates(self, TTran, TInt, Dtb=None):

        if Dtb is None:
            Dtb = self.Dtb

        if os.path.exists(Dtb):
            f = h5py.File(Dtb, 'r')
        else:
            utils.raise_value_err(f"File '{Dtb}' not found.")

        TStr        = f'/T_{int(TTran)}_{int(TInt)}/Rates/'
        self.Rates  = f[TStr][self.Name][()]

        f.close()


# 0D Master Equation Simulation
###############################################################################
class MasterEquation():

    def __init__(self, System, MolFracNames=None):

        if MolFracNames is None:
            MolFracNames = [System.Atoms[0].Name, System.Molecules[0].Name]

        self.System     = System
        self.BoxHeader  = ['Time'] + list(MolFracNames) + ['Temp','Density','Pressure','NumberDen','Energy']
        self.MEPath     = cfg.MERuns + f'/{System.NameLong}/'

    def read_box(self, PathToCase):
        BoxData         = pd.read_csv(self.MEPath + PathToCase + '/box.dat', delimiter='  ', skiprows=1, engine='python')
        BoxData.columns = self.BoxHeader
        return BoxData

    def read_popevolution(self, PathToCase, TimeInstants):
        Ni = {}
        for iMol, Mol in enumerate(self.System.Molecules):
            if Mol.Name not in Ni:
                FileName     = self.MEPath + PathToCase + f'/pop_{Mol.Name}.dat'
                Ni_ov_g      = np.loadtxt(FileName, comments='&', delimiter="   ", skiprows=2, usecols=(1))
                Ni[Mol.Name] = Ni_ov_g.reshape((TimeInstants, Mol.NLevels)) * Mol.g.reshape(1,-1)
        return Ni

    def compute_energies(self, Ni):
        E = {'EVib': {}, 'ERot': {}}
        for iMol, Mol in enumerate(self.System.Molecules):
            if Mol.Name not in E['EVib']:
                NiTot = np.sum(Ni[Mol.Name], axis=1)
                for k in E.keys():
                    Ei = Mol.DiatData[k].to_numpy().reshape(1,-1)
                    E[k][Mol.Name] = np.sum(Ni[Mol.Name] * Ei, axis=1) / NiTot
        return E


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
