import numpy as np
import pandas as pd

import surqct.src.Utils.Paths as Paths
import surqct.src.Utils.Const as const


class Atom():

    def __init__(self):

        self.Name       = ''
        self.Mass       = 0.0
        self.NLevels    = 0

class Molecule():

    def __init__(self):

        self.Name       = ''
        self.Mass       = 0.0

    def read_diatdata(self):
        Data = pd.read_csv(self.PathToDiatFile, delimiter=r"\s+", skiprows=15, header=None)

        self.vqn     = np.array(Data.values[:,0],  dtype=np.int16  )
        self.jqn     = np.array(Data.values[:,1],  dtype=np.int16  )
        self.EInt    = np.array(Data.values[:,2],  dtype=np.float64) * const.Ha_to_eV
        self.Egam    = np.array(Data.values[:,3],  dtype=np.float64)
        self.rMin    = np.array(Data.values[:,4],  dtype=np.float64)
        self.rMax    = np.array(Data.values[:,5],  dtype=np.float64)
        self.VMin    = np.array(Data.values[:,6],  dtype=np.float64) * const.Ha_to_eV
        self.VMax    = np.array(Data.values[:,7],  dtype=np.float64) * const.Ha_to_eV
        self.Tau     = np.array(Data.values[:,8],  dtype=np.float64)
        self.rIn     = np.array(Data.values[:,9],  dtype=np.float64)
        self.rOut    = np.array(Data.values[:,10], dtype=np.float64)

        self.NLevels = len(self.EInt)
        self.DissE   = np.amin(self.VMax)

        self.EVib, self.ERot = self.compute_vibrotenergy()
        self.g       = self.compute_degeneracy()

        self.EInt    = self.EInt - np.amin(self.EInt)
        self.EVib    = self.EVib - np.amin(self.EVib)
        self.ERot    = self.ERot - np.amin(self.ERot)

    def compute_degeneracy(self):
        jqn = self.jqn.astype(float)
        g   = 2*jqn + 1
        for i in range(len(jqn)):
            if (jqn[i] % 2 != 0):
                g[i] = self.DegeneracyFactor[0]*g[i]
            else:
                g[i] = self.DegeneracyFactor[1]*g[i]
        return g

    def compute_vibrotenergy(self):
        EVib    = np.zeros((self.NLevels,1))
        ERot    = np.zeros((self.NLevels,1))
        ETemp   = np.zeros((np.amax(self.vqn)+1,1))
        for iLevel in range(self.NLevels):
            if (self.jqn[iLevel] == 0):
                EVib[iLevel]            = self.EInt[iLevel]
                ETemp[self.vqn[iLevel]] = self.EInt[iLevel]
            else:
                EVib[iLevel]            = ETemp[self.vqn[iLevel]]
                ERot[iLevel]            = self.EInt[iLevel] - EVib[iLevel]
        return EVib, ERot



class RateTypes():

    def __init__(self,NbProcesses):

        self.KDiss      = 1
        self.KInel      = 1
        self.KExch      = [1 for iProc in range(2,NbProcesses)]
        
class System():

    def __init__(self, SystemName, NbAtoms, NbMolecules, NbProcesses):

        self.Name           = SystemName

        self.NAtoms         = NbAtoms
        self.Atom           = [Atom() for iA in range(NbAtoms)]

        self.NMolecules     = NbMolecules
        self.Molecule       = [Molecule() for iMol in range(NbMolecules)]

        self.NbProcesses    = NbProcesses
        self.Rates          = RateTypes(NbProcesses)

    def read_populations(self, Path, TimeInstants):

        Ni = {}
        for iMol, Mol in enumerate(self.Molecule):

            FileName     = Path + f'/pop_{Mol.Name}.dat'
            Ni_ov_g      = np.loadtxt(FileName, comments='&', delimiter="   ", skiprows=2, usecols=(1))
            Ni_ov_g      = Ni_ov_g.reshape((TimeInstants, Mol.NLevels))
            assert (Ni_ov_g >= 0.).all()
            Ni[Mol.Name] = Ni_ov_g * Mol.g.reshape(1,-1)
        
        return Ni

    def compute_energies(self, Ni):

        E = {'EInt': {}, 'EVib': {}, 'ERot': {}}
        for iMol, Mol in enumerate(self.Molecule):
            NiTot = np.sum(Ni[Mol.Name], axis=1)
            for k in E.keys():
                Ek = getattr(Mol, k).reshape(1,-1)
                E[k][Mol.Name] = np.sum(Ni[Mol.Name] * Ek, axis=1) / NiTot
        
        return E


def O3_UMN():

    SystemName      = 'O3_UMN'
    NbAtoms         = 1
    NbMolecules     = 1
    NbProcesses     = 3
    Syst            = System(SystemName, NbAtoms, NbMolecules, NbProcesses)

    # Atoms
    Syst.Atom[0].Name = 'O'

    # Molecules
    Syst.Molecule[0].Name             = 'O2'
    Syst.Molecule[0].PathToDiatFile   = Paths.DiatFile['O2']['UMN']
    Syst.Molecule[0].DegeneracyFactor = [0.5, 0.5]
    for Mol in Syst.Molecule:
        Mol.read_diatdata()

    # Master Equation
    Syst.BoxHeader      = ['time',Syst.Atom[0].Name,Syst.Molecule[0].Name,'Temp','Density','Pressure','NbPart','Energy']
    Syst.QCTMEPath      = Paths.WORKSPACE + f'/AirDatabase/Run0D/run/{SystemName}/QCT'
    Syst.SurQCTMEPath   = Paths.WORKSPACE + f'/AirDatabase/Run0D/run/{SystemName}/SurQCT'

    # QCT Rates
    Syst.PathToHDF5Fld  = Paths.HDF5Dtb + '/Orig/O3_UMN.hdf5'
    
    return Syst


def N3_NASA():   

    SystemName      = 'N3_NASA'
    NbAtoms         = 1
    NbMolecules     = 1
    NbProcesses     = 3
    Syst            = System(SystemName, NbAtoms, NbMolecules, NbProcesses)

    # Atoms
    Syst.Atom[0].Name = 'N'

    # Molecules
    Syst.Molecule[0].Name             = 'N2'
    Syst.Molecule[0].PathToDiatFile   = Paths.DiatFile['N2']['LeRoy']
    Syst.Molecule[0].DegeneracyFactor = [0.5, 0.5]
    for Mol in Syst.Molecule:
        Mol.read_diatdata()

    # Master Equation
    Syst.BoxHeader      = ['time',Syst.Atom[0].Name,Syst.Molecule[0].Name,'Temp','Density','Pressure','NbPart','Energy']
    Syst.QCTMEPath      = Paths.WORKSPACE + f'/AirDatabase/Run0D/run/{SystemName}/QCT'
    Syst.SurQCTMEPath   = Paths.WORKSPACE + f'/AirDatabase/Run0D/run/{SystemName}/SurQCT'

    # QCT Rates
    Syst.PathToHDF5Fld  = Paths.WORKSPACE + '/AirDatabase/HDF5/SemiClassicalApprox/N3_NASA.hdf5'

    return Syst


def NON_UMN():

    SystemName      = 'NON_UMN'
    NbAtoms         = 2
    NbMolecules     = 2
    NbProcesses     = 4
    Syst            = System(SystemName, NbAtoms, NbMolecules, NbProcesses)

    # Atoms
    Syst.Atom[0].Name = 'N'
    Syst.Atom[1].Name = 'O'

    # Molecules
    Syst.Molecule[0].Name             = 'NO'
    Syst.Molecule[0].PathToDiatFile   = Paths.DiatFile['NO']['UMN']
    Syst.Molecule[0].DegeneracyFactor = [1.0, 1.0]
    Syst.Molecule[1].Name             = 'N2'
    Syst.Molecule[1].PathToDiatFile   = Paths.DiatFile['N2']['UMN_ForN2O2']
    Syst.Molecule[1].DegeneracyFactor = [3.0, 6.0]
    for Mol in Syst.Molecule:
        Mol.read_diatdata()

    # Master Equation
    Syst.Box_Header     = ['time',Syst.Atom[0].Name,Syst.Atom[1].Name,Syst.Molecule[1].Name,Syst.Molecule[0].Name,'Temp','Density','Pressure','NbPart','Energy']
    Syst.QCTMEPath      = Paths.WORKSPACE + f'/AirDatabase/Run0D/run/{SystemName}/QCT'
    Syst.SurQCTMEPath   = Paths.WORKSPACE + f'/AirDatabase/Run0D/run/{SystemName}/SurQCT'

    # QCT Rates
    Syst.PathToHDF5Fld  = Paths.WORKSPACE + 'AirDatabase/HDF5/Orig/NON_UMN.hdf5'

    return Syst
