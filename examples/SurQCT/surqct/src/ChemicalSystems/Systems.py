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
        # Extracting all the rot-vib levels
        xMat           = pd.read_csv(self.PathToDiatFile, delim_whitespace=True, skiprows=15, header=None)
        xMat.columns   = ['vqn','jqn','EInt','egam','rMin','rMax','VMin','VMax','Tau','ri','ro']    

        # Shifting Energies so that Zero is the Min of Diatomic Potential at J=0
        xMat['EInt']   = ( xMat['EInt'].to_numpy() - np.amin(xMat['VMax'].to_numpy()) ) * const.Ha_to_eV
        xMat['VMax']   = ( xMat['VMax'].to_numpy() - np.amin(xMat['VMax'].to_numpy()) ) * const.Ha_to_eV
        xMat['VMin']   = ( xMat['VMin'].to_numpy() - np.amin(xMat['VMax'].to_numpy()) ) * const.Ha_to_eV

        xMat['EVib'], xMat['ERot'] = self.compute_vibrotenergy(xMat['EInt'].to_numpy(), xMat['vqn'].to_numpy(int), xMat['jqn'].to_numpy(int))
        xMat['g']      = self.compute_degeneracy(xMat['jqn'].to_numpy(int))

        self.DiatData = xMat
        self.NLevels  = len(self.DiatData)

    def compute_degeneracy(self, jqn):
        g = 2*jqn + 1
        for i in range(len(jqn)):
            if (jqn[i] % 2 != 0):
                g[i] = self.DegeneracyFactor[0]*g[i]
            else:
                g[i] = self.DegeneracyFactor[1]*g[i]
        return g

    def compute_vibrotenergy(self, EInt, vqn, jqn):
        NLevels = len(EInt)
        EVib    = np.zeros((NLevels,1))
        ERot    = np.zeros((NLevels,1))
        ETemp   = np.zeros((np.amax(vqn)+1,1))
        for iLevel in range(NLevels):
            if (jqn[iLevel] == 0):
                EVib[iLevel]       = EInt[iLevel]
                ETemp[vqn[iLevel]] = EInt[iLevel]
            else:
                EVib[iLevel] = ETemp[vqn[iLevel]]
                ERot[iLevel] = EInt[iLevel] - EVib[iLevel]
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
            Pop          = pd.read_csv(FileName,sep="   ",skiprows=2,header=None,index_col=False,comment='&', engine='python')
            Pop.columns  = ['E','Pop']

            Ni_ov_g      = np.reshape(Pop.Pop.to_numpy(), (TimeInstants+1, Mol.NLevels))
            Ni[Mol.Name] = Ni_ov_g * Mol.DiatData.g.to_numpy().reshape(1,-1)
        
        return Ni

    def compute_energies(self, Ni):

        EInt, EVib, ERot    = [{}]*3
        for iMol, Mol in enumerate(self.Molecule):
            NiTot           = np.sum(Ni[Mol.Name], axis=1)
            EInt[Mol.Name]  = np.sum(Ni[Mol.Name] * Mol.DiatData.EInt.to_numpy().reshape(1,-1), axis=1) / NiTot
            EVib[Mol.Name]  = np.sum(Ni[Mol.Name] * Mol.DiatData.EVib.to_numpy().reshape(1,-1), axis=1) / NiTot
            ERot[Mol.Name]  = np.sum(Ni[Mol.Name] * Mol.DiatData.ERot.to_numpy().reshape(1,-1), axis=1) / NiTot
        
        return EInt, EVib, ERot


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
    Syst.PathToHDF5Fld  = Paths.WORKSPACE + '/AirDatabase/HDF5/Orig/O3_UMN.hdf5'
    
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
