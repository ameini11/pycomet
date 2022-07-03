import numpy as np
import pandas as pd

import surqct.src.Utils.Paths as Paths
import surqct.src.Utils.Const as const


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



# O2 + O System (From UMN, Prof. D. Truhlar)
def O3_UMN(ProcBool=[1,1,1,0], PathToHFHDF5File=None):

    NameLong      = 'O3_UMN'
    Name          = 'O3'

    NAtoms        = 3
    NMolecules    = 1
    NComp         = 2
    Syst          = System(NameLong, Name, NAtoms, NMolecules, ProcBool, PathToHFHDF5File)

    # Atoms
    for Atom in Syst.Atom:
        Atom.Name  = 'O'
        Atom.Size  = 200
        Atom.Mass  = 15.9994e-3
        Atom.Qe    = 9.
        Atom.Ef    = 9.

    # Molecules
    Syst.Molecule[0].Name             = 'O2'
    Syst.Molecule[0].Mass             = 15.9994e-3 * 2
    Syst.Molecule[0].PathToDiatFile   = cfg.CoarseAIR + '/dtb/Molecules/O2/UMN/FromUMN_Sorted.inp'
    Syst.Molecule[0].DegeneracyFactor = [0.5, 0.5]
    Syst.Molecule[0].Qe               = 3.

    # Thermochemical Processes

    # Master Equation
    Syst.BoxHeader      = ['time',Syst.Atom[0].Name,Syst.Molecule[0].Name,'Temp','Density','Pressure','NbPart','Energy']
    Syst.QCTMEPath      = Paths.WORKSPACE + f'/AirDatabase/Run0D/run/{SystemName}/QCT'
    Syst.SurQCTMEPath   = Paths.WORKSPACE + f'/AirDatabase/Run0D/run/{SystemName}/SurQCT'

    # QCT Rates
    Syst.PathToHDF5Fld  = Paths.HDF5Dtb + '/Orig/O3_UMN.hdf5'



    Species={'in': ['O2','O'], 'out': ['O2','O']}
