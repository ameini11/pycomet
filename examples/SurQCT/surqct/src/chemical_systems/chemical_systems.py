from surqct.src.chemical_systems.system import System
from surqct.src.chemical_systems.system import MasterEquation
from surqct.src.chemical_systems.system import ThermochemicalProc
from surqct.src.chemical_systems import chemical_systems
from surqct.src.utilities import utils
from surqct import config as cfg


def get_system(NameLong, ProcBool=[1,1,1,0], PathToHDF5File=None):
    if hasattr(chemical_systems, NameLong):
        return getattr(chemical_systems, NameLong)(ProcBool, PathToHDF5File)
    else:
        utils.raise_value_err(f"Chemical system '{NameLong}' not recognized.")


# List of available systems
###############################################################################
# O2 + O System (From UMN, Prof. D. Truhlar)
# -----------------------------------------------------------------------------
def O3_UMN(ProcBool=[1,1,1,0], PathToHDF5File=None):

    NameLong        = 'O3_UMN'
    Name            = 'O3'

    NAtoms          = 3
    NMolecules      = 1
    NComp           = 2
    Syst            = System(NameLong, Name, NAtoms, NMolecules)

    # Atoms
    for Atom in Syst.Atoms:
        Atom.Name   = 'O'
        Atom.Mass   = 15.9994e-3    # [kg/mol]
        Atom.Qe     = 9.
        Atom.Ef     = 246795.       # [J/mol]

    # Molecules
    Syst.Molecules[0].Name             = 'O2'
    Syst.Molecules[0].Mass             = 15.9994e-3 * 2    # [kg/mol]
    Syst.Molecules[0].PathToDiatFile   = cfg.CoarseAIR + '/dtb/Molecules/O2/UMN/FromUMN_Sorted.inp'
    Syst.Molecules[0].DegeneracyFactor = [0.5, 0.5]
    Syst.Molecules[0].Qe               = 3.

    # Thermochemical Processes
    if PathToHDF5File is None:
        PathToHDF5Fld       = cfg.HDF5Dtb + '/Orig/O3_UMN.hdf5'
    print(Syst)
    Syst.ThermochemicalProc = ThermochemicalProc(Syst, ProcBool, PathToHDF5File)

    # Master Equation
    Syst.MasterEquation     = MasterEquation(Syst)

    return Syst


# N2 + N System (Recomputed. PES from NASA Ames, Dr. R. L. Jaffe & Dr. D. Schwenke)
# -----------------------------------------------------------------------------
def N3_NASA(ProcBool=[1,1,1,0], PathToHDF5File=None):

    NameLong        = 'N3_NASA'
    Name            = 'N3'

    NAtoms          = 3
    NMolecules      = 1
    NComp           = 2
    Syst            = System(NameLong, Name, NAtoms, NMolecules)

    # Atoms
    for Atom in Syst.Atoms:
        Atom.Name   = 'N'
        Atom.Mass   = 14.0067e-3
        Atom.Qe     = 12.
        Atom.Ef     = 470818.

    # Molecules
    Syst.Molecules[0].Name             = 'N2'
    Syst.Molecules[0].Mass             = 14.0067e-3 * 2
    Syst.Molecules[0].PathToDiatFile   = cfg.CoarseAIR + '/dtb/Molecules/N2/LeRoy/MyLeroy_FromRobyn.inp'
    Syst.Molecules[0].DegeneracyFactor = [0.5, 0.5]

    # Thermochemical Processes
    if PathToHDF5File is None:
        PathToHDF5Fld       = cfg.HDF5Dtb + '/SemiClassicalApprox/N3_NASA.hdf5'
    Syst.ThermochemicalProc = ThermochemicalProc(Syst, ProcBool, PathToHDF5File)

    # Master Equation
    Syst.MasterEquation     = MasterEquation(Syst)

    return Syst


# NO + N System (From UMN)
# -----------------------------------------------------------------------------
def NON_UMN(ProcBool=[1,1,1,1], PathToHDF5File=None):

    NameLong        = 'NON_UMN'
    Name            = 'NON'

    NAtoms          = 3
    NMolecules      = 2
    NComp           = 4
    Syst            = System(NameLong, Name, NAtoms, NMolecules)

    # Atoms
    for Atom in [Syst.Atoms[0], Syst.Atoms[2]]:
        Atom.Name       = 'N'
        Atom.Mass       = 14.0067e-3
        Atom.Qe         = 12.
        Atom.Ef         = 470818.
    Syst.Atoms[1].Name  = 'O'
    Syst.Atoms[1].Mass  = 15.9994e-3
    Syst.Atoms[1].Qe    = 9.
    Syst.Atoms[1].Ef    = 246795.

    # Molecules
    Syst.Molecules[0].Name             = 'NO'
    Syst.Molecules[0].Mass             = 14.0067e-3 + 15.9994e-3
    Syst.Molecules[0].PathToDiatFile   = cfg.CoarseAIR + '/dtb/Molecules/NO/UMN/Recomputed.inp'
    Syst.Molecules[0].Qe               = 4.

    Syst.Molecules[1].Name             = 'N2'
    Syst.Molecules[1].Mass             = 14.0067e-3 * 2
    Syst.Molecules[1].PathToDiatFile   = cfg.CoarseAIR + '/dtb/Molecules/N2/UMN_ForN2O2/Recomputed.inp'
    Syst.Molecules[1].DegeneracyFactor = [3.0, 6.0]

    # Thermochemical Processes
    if PathToHDF5File is None:
        PathToHDF5Fld       = cfg.HDF5Dtb + '/Orig/NON_UMN.hdf5'
    CompInOut               = {
        'Diss': {
            'In':  {
                'Molecules':    [Syst.Molecules[0]],
                'Atoms':        [Syst.Atoms[0]]
            },
            'Out':  {
                'Molecules':    None,
                'Atoms':        Syst.Atoms
            }
        },
        'Inel': {
            'In':  {
                'Molecules':    [Syst.Molecules[0]],
                'Atoms':        [Syst.Atoms[0]]
            },
            'Out':  {
                'Molecules':    [Syst.Molecules[0]],
                'Atoms':        [Syst.Atoms[0]]
            }
        },
        'Exch_1': {
            'In':  {
                'Molecules':    [Syst.Molecules[0]],
                'Atoms':        [Syst.Atoms[0]]
            },
            'Out':  {
                'Molecules':    [Syst.Molecules[1]],
                'Atoms':        [Syst.Atoms[1]]
            }
        },
        'Exch_2': {
            'In':  {
                'Molecules':    [Syst.Molecules[0]],
                'Atoms':        [Syst.Atoms[0]]
            },
            'Out':  {
                'Molecules':    [Syst.Molecules[0]],
                'Atoms':        [Syst.Atoms[0]]
            }
        }
    }
    Syst.ThermochemicalProc = ThermochemicalProc(Syst, ProcBool, PathToHDF5File, CompInOut)

    # Master Equation
    MolFracNames            = ['N', 'O', 'N2', 'NO']
    Syst.MasterEquation     = MasterEquation(Syst, MolFracNames)

    return Syst
