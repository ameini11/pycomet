from surqct.src.chemical_systems.component import Atom, Molecule
from surqct.src.chemical_systems.processes import Processes
from surqct.src.chemical_systems.master_equation import MasterEquation
from surqct.src.chemical_systems import systems
from surqct.src.utilities import utils
from surqct import config as cfg


def get_system(
    NameLong,
    ProcBool=None,
    ProcNames=None,
    KinSource='QCT',
    PathToHDF5File=None
):
    if hasattr(systems, NameLong):
        Syst = getattr(systems, NameLong)(
            ProcBool=ProcBool,
            ProcNames=ProcNames,
            KinSource=KinSource,
            PathToHDF5File=PathToHDF5File
        )
        # Initializing diatomic potential data
        for Mol in Syst.Molecules:
            Mol.read_diatdata()
        return Syst
    else:
        utils.raise_value_err(
            f"Chemical system '{NameLong}' not available."
        )


class System():

    def __init__(self, NameLong, Name, NAtoms, NMolecules, KinSource):

        # Identifier
        self.NameLong   = NameLong
        self.Name       = Name

        # Atoms
        self.NAtoms     = NAtoms
        self.Atoms      = [Atom() for iA in range(NAtoms)]

        # Molecules
        self.NMolecules = NMolecules
        self.Molecules  = [Molecule() for iMol in range(NMolecules)]

        # Components
        self.Components = self.Atoms + self.Molecules
        
        # Kinetic source (QCT or SurQCT)
        self.KinSource  = KinSource


# List of available systems
###############################################################################
# O2 + O System (From UMN, Prof. D. Truhlar)
# -----------------------------------------------------------------------------
def O3_UMN(ProcBool=[1,1,1,0], ProcNames=None, KinSource='QCT', PathToHDF5File=None):

    NameLong        = 'O3_UMN'
    Name            = 'O3'

    NAtoms          = 3
    NMolecules      = 1
    NComp           = 2
    Syst            = System(NameLong, Name, NAtoms, NMolecules, KinSource)

    # Atoms
    for iA in range(NAtoms):
        Syst.Atoms[iA]  = get_properties(Name='O', Component=Syst.Atoms[iA])

    # Molecules
    Syst.Molecules[0]   = get_properties(Name='O2', Component=Syst.Molecules[0])
    
    # Thermochemical Processes
    if PathToHDF5File is None:
        PathToHDF5Fld   = cfg.HDF5Dtb + f'/Orig/{NameLong}.hdf5'
    Syst.Processes      = Processes(
        System=Syst,
        ProcBool=ProcBool,
        ProcNames=ProcNames,
        PathToHDF5File=PathToHDF5File
    )
    Syst.Processes.DissCorrFactor = 16./3.

    # Master Equation
    Syst.MasterEquation = MasterEquation(Syst)

    return Syst


# N2 + N System (Recomputed. PES from NASA Ames, Dr. R. L. Jaffe & Dr. D. Schwenke)
# -----------------------------------------------------------------------------
def N3_NASA(ProcBool=[1,1,1,0], ProcNames=None, KinSource='QCT', PathToHDF5File=None):

    NameLong        = 'N3_NASA'
    Name            = 'N3'

    NAtoms          = 3
    NMolecules      = 1
    NComp           = 2
    Syst            = System(NameLong, Name, NAtoms, NMolecules, KinSource)

    # Atoms
    for iA in range(NAtoms):
        Syst.Atoms[iA]  = get_properties(Name='N', Component=Syst.Atoms[iA])

    # Molecules
    Syst.Molecules[0]   = get_properties(Name='N2', Component=Syst.Molecules[0])

    # Thermochemical Processes
    if PathToHDF5File is None:
        PathToHDF5Fld = cfg.HDF5Dtb + f'/SemiClassicalApprox/{NameLong}.hdf5'
    Syst.Processes = Processes(
        System=Syst,
        ProcBool=ProcBool,
        ProcNames=ProcNames,
        PathToHDF5File=PathToHDF5File
    )

    # Master Equation
    Syst.MasterEquation = MasterEquation(Syst)

    return Syst


# NO + N System (From UMN)
# -----------------------------------------------------------------------------
def NON_UMN(ProcBool=[1,1,1,1], ProcNames=None, KinSource='QCT', PathToHDF5File=None):

    NameLong        = 'NON_UMN'
    Name            = 'NON'

    NAtoms          = 3
    NMolecules      = 2
    NComp           = 4
    Syst            = System(NameLong, Name, NAtoms, NMolecules, KinSource)

    # Atoms
    for iA in (0, 2):
        Syst.Atoms[iA]  = get_properties(Name='N', Component=Syst.Atoms[iA])
    Syst.Atoms[1]       = get_properties(Name='O', Component=Syst.Atoms[1])

    # Molecules
    Syst.Molecules[0]   = get_properties(Name='NO', Component=Syst.Molecules[0])
    Syst.Molecules[1]   = get_properties(Name='N2', Component=Syst.Molecules[1])
    Syst.Molecules[1].PathToDiatFile   = cfg.CoarseAIR + '/dtb/Molecules/N2/UMN_ForN2O2/Recomputed.inp'
    Syst.Molecules[1].DegeneracyFactor = [3.0, 6.0]

    # Thermochemical Processes
    if PathToHDF5File is None:
        PathToHDF5Fld = cfg.HDF5Dtb + f'/Orig/{NameLong}.hdf5'
    CompInOut = {
        'Diss': {
            'In':  (Syst.Molecules[0], Syst.Atoms[0]),
            'Out': tuple(Syst.Atoms)
        },
        'Inel': {
            'In':  (self.System.Molecules[0], self.System.Atoms[0]),
            'Out': (self.System.Molecules[0], self.System.Atoms[0])
        },
        'Exch_1': {
            'In':  (self.System.Molecules[0], self.System.Atoms[0]),
            'Out': (self.System.Molecules[1], self.System.Atoms[1])
        },
        'Exch_2': {
            'In':  (self.System.Molecules[0], self.System.Atoms[0]),
            'Out': (self.System.Molecules[0], self.System.Atoms[0])
        }
    }
    Syst.Processes = Processes(
        System=Syst,
        ProcBool=ProcBool,
        ProcNames=ProcNames,
        PathToHDF5File=PathToHDF5File,
        CompInOut=CompInOut
    )

    # Master Equation
    CompNames           = ['N', 'O', 'N2', 'NO']
    Syst.MasterEquation = MasterEquation(Syst, CompNames)

    return Syst
