import sys

from surqct import config as cfg


def get_component_prop(
    Name,
    Component
):
    if hasattr(sys.modules[__name__], Name):
        return getattr(sys.modules[__name__], Name)(Component)
    else:
        utils.raise_value_err(
            f"Chemical component '{Name}' not available."
        )


# Molecules
###############################################################################
def O2(Molecule):

    Molecule.Name               = 'O2'

    Molecule.ElementNames       = 'O'
    Molecule.ElementQnts        = '2'

    Molecule.MolarMass          = 15.9994e-3 * 2
    Molecule.FormationE         = 0.
    Molecule.LinFactor          = 1
    Molecule.SymmFactor         = 2.
    Molecule.ThetaRot           = 2.08
    Molecule.ThetaVib           = 2273.6

    Molecule.PathToDiatFile     = cfg.paths.CoarseAIR + 'dtb/Molecules/O2/UMN/FromUMN_Sorted.inp'

    Molecule.ElecDeg            = 3.
    Molecule.DegeneracyFactor   = [0.5, 0.5]

    return Molecule


def N2(Molecule):

    Molecule.Name               = 'N2'

    Molecule.ElementNames       = 'N'
    Molecule.ElementQnts        = '2'

    Molecule.MolarMass          = 14.0067e-3 * 2
    Molecule.FormationE         = 0.
    Molecule.LinFactor          = 1
    Molecule.SymmFactor         = 2.
    Molecule.ThetaRot           = 2.88
    Molecule.ThetaVib           = 3392.7

    Molecule.PathToDiatFile     = cfg.paths.CoarseAIR + 'dtb/Molecules/N2/LeRoy/MyLeroy_FromRobyn.inp'

    Molecule.ElecDeg            = 1.
    Molecule.DegeneracyFactor   = [0.5, 0.5]

    return Molecule


def NO(Molecule):

    Molecule.Name               = 'NO'

    Molecule.ElementNames       = 'N O'
    Molecule.ElementQnts        = '1 1'

    Molecule.MolarMass          = 14.0067e-3 + 15.9994e-3
    Molecule.FormationE         = 90773.
    Molecule.LinFactor          = 1
    Molecule.SymmFactor         = 1.0
    Molecule.ThetaRot           = 2.45
    Molecule.ThetaVib           = 2740.0

    Molecule.PathToDiatFile     = cfg.paths.CoarseAIR + 'dtb/Molecules/NO/UMN/Recomputed.inp'

    Molecule.ElecDeg            = 4.
    Molecule.DegeneracyFactor   = [1., 1.]

    return Molecule


# Atoms
###############################################################################
def O(Atom):

    Atom.Name               = 'O'

    Atom.ElementNames       = 'O'
    Atom.ElementQnts        = '1'

    Atom.MolarMass          = 15.9994e-3
    Atom.FormationE         = 246795.
    Atom.IonPot             = 13.618054
    Atom.Polar              = 0.802
    Atom.NuclearCharge      = 8

    Atom.LJDiam             = 2.75
    Atom.LJEps              = 80.

    Atom.ElecDeg            = 9.

    return Atom


def N(Atom):

    Atom.Name               = 'N'

    Atom.ElementNames       = 'N'
    Atom.ElementQnts        = '1'

    Atom.MolarMass          = 14.0067e-3
    Atom.FormationE         = 470818.
    Atom.IonPot             = 14.53413
    Atom.Polar              = 1.1
    Atom.NuclearCharge      = 7

    Atom.LJDiam             = 3.298
    Atom.LJEps              = 71.4

    Atom.ElecDeg            = 12.

    return Atom


def C(Atom):

    Atom.Name               = 'C'

    Atom.ElementNames       = 'C'
    Atom.ElementQnts        = '1'

    Atom.MolarMass          = 12.011e-3
    Atom.FormationE         = 711194.
    Atom.IonPot             = 11.2603
    Atom.Polar              = 1.76
    Atom.NuclearCharge      = 6

    Atom.LJDiam             = 3.298
    Atom.LJEps              = 71.4

    Atom.ElecDeg            = 1.

    return Atom
