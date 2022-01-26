from numpy import double


class const(object):

    Eh_to_eV    = double(27.2113839712790e0)        # Hartree to Electronvolt
    Ue          = double(1.602191e-19)              # ElectronVolt to Joule
    UKb         = double(1.380658e-23)              # Boltzmann Constant
    hp          = double(6.62607015e-34)            # Planck Constant
    Av          = double(6.0221367e23)              # Avogadro Number
    mO          = double(15.9994e-3/Av)             # Mass of atomic oxygen
    URG         = UKb*Av                            # Universal gas constant [J/(mol K)]
