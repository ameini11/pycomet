# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining input parameters for PhysicalModel in 'physics/physical_model.py'
# --------------------------------------------------------------------------- #


# Physical model: Park 2T model for O3 system
# =============================================================================
# Input space (limit values)
inp_space = {
    'XO2': [   0.,      1.],
    'Th':  [ 200.,  20000.],
    'Tv':  [ 200.,  20000.],
    'p':   [1000., 100000.]
}
        
# Thermodynamics - HyperNet code & database
specie_thermo = {
    0: {
        'specie': {
            'name': 'O2',
            'model': {
               'levels': 'O2g.csv'
            }
        },
        'thermo': {
            'model': 'RigidRotorHarmonicOscillator',
            'Tr_index': 0,
            'Tv_index': 1
        }
    },
    1: {
        'specie': {
            'name': 'O',
            'model': {
               'levels': 'Og.csv'
            }
        },
        'thermo': {
            'model': 'Constant'
        }
    }
}

process = ['adiabatic', 'isochoric']

use_norm_energy = True