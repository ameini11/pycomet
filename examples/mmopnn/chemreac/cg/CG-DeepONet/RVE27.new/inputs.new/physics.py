# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining input parameters for PhysicalModel in 'physics/physical_model.py'
# --------------------------------------------------------------------------- #


# Physical model: CG model for O3 system
# =============================================================================
# Thermodynamics - HyperNet code & database
specie_thermo = {
    0: {
        'specie': {
            'name': 'O2',
            'model': {
               'levels': 'O3_UMN/O2.csv',
               'lev_mapping': f'O3_UMN/O2/LevelsMap_RVE27.csv'
            }
        },
        'thermo': {
            'model': 'CoupledModes'
        }
    },
    1: {
        'specie': {
            'name': 'O',
            'model': {
               'levels': 'O3_UMN/O.csv',
            }
        },
        'thermo': {
            'model': 'Constant'
        }
    }
}

process = ['adiabatic', 'isochoric']
