# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining input parameters for PhysicalModel in 'physics/physical_model.py'
# --------------------------------------------------------------------------- #


# Physical model: CG model for O3 system
# =============================================================================
# Time limits
tlim = [1.e-10, 1.e-6]

# Thermodynamics - HyperNet code & database
specie_names = {
    "m": "O2",
    "a": "O"
}

bins = (1, 3, 9, 27)
    
specie_thermo = {
    0: {
        'specie': {
            'name': 'O2',
            'model': {
               'levels': 'O3_UMN/O2.csv',
               'lev_mapping': f'O3_UMN/O2/LevelsMap_RVE{bins[-1]}.csv'
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

mol_specie_thermos = [
    {
        "specie": {
            "name": "O2",
            "model": {
               "levels": "O3_UMN/O2.csv",
               "lev_mapping": f"O3_UMN/O2/LevelsMap_RVE{nb_bins}.csv"
            }
        },
        "thermo": {
            "model": "CoupledModes"
        }
    } for nb_bins in bins
]

process = ['adiabatic', 'isochoric']
