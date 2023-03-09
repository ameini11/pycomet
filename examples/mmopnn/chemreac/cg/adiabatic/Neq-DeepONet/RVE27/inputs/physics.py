# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining input parameters for PhysicalModel in 'physics/physical_model.py'
# --------------------------------------------------------------------------- #


# Physical model: CG model for O3 system
# =============================================================================
# Time limits
tlim = [1.e-10, 1.e-6]

# Thermodynamics - HyperNet code & database
bins = (1, 3, 9, 27)

groups_graph = {
    0: {
        0: {
            0: [ 0, 1, 2],
            1: [ 3, 4, 5],
            2: [ 6, 7, 8]
        },
        1: {
            3: [ 9,10,11],
            4: [12,13,14],
            5: [15,16,17]
        },
        2: {
            6: [18,19,20],
            7: [21,22,23],
            8: [24,25,26]
        }
    }
}
    
# Thermodynamics - HyperNet code & database
specie_names = {
    "m": "O2",
    "a": "O"
}

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
