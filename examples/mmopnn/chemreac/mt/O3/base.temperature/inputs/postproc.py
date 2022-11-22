# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining postprocessing quantities for testing phase.
# --------------------------------------------------------------------------- #


# Plotting quantities
# =============================================================================
figures = [
    {
        'fig_name':  'mass',
        'indices':   ({1: 0}, {0: [0,1]}, {0: [0,1]}),
        'var_names': [r'$Y_{O_2}$', r'$Y_O$'],
        'labels':    [r'$t$ [s]', r'$Y$'],
        'scales':    ['log', 'linear']
    },
    {
        'fig_name':  'temp',
        'indices':   ({1: 0}, {1: [0,1]}, {1: [0,1]}),
        'var_names': [r'$T_{rt}$', r'$T_{v}$'],
        'labels':    [r'$t$ [s]', r'$T$ [K]'],
        'scales':    ['log', 'linear']
    }
]
