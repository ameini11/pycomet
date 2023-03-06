# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining postprocessing quantities for testing phase.
# --------------------------------------------------------------------------- #


import numpy as np


# Plotting quantities
# =============================================================================
idx = list(np.arange(3))
figures = [
    {
        'fig_name':  'mass_O',
        'indices':   ({1: 0}, {0: [0,1]}, {0: [0,1]}),
        'labels':    [r'$t$ [s]', r'$Y$'],
        'scales':    ['log', 'log']
    },
    {
        'fig_name':  'mass_O2',
        'indices':   ({1: 0}, {1: idx}, {1: idx}),
        'labels':    [r'$t$ [s]', r'$Y$'],
        'scales':    ['log', 'log']
    },
    {
        'fig_name':  'temp',
        'indices':   ({1: 0}, {2: [0,1]}, {2: [0,1]}),
        'labels':    [r'$t$ [s]', r'$T$ [K]'],
        'scales':    ['log', 'linear']
    }
]
