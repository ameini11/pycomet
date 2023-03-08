# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining postprocessing quantities for testing phase.
# --------------------------------------------------------------------------- #


import numpy as np
from inputs.general import dim


# Plotting quantities
# =============================================================================
idx = [list(np.arange(i)) for i in dim['out']]
figures = [
    {
        'fig_name':  'neq0',
        'indices':   ({1: 0}, {0: idx[0]}, {0: idx[0]}),
        'labels':    [r'$t$ [s]', r'$Y$'],
        'scales':    ['log', 'log']
    },
    {
        'fig_name':  'neq1',
        'indices':   ({1: 0}, {1: idx[1]}, {1: idx[1]}),
        'labels':    [r'$t$ [s]', r'$\delta$'],
        'scales':    ['log', 'log']
    },
    {
        'fig_name':  'neq2',
        'indices':   ({1: 0}, {2: idx[2]}, {2: idx[2]}),
        'labels':    [r'$t$ [s]', r'$\delta$ [K]'],
        'scales':    ['log', 'linear']
    }
]
