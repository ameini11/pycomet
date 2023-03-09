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
        'fig_name':  f'neq{i}',
        'indices':   ({1: 0}, {i: idx[i]}, {i: idx[i]}),
        'labels':    [r'$t$ [s]', r'$\delta$'],
        'scales':    ['log', 'linear']
    } for i in range(len(idx))
]
