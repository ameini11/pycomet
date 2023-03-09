# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining postprocessing quantities for testing phase.
# --------------------------------------------------------------------------- #


import numpy as np
from inputs.general import dim


# Plotting quantities
# =============================================================================
loc = list(np.arange(len(dim['out'])))
idx = [list(np.arange(i)) for i in dim['out']]
figures = [
    # *[{
    #     'fig_name':  f'ts{i}',
    #     'indices':   ({1: 0}, {loc[i]: idx[i]}, {loc[i]: idx[i]}),
    #     'labels':    [r'$t$ [s]', r'$Y$'],
    #     'scales':    ['log', 'log']
    # } for i in loc[:-1]],
    {
        'fig_name':  f'ts{-2}',
        'indices':   ({1: 0}, {loc[-2]: idx[-2]}, {loc[-2]: idx[-2]}),
        'labels':    [r'$t$ [s]', r'$Y$'],
        'scales':    ['log', 'log']
    },
    {
        'fig_name':  'temp',
        'indices':   ({1: 0}, {loc[-1]: idx[-1]}, {loc[-1]: idx[-1]}),
        'labels':    [r'$t$ [s]', r'$T$ [K]'],
        'scales':    ['log', 'linear']
    }
]
