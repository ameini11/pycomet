# *************************************************************************** #
#                                    PyCOMET                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining postprocessing quantities for testing phase.
# --------------------------------------------------------------------------- #


import tensorflow as tf
import inputs.general as inp_gen


# Plotting quantities
# =============================================================================
# Style -----------------------------------------------------------------------
var_names = [
    [
        r'$\delta^{({%s},{%s})}$' % (3, j+1) for j in range(d)
    ] for i, d in enumerate(inp_gen.dim['out'])
]

plot_style = {
    'var_names':    var_names,
    'labels':       [r'$t\quad[s]$', r'$\delta$'],
    'scales':       ['log', 'linear'],
    'fig_name':     ['NEQ3']
}
