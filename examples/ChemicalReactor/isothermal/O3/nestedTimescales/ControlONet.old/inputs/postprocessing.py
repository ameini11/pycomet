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
# Outputs manipulation --------------------------------------------------------
def output_ops(y):
    return y[:2] + [tf.concat(y[2:], 1)]

# Style -----------------------------------------------------------------------
var_names = [
    [
        r'$\epsilon^{({%s},{%s})}$' % (i+1, j+1) for j in range(d)
    ] for i, d in enumerate([1, 3, 9])
]

plot_style = {
    'output_ops':   output_ops,
    'var_names':    var_names,
    'labels':       [r'$t\quad[s]$', r'$\epsilon$'],
    'scales':       ['log', 'linear'],
    'fig_name':     ['NEQ1', 'NEQ2', 'NEQ3']
}
