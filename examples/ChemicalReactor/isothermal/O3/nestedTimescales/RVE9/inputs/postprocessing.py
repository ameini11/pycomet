# *************************************************************************** #
#                                    PrODE                                    #
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
    return tf.concat([y[0][:,1:2], y[-1]], 1)

# Style -----------------------------------------------------------------------
var_names = [r'$O$'] \
    + [r'$O_2^{(2,{%s})}$' % (i+1) for i in range(inp_gen.dim['out'][2])]

plot_style = {
    'output_ops':   output_ops,
    'var_names':    var_names,
    'labels':       [r'$t\quad[s]$', r'$Y$'],
    'scales':       ['log', 'linear'],
    'log_scale':     True
}
