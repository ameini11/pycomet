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


# Training quantities
# =============================================================================
limit_y = False

# Plotting quantities
# =============================================================================
# Outputs manipulation --------------------------------------------------------
def output_ops(y):
    return tf.concat(y, 1)

# Style -----------------------------------------------------------------------
var_names = [r'$O_2$', r'$O$'] \
    + [r'$O_2^{(1,{%s})}$' % (i+1) for i in range(inp_gen.dim['out'][1])] \
    + [r'$O_2^{(2,{%s})}$' % (i+1) for i in range(inp_gen.dim['out'][2])]

plot_style = {
    'output_ops':   output_ops,
    'var_names':    var_names,
    'labels':       [r'$t\quad[s]$', r'$Y$'],
    'scales':       ['log', 'linear'],
    'log_plot':     True
}
