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
import inputs.network as inp_net

# Training quantities
# =============================================================================
limit_y = False

# Plotting quantities
# =============================================================================
# Outputs manipulation --------------------------------------------------------
def output_ops(y):
    y[0] = y[0] * y[1][:,0:1]
    return tf.concat(y, 1)
    
# Style -----------------------------------------------------------------------
plot_style = {
    'output_ops':   output_ops,
    'var_names':    [r'$O_2^{({%s})}$' % (i+1) for i in \
                        range(inp_net.opnn_dim['exc']['out'])] + [r'$O_2$', r'$O$'],
    'labels':       [r'$t\quad[s]$', r'$Y$'],
    'scales':       ['log', 'linear'],
    'log_plot':     True
}
