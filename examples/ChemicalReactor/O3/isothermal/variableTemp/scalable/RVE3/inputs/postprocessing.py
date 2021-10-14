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
plot_style = {
    'output_ops':   output_ops,
    'var_names':    inp_gen.var_names,
    'labels':       [r'$t\quad[s]$', r'$Y$'],
    'scales':       ['log', 'linear'],
    'log_plot':     True
}
