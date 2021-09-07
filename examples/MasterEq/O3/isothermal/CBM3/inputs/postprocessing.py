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


# Plotting quantities
# =============================================================================
# Variables -------------------------------------------------------------------
var_names = [ r'$O_2^{({%s})}$' % (i+1) for i in range(inp_net.dim["out"][0]) ]
var_names.append(r'$O$')

# Axis ------------------------------------------------------------------------
labels = [r'$t\quad[s]$', r'$Y$']         # x, y labels
scales = ['log', 'linear']                # x, y scales
log_plot = True
limit_y = False

# Axis ------------------------------------------------------------------------
def output_ops(y):
    return tf.concat(y, 1)