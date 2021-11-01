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

# Physical quantities
# =============================================================================
# Style -----------------------------------------------------------------------
plot_style = {
    'var_names':    [r'$O_2^{({%s})}$' % (i+1) for i in \
                        range(inp_net.opnn_dim['O2']['out'])] + [r'$O$'],
    'labels':       [r'$t\quad[s]$', r'$Y$'],
    'scales':       ['log', 'linear'],
    'log_plot':     True
}
