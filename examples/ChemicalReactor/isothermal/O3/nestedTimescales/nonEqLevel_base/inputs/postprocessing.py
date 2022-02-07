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

# Plotting flags
# =============================================================================
flags = {
    'mass': True,
    'levels': False,
    'noneq': False
}

# Plotting quantities
# =============================================================================
# Style -----------------------------------------------------------------------
TS0_lbl = [r'$O_2$', r'$O$']
TS1_lbl = [r'$O_2^{({%s})}$' % (i+1) for i in range(inp_gen.dim['out'][1])]
TS2_lbl = []
for j in range(inp_gen.dim['out'][1]):
    TS2_lbl += [
        r'$O_2^{({%s},{%s})}$' % (j+1, i+1) for i in range(inp_gen.dim['out'][1])
    ]

var_lbl = {
    'TS0': TS0_lbl,
    'TS1': TS1_lbl,
    'TS2': TS2_lbl
}

# Levels plotting
groupings = ['RVE1', 'RVE3', 'RVE9']
time_shots = [10.**(-i) for i in range(12,3,-1)]
timescales = [r'$TS0$',r'$TS1$',r'$TS2$']
colors = ['tab:orange','tab:blue','tab:green']

# Style -----------------------------------------------------------------------
plot_style = {
    'labels':       [r'$t\quad[s]$', r'$Y$'],
    'scales':       ['log', 'linear'],
    'log_plot':     True
}

out_dim = [2, 3, [3,3,3]]
alpha = [
    [r'$\Delta\alpha^{(1)}$'],
    [r'$\Delta\alpha^{(1,1)}$', r'$\Delta\alpha^{(1,2)}$', r'$\Delta\alpha^{(1,3)}$']
]
