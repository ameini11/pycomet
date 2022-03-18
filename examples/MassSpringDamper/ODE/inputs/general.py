# *************************************************************************** #
#                                    PrODE                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining global input parameters and model characteristics 
# --------------------------------------------------------------------------- #


# Model/architecture properties
# =============================================================================
model_name = 'MassSpringDamper_OpNN'

dim = {
    "inp": [2, 1],  # Branch-Trunk input dimensionality
    "out": 2        # Output dimensionality
}

# Execution Flags
# =============================================================================
flags = {
    'load':  0,       # Loading model       0: no, 1: saved model, 2: ckpt
    'train': 1,       # Training model      0: no, 1: yes
    'test':  1,       # Testing model       0: no, 1: yes
    'plot':  2,       # Plotting            0: no, 1: offline, 2: online
    'save':  1        # Saving model        0: no, 1: yes
}

# Additional user paths
# =============================================================================
paths = None

# Physical model
# =============================================================================
physics_informed = True

ode_order = 1
params = [1., 3., 0.5]  # m, k, c

# Input space (limit values)
input_space = {
    'x': [-5., 5.],     # Position [m]
    'v': [-5., 5.],     # Velocity [m/s]
}

# Time limits
time_limit = [0., 1.e1]
