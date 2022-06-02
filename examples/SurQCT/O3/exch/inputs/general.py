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
model_name = 'SurQCT_OpNN'

dim = {
    "inp": [5, 5],  # Branch-Trunk input dimensionality
    "out": 1        # Output dimensionality
}

# Execution Flags
# =============================================================================
flags = {
    'load':  2,       # Loading model       0: no, 1: saved model, 2: ckpt
    'train': 1,       # Training model      0: no, 1: yes
    'test':  0,       # Testing model       0: no, 1: yes
    'plot':  0,       # Plotting            0: no, 1: offline, 2: online
    'save':  1        # Saving model        0: no, 1: yes
}

# Physical model
# =============================================================================
physics_informed = False
