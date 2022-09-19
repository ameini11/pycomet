# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining global input parameters and model characteristics 
# --------------------------------------------------------------------------- #


# Model/architecture properties
# =============================================================================
model_name = 'msd_flexopnn'

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