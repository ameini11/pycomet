# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining global input parameters and model characteristics 
# --------------------------------------------------------------------------- #


# Model/architecture properties
# =============================================================================
model_name = 'mt_O3_mmopnn'

dim = {
    "inp": [4, 1],  # Branch-Trunk input dimensionality
    "out": [2, 2]   # Output dimensionality
}

# Execution Flags
# =============================================================================
flags = {
    'load':  1,       # Loading model       0: no, 1: saved model, 2: ckpt
    'train': 1,       # Training model      0: no, 1: yes
    'test':  1,       # Testing model       0: no, 1: yes
    'plot':  2,       # Plotting            0: no, 1: offline, 2: online
    'save':  1        # Saving model        0: no, 1: yes
}