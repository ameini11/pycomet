# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining input parameters for PhysicalModel in 'physics/physical_model.py'
# --------------------------------------------------------------------------- #


# Physical model
# =============================================================================
params = {
    'm': 1.,
    'k': 3.,
    'c': .5
}

# Input space (limit values)
inp_space = {
    'x': [-5., 5.],     # Position [m]
    'v': [-5., 5.],     # Velocity [m/s]
}

# Time limits
time_limits = [0., 1.e1]
