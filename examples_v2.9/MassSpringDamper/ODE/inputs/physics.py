# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining input parameters for PhysicalModel in 'physics/physical_model.py'
# --------------------------------------------------------------------------- #


# Physical model
# =============================================================================
params = [1., 3., 0.5]  # m, k, c

# Input space (limit values)
input_space = {
    'x': [-5., 5.],     # Position [m]
    'v': [-5., 5.],     # Velocity [m/s]
}

# Time limits
time_limits = [0., 1.e1]
