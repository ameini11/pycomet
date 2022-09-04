# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining input parameters for PhysicalModel in 'physics/physical_model.py'
# --------------------------------------------------------------------------- #


# Physical model: mass-spring-damper dynamics
# =============================================================================
params = {
    'm': 1.,
    'k': 3.,
    'c': .5
}

# Initial condition
init_cond = {
    'x': -5.,       # Position [m]
    'v':  5.        # Velocity [m/s]
}

# Time limits
time_limits = [0., 1.e1]