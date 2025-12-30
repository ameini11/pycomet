# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining input parameters for PhysicalModel in 'physics/physical_model.py'
# --------------------------------------------------------------------------- #


# Physical model: mass-spring-damper dynamics
# =============================================================================
spaces = {
  'params': {
    'm': [1.,5.],
    'k': [1.,5.],
    'c': [.1,.9]
  },
  'ics': {
    'x': [-5., 5.],     # Position [m]
    'v': [-5., 5.],     # Velocity [m/s]
  },
  'time': [0., 1.e1]       # Time [s]
}