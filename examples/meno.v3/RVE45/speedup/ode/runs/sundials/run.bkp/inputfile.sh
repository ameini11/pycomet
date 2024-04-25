# ---------------------------
# Path to data and mixture, reaction, transfer files
PATH     = /home/zanardi/Workspace/AirDatabase/ThermoChemModels/CG/database/O3_UMN/RVE45
MIXTURE  = 45g
REACTION = kFull.dat
TRANSFER = empty

# ---------------------------
# Initial conditions (pressure temperature and mole fractions)
P0  = 1000.d0            # Pressure at T0 not T_BATH
T0  = 500.d0
XC0 = 0.05d0 0.95d0        # Put the Boltzmann distribution for the groups at T0
COMP0 = O O2_1

# Bath temperature
T_BATH = 10000.d0

# Model
MODE = ADIABATIC

#Read initial solution from 'init_sol.dat'
READ_INIT_SOL = T

# ---------------------------
# Time-grid parameters
TS = 0.d0
TE = 1.00000d-03
DT_MAX = 1.d0
DT_MIN = 1.d-15
SF = 1.015d0

# ---------------------------
# ODE solver settings
ATOL = 1.d-12
RTOL = 1.d-9
MAX_ORDER = 2
EVAL_EXEC_TIME = T
