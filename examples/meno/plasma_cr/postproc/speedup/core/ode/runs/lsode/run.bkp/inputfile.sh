#---------------------------
# Path to data and mixture, reaction, transfer files
PATH     = /home/zanardi/Workspace/AirDatabase/ThermoChemModels/StS/cr_argon/database
MIXTURE  = kapper
REACTION = kapper_kin_rad
TRANSFER = TTe

#---------------------------
#Initial conditions (pressure temperature and mole fractions)
P0  = 1000.d0
T0  = 300.d0
COMP0 = em Ar
XC0 = 0.d0 1.d0

# Bath temperature
T_BATH = 35000.d0

# Model
MODE = ADIABATIC

#Read initial solution from 'init_sol.dat'
READ_INIT_SOL = T

#---------------------------
#Time-grid parameters
TS = 0.d0
TE = 1.00000d-05
DT_MAX = 1.00000d-05
DT_MIN = 1.d-14
SF = 1.015d0

#---------------------------
#ODE solver settings
ATOL = 0.d0
RTOL = 1.d-9
MAX_ORDER = 5
EVAL_EXEC_TIME = T