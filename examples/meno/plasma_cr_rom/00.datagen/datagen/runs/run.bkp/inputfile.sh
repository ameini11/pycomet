#---------------------------
# Path to data and mixture, reaction, transfer files
# > Remember to activate the ROM model in mixture file
PATH     = /u/zanardi3/Workspace/cr_argon/database
MIXTURE  = kapper_rom
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
TE = 1.d-2
DT_MAX = 1.d-3
DT_MIN = 1.d-14
SF = 1.015d0

#---------------------------
#ODE solver settings
ATOL = 1.d-6
RTOL = 1.d-9
