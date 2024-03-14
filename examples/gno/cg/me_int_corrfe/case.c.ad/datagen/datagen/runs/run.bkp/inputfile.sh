#---------------------------
# Path to data and mixture, reaction, transfer files
PATH     = /home/zanardi/Workspace/AirDatabase/ThermoChemModels/CG/database/O3_UMN/MAPEQ205
MIXTURE  = 205g
REACTION = kFull.dat
TRANSFER = empty

#---------------------------
#Initial conditions
P0 = 10000.d0
T0 = 300.d0

#Bath temperature
T_BATH = 10000.d0

#Simulation mode
MODE = ISOTHERMAL

#Read initial solution from 'init_sol.dat'
READ_INIT_SOL = T

#---------------------------
#Time-grid parameters
TS = 0.d0
TE = 1.d0
DT_MAX = 1.d-1
DT_MIN = 1.d-14
SF = 1.02d0

#---------------------------
#ODE solver settings
ATOL = 1.d-18
RTOL = 1.d-15
