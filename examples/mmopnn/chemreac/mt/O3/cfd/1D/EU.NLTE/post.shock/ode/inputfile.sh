#---------------------------
# Path to data and mixture, reaction, transfer files
PATH     = /home/zanardi/Workspace/AirDatabase/ThermoChemModels/MT/database
MIXTURE  = oxygen2g
REACTION = oxygen2
TRANSFER = TTv

#---------------------------
#Free-strem conditions (pressure, temperature, velocity and mole fractions)
P_OO  = 1000.d0
T_OO  = 500.d0
U_OO  = 3000.d0
COMP_OO = O O2
XC_OO   = 0.05 0.95

#---------------------------
#Space-grid parameters
XS = 0.
XE = 1.
DX_MAX = 1.d-4
DX_MIN = 1.d-12
SF = 1.015d0

#---------------------------
#ODE solver settings
ATOL = 1.d-15
RTOL = 1.d-9
