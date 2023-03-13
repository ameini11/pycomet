#---------------------------
# Path to data and mixture, reaction, transfer files
PATH     = /home/zanardi/Workspace/AirDatabase/ThermoChemModels/CG/database/O3_UMN
MIXTURE  = 27g
REACTION = RVE27/kD_kExf.dat
TRANSFER = empty

#---------------------------
#Free-strem conditions (pressure, temperature, velocity and mole fractions)
P0 = 100.d0
T0 = 300.d0
COMP0 = O O2_1 O2_2 O2_3 O2_4 O2_5 O2_6 O2_7 O2_8 O2_9 O2_10 O2_11 O2_12 O2_13 O2_14 O2_15 O2_16 O2_17 O2_18 O2_19 O2_20 O2_21 O2_22 O2_23 O2_24 O2_25 O2_26 O2_27
XC0 = 1.d-1 9.d-1 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0 0.d0

#Read initial solution from 'init_sol.dat'
READ_INIT_SOL = T

#Bath temperature
T_BATH = 5000.d0

#Simulation mode
MODE = ADIABATIC

#---------------------------
#Time-grid parameters
TS = 0.d0
TE = 1.d-3
DT_MAX = 1.d-3
DT_MIN = 1.d-12
SF = 1.005d0

#---------------------------
#ODE solver settings
ATOL = 1.d-15
RTOL = 1.d-12
