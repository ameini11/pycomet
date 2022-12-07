#---------------------------
# Path to data and mixture, reaction, transfer files
PATH     = /home/zanardi/Workspace/AirDatabase/ThermoChemModels/CG/database/O3_UMN
MIXTURE  = 27g
REACTION = RVE27/kD_kExf.dat
TRANSFER = empty

#---------------------------
#Free-strem conditions (pressure, temperature, velocity and mole fractions)
P_OO  = 1000.d0
T_OO  = 500.d0
U_OO  = 3000.d0
COMP_OO = O O2_1 O2_2 O2_3 O2_4 O2_5 O2_6 O2_7 O2_8 O2_9 O2_10 O2_11 O2_12 O2_13 O2_14 O2_15 O2_16 O2_17 O2_18 O2_19 O2_20 O2_21 O2_22 O2_23 O2_24 O2_25 O2_26 O2_27
READ_INIT_COMP = T

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
