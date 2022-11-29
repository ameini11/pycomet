#----------------------------------------------------------
#Example of input file for HEGEL - High-fidElity tool for maGnEto-gasdynamics appLications 
#----------------------------------------------------------

SIMULATION_START_TIME   = 0.290528e-5
SIMULATION_START_ITER   = 500
SIMULATION_RESTART_PATH = ../output/
SIMULATION_RESTART_FILE = restart_EU_NLTE_500.dat

SIMULATION_OVWRITE_CONV_FILE = T
SIMULATION_WRITE_AT_RESTART = F

#---------------------------
#Mesh
#--------------------------
MESH_NB_DIM     = 1
MESH_PATH       = /home/zanardi/Workspace/CFD/HEGEL/1D/EU.NLTE/shock/meshes/n701
MESH_NODE_FILE  = nodes.dat
MESH_BOUND_FILE = bound.dat
MESH_BOUND      = INLET OUTLET    #Boundaries

#----------------------------
#Fluid model
#----------------------------

FLUID_MODEL    = EU_NLTE                    #Fluid model
FLUID_NB_DIM   = 1                          #Number of dimensions
FLUID_GEOM     = CART                       #Geometry (CART = Cartesian; CYL = Cylindrical; SPH = Spherical)
FLUID_BC       = SUP_IN SUP_OUT             #Boundary conditions

#Boundary condition data
#Primitive variables: densities [kg/m^3], velocity components [m/s], temperatures [K]
INLET.PRIM = 1.9242855173e-04 7.3122849656e-03 3.0000000000e+03 5.0000000000e+02 5.0000000000e+02

#Post-shock pressure (set ambivalent inlet/outlet treatment to false)
OUTLET.P = 0.609808565110e+05
OUTLET.AMB = F

#----------------------------
#Numerics
#----------------------------

#Settings for Strang splitting
NUM_TIME_STEP = 1.d-5                      #Time-step
NUM_TIME_STEP_BOUND = T
NUM_CFL = 2.d0                            #CFL number
NUM_INTER_CFL = F
NUM_CFL_RATE  = 1000
NUM_CFL_FILE  = ./cfl
# NUM_STOP_COND = RES -3.                  #Stop condition
NUM_STOP_COND = ITER 5000                  #Stop condition
NUM_TIME_INT = STRANG                      #Time-integration: Three-point backward
NUM_STRANG_ODE_SET = BOX_AD
NUM_STRANG_ODE_RTOL = 1.d-6
NUM_STRANG_ODE_ATOL = 1.d-9
NUM_FLUX = VAN_LEER                        #Numerical inviscid flux
NUM_TIME_ACCURATE = T
NUM_REC_PROC = MUSCL                       #Reconstruction procedure
NUM_LIMITER = VAN_ALBADA            #Slope limiter

#----------------------------
#Physico-chemical library 
#----------------------------

LIBRARY_NAME = plato
LIBRARY_PATH = /home/zanardi/Workspace/AirDatabase/ThermoChemModels/MT/database
LIBRARY_MIXTURE = oxygen2g
LIBRARY_REACTION = oxygen2
LIBRARY_TRANSFER = TTv

#----------------------------
#Initial solution
#----------------------------
INIT_SOL_COORD = x
INIT_SOL_INTER = 0.005d0
#Primitive variables: densities [kg/m^3], velocity components [m/s], temperatures [K]
INIT_SOL_PRIM = 1.9242855173e-04 7.3122849656e-03 3.0000000000e+03 5.0000000000e+02 5.0000000000e+02 7.9338040408e-03 5.9100799912e-02 3.3585848539e+02 3.1305074272e+03 3.1305074272e+03

#----------------------------
#Output options
#----------------------------
OUT_RATE = 100
