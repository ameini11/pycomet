#----------------------------------------------------------
#Example of input file for HEGEL - High-fidElity tool for maGnEto-gasdynamics appLications 
#----------------------------------------------------------

#---------------------------
#Mesh 
#---------------------------

MESH_PATH       = /home/zanardi/Codes/CFD/HEGEL/hegel/input/mesh/ST
MESH_BOUND_FILE = bound_ST.dat
MESH_NODE_FILE  = mesh_ST.dat
MESH_BOUND      = LEFT RIGHT

#----------------------------
#Fluid model
#----------------------------

FLUID_MODEL    = EU_NLTE                    #Fluid model
FLUID_NB_DIM   = 1                          #Number of dimensions
FLUID_GEOM     = CART                       #Geometry (CART = Cartesian; CYL = Cylindrical; SPH = Spherical)
FLUID_BC       = SUP_OUT SUP_OUT            #Boundary conditions
# FLUID_SRC_TERM = KINETICS                   #Source term(s) (no chemistry source term must be provided for Strang splitting)

#----------------------------
#Numerics
#----------------------------

#Settings for dual time-stepping
# NUM_TIME_STEP = 1.d-7              #Global time-step
# NUM_CFL = 0.5d0                    #CFL number
# NUM_STOP_COND = ITER 10000         #Stop condition
# NUM_TIME_INT = FE                  #Time-integration: Forward-Euler (explicit)
# NUM_LHS_JAC = DIAG                         #Block diagonal approximation for LHS Jacobian matrix 
# NUM_FLUX = VAN_LEER                #Numerical inviscid flux
# NUM_TIME_ACCURATE = T              #Time-accurate flag (simulation is run at constant time-step)
# # NUM_DUAL_TIME     = T
# NUM_REC_PROC = MUSCL               #Reconstruction procedure
# NUM_LIMITER = VAN_ALBADA           #Slope limiter

#Settings for Strang splitting 
NUM_TIME_STEP = 1.d-7                      #Time-step
NUM_TIME_STEP_BOUND = .false.
NUM_CFL = 0.5d0                            #CFL number
NUM_STOP_COND = ITER 1000                  #Stop condition
NUM_TIME_INT = STRANG                      #Time-integration: Three-point backward
NUM_STRANG_ODE_SET = BOX_AD
NUM_STRANG_ODE_RTOL = 1.d-6
NUM_STRANG_ODE_ATOL = 1.d-9
NUM_FLUX = VAN_LEER                        #Numerical inviscid flux
NUM_TIME_ACCURATE = T
NUM_REC_PROC = MUSCL                       #Reconstruction procedure
NUM_LIMITER = VAN_ALBADA                   #Slope limiter

#----------------------------
#Physico-chemical library 
#----------------------------

LIBRARY_NAME = plato
LIBRARY_PATH = /home/zanardi/Workspace/AirDatabase/ThermoChemModels/MT/database
LIBRARY_MIXTURE = oxygen2g
LIBRARY_REACTION = oxygen2
LIBRARY_TRANSFER = TTv

#----------------------------
#Surrogate modeling
#----------------------------

SURROGATE_LIBRARY = pycomet
SURROGATE_NAME = KIN_OP
SURROGATE_INPFILE = /home/zanardi/Workspace/CFD/HEGEL/1D/EU.NLTE/sod.O3/surrogate/input/model/inpfile.json
SURROGATE_USED = .true.

#----------------------------
#Initial solution
#----------------------------
INIT_SOL_COORD = x
INIT_SOL_INTER = 1.d0
#Primitive variables:
#>> rho_O2, rho_O, u, Th, Tv (left and right)
INIT_SOL_PRIM  = 2.d-4 2.d-2 0.d0 1.d4 1.d3 1.d-4 1.d-2 0.d0 1.d3 1.d3

#----------------------------
#Output options
#----------------------------  

OUT_RATE = 10
