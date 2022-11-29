#----------------------------------------------------------
#Example of input file for HEGEL - High-fidElity tool for maGnEto-gasdynamics appLications 
#----------------------------------------------------------

#---------------------------
#Mesh
#--------------------------
MESH_NB_DIM     = 1
MESH_PATH       = /home/zanardi/Workspace/CFD/HEGEL/1D/EU.NLTE/shock/meshes/n201
MESH_NODE_FILE  = nodes.dat
MESH_BOUND_FILE = bound.dat
MESH_BOUND      = INLET OUTLET    #Boundaries

#----------------------------
#Fluid model
#----------------------------

FLUID_MODEL    = EU_NLTE                    #Fluid model
FLUID_NB_DIM   = 1                          #Number of dimensions
FLUID_GEOM     = CART                       #Geometry (CART = Cartesian; CYL = Cylindrical; SPH = Spherical)
FLUID_BC       = SUP_IN FARFIELD             #Boundary conditions
FLUID_SRC_TERM = KINETICS                   #Source term(s)

#Boundary condition data
#Primitive variables: densities [kg/m^3], velocity components [m/s], temperatures [K]
INLET.PRIM = 1.9242855173e-04 7.3122849656e-03 3.0000000000e+03 5.0000000000e+02 5.0000000000e+02

#Post-shock pressure (set ambivalent inlet/outlet treatment to false)
OUTLET.P = 0.609808565110e+05
OUTLET.AMB = F

#----------------------------
#Numerics
#----------------------------

NUM_TIME_STEP = 1.d-8               #Time-step
NUM_TIME_STEP_BOUND = T
NUM_CFL = 2.0                      #CFL number
NUM_INTER_CFL = F
NUM_CFL_RATE  = 1000
NUM_CFL_FILE  = ./cfl
NUM_STOP_COND = RES -10.d0          #Stop condition
# NUM_STOP_COND = ITER 6000         #Stop condition
NUM_FLUX_JAC = POS_NEG_SPLIT        #Numerical flux Jacobian approximation
NUM_TIME_INT = BE                   #Time-integration: Backward-Euler (implicit)
NUM_LHS_JAC = FULL                  #Full matrix formulation for LHS Jacobian
NUM_FLUX = VAN_LEER                 #Numerical inviscid flux
NUM_REC_PROC = MUSCL                #Reconstruction procedure
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
OUT_RATE = 500
