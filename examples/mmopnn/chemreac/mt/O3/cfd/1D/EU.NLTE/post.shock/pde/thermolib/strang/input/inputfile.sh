#----------------------------------------------------------
#Example of input file for HEGEL - High-fidElity tool for maGnEto-gasdynamics appLications 
#----------------------------------------------------------

SIMULATION_START_TIME   = 0.d0
SIMULATION_START_ITER   = 0
SIMULATION_RESTART_PATH = ../restart/
SIMULATION_RESTART_FILE = restart_EU_NLTE.dat

SIMULATION_OVWRITE_CONV_FILE = T
SIMULATION_WRITE_AT_RESTART = T

#---------------------------
#Mesh
#--------------------------
MESH_NB_DIM     = 1
MESH_PATH       = /home/zanardi/Workspace/CFD/HEGEL/1D/EU.NLTE/post.shock/meshes/n146
MESH_NODE_FILE  = nodes.dat
MESH_BOUND_FILE = bound.dat
MESH_BOUND      = INLET OUTLET               #Boundaries

#----------------------------
#Fluid model
#----------------------------

FLUID_MODEL     = EU_NLTE                    #Fluid model
FLUID_NB_DIM    = 1                          #Number of dimensions
FLUID_GEOM      = CART                       #Geometry (CART = Cartesian; CYL = Cylindrical; SPH = Spherical)
FLUID_BC        = SUP_IN SUB_OUT             #Boundary conditions

#Boundary condition data
#Free-stream primitive variables: densities [kg/m^3], velocity components [m/s], temperatures [K]
INLET.PRIM = 1.8291481170e-04 6.9507628444e-03 7.0134124165e+02 8.7336232801e+03 3.0000000000e+02

#Post-shock pressure (set ambivalent inlet/outlet treatment to false)
OUTLET.P = 1.8609741146e+04
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
NUM_STOP_COND = RES -3.                  #Stop condition
# NUM_STOP_COND = ITER 100000                  #Stop condition
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
INIT_SOL_INTER = 5.d-4
#Primitive variables: densities [kg/m^3], velocity components [m/s], temperatures [K]
INIT_SOL_PRIM = 5.2332702323e-03 1.1422946048e-02 3.0037688396e+02 3.2719319704e+03 3.2719319704e+03

#----------------------------
#Output options
#----------------------------  
OUT_RATE = 1000
