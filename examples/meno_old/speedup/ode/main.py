#!/home/zanardi/.conda/envs/sciml/bin/python
# -*- coding: utf-8 -*-

# Import libraries
# =====================================
import sys

libs = {
  "hypernet": "/home/zanardi/Codes/TC/HyperNet/hypernet",
  "pycomet": "/home/zanardi/Codes/ML/PyCOMET/pycomet"
}
for lib in libs.values():
  sys.path.append(lib)

from timer import ODESolveTimer


if __name__ == '__main__':

  # Inputs
  # =====================================
  # Number of scenarios
  scenarios = int(1e1)
  # Final times
  t_ends = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
  # Input space
  inp_space = {
    "X_O": [   0,  0.95],
    "p":   [ 1e3,   1e4],
    "Ti":  [ 1e3,   8e3],
    "T":   [ 8e3, 1.5e4]
  }
  # Physics
  path_to_physics = "../../physics/NRVE45/"
  load_plato = "module load plato/dev-gcc-11.3.0-openmpi-4.1.4"
  # Multi-threading
  nb_workers = 1
  nb_threads = 1

  # Execution time
  # =====================================
  ode_timer = ODESolveTimer(    
    inp_space=inp_space,
    scenarios=scenarios,
    path_to_physics=path_to_physics,
    load_plato=load_plato,
    nb_workers=nb_workers,
    nb_threads=nb_threads
  )
  timings = ode_timer(t_ends)
