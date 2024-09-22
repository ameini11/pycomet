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
from silx.io.dictdump import h5todict


if __name__ == '__main__':

  # Inputs
  # =====================================
  prefix = "/home/zanardi/Codes/ML/PyCOMET/run/meno.v3/RVE45/"
  # Number of scenarios
  scenarios = int(1e3)
  # Max order
  max_orders = [2, 3, 4, 5]
  # Final times
  t_ends = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
  # Input space
  inp_space = {
    "X_O": [   0,  0.95],
    "p":   [ 1e3,   1e4],
    "Ti":  [ 1e3,   8e3],
    "T":   [ 8e3, 1.5e4]
  }
  path_to_init_sols = prefix + "/adiabatic/datagen/datagen/runs/samples/init_sols.hdf5"
  # Physics
  path_to_physics = prefix + "/physics/"
  # Software
  integrators = {
    # "lsode": "module load plato/dev-gcc-11.3.0-openmpi-4.1.4",
    "sundials": "module load plato/dev-gcc-9.3.0-openmpi-3.1.3-sundials-5.8.0"
  }
  # Multi-threading
  nb_workers = 1
  nb_threads = 1

  # Execution time
  # =====================================
  init_sols = None
  if (path_to_init_sols is not None):
    init_sols = h5todict(path_to_init_sols)["train"]
  for (integrator, module_load) in integrators.items():
    ode_timer = ODESolveTimer(
      inp_space=inp_space,
      scenarios=scenarios,
      path_to_physics=path_to_physics,
      integrator=integrator,
      load_plato=module_load,
      nb_workers=nb_workers,
      nb_threads=nb_threads
    )
    for max_order in max_orders:
      timings = ode_timer(t_ends, max_order=max_order, init_sols=init_sols)
