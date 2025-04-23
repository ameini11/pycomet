#!/home/zanardi/.conda/envs/sciml/bin/python
# -*- coding: utf-8 -*-

# Import libraries
# =====================================
import sys
sys.path.append("/home/zanardi/Codes/ML/PyCOMET/pycomet/")

from timers import ODESolveTimer
from silx.io.dictdump import h5todict


if (__name__ == '__main__'):

  # Inputs
  # =====================================
  prefix = "/home/zanardi/Codes/ML/PyCOMET/run/noneq/RVE45_adiabatic/datagen/datagen/"
  # Max order
  max_orders = [2, 3, 4, 5]
  rtol_orders = [6, 7, 8, 9]
  # Final times
  t_ends = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
  # Initial solutions
  path_to_init_sols = prefix + "/runs/samples/init_sols.hdf5"
  # Software
  integrators = {
    "lsode": "module load plato/dev-gcc-12.3.0-openblas-0.3.20",
    "sundials": "module load plato/dev-gcc-9.3.0-openblas-0.3.9-sundials-5.8.0"
  }
  # Multi-threading
  nb_workers = 1
  nb_threads = 1

  # Execution time
  # =====================================
  init_sols = h5todict(path_to_init_sols)["train"]
  for (integrator, module_load) in integrators.items():
    if (module_load is not None):
      ode_timer = ODESolveTimer(
        path_to_wd="./",
        integrator=integrator,
        load_plato=module_load,
        nb_workers=nb_workers,
        nb_threads=nb_threads
      )
      for rtol_order in rtol_orders:
        for max_order in max_orders:
          timings = ode_timer(
            t_ends=t_ends,
            init_sols=init_sols,
            rtol_order=rtol_order,
            max_order=max_order
          )
