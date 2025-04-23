"""
ODE Solver Timing Module

This module defines the `ODESolveTimer` class, which benchmarks the runtime
performance of a 0D chemical reactor simulation using an external Fortran-based
ODE library (e.g., PLATO). The module supports automated configuration of
simulation inputs, parallel execution, and extraction of timing statistics.

Key Features:
- Runs simulations with varying end times and initial conditions
- Automatically configures solver settings via shell script templates
- Supports multiprocessing with joblib
- Measures and exports timing data to CSV files for analysis
"""

import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import joblib as jl
import pycomet as pcm

from tqdm import tqdm


class ODESolveTimer(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    path_to_wd: str,
    integrator: str = "lsode",
    load_plato: str = "module load plato",
    nb_workers: int = 1,
    nb_threads: int = 1
  ) -> None:
    """
    Initialize the ODESolveTimer with environment setup and directories.

    :param path_to_wd: Path to working directory.
    :type integrator: str
    :param integrator: Name of the ODE integrator used.
    :type integrator: str
    :param load_plato: Command to load the PLATO environment/module.
    :type load_plato: str
    :param nb_workers: Number of parallel jobs.
    :type nb_workers: int
    :param nb_threads: Number of threads per simulation.
    :type nb_threads: int
    """
    self.integrator = integrator.lower()
    # Paths
    # ---------------------------------
    self.path_to_wd = os.path.abspath(path_to_wd)
    self.path_to_data = self.path_to_wd + f"/data/{self.integrator}/"
    self.path_to_runs = self.path_to_wd + f"/runs/{self.integrator}/"
    os.makedirs(self.path_to_data, exist_ok=True)
    # Environment
    # -----------------------------------
    self.nb_workers = max(1, nb_workers)
    self.nb_threads = max(1, nb_threads)
    self.env = pcm.utils.load_external_module(
      load_cmd=load_plato,
      nb_threads=self.nb_threads
    )

  # Generate data
  # ===================================
  def __call__(
    self,
    t_ends: list,
    init_sols: np.ndarray,
    rtol_order: int = 8,
    max_order: int = 2
  ) -> pd.DataFrame:
    """
    Run simulations over a range of final times and return execution timing
    stats.

    :param t_ends: List of final times to simulate.
    :type t_ends: list
    :param init_sols: Array of initial solution vectors.
    :type init_sols: np.ndarray
    :param rtol_order: Order of relative tolerance (e.g., 8 for 1e-8).
    :type rtol_order: int
    :param max_order: Maximum order of the integrator.
    :type max_order: int
    :return: DataFrame containing timing statistics.
    :rtype: pd.DataFrame
    """
    # Evaluate execution time
    print("Evaluating execution time")
    print(f"> rtol = 1.d-{rtol_order} | order = {max_order}")
    timings = [
      self.eval_exec_time(
        t_end=t_end,
        init_sols=init_sols,
        rtol_order=rtol_order,
        max_order=max_order
      ) for t_end in t_ends
    ]
    timings = np.vstack(timings)
    # Save timings
    timings = pd.DataFrame(
      data=timings,
      columns=["t_end [s]", "mean [s]", "std [s]"]
    )
    timings = timings.sort_values(by=["t_end [s]"])
    timings.to_csv(
      path_or_buf=self.path_to_data \
        + f"/timings_rtol{rtol_order}_ord{max_order}.csv",
      float_format="%.8e",
      index=False
    )
    return timings

  # Evaluate execution time
  # ===================================
  def eval_exec_time(
    self,
    t_end: float,
    init_sols: np.ndarray,
    rtol_order: int = 8,
    max_order: int = 2
  ) -> np.ndarray:
    """
    Evaluate execution time for a specific final time using initial conditions.

    :param t_end: Final time for the simulation.
    :type t_end: float
    :param init_sols: Array of initial solution vectors.
    :type init_sols: np.ndarray
    :param rtol_order: Order of relative tolerance.
    :type rtol_order: int
    :param max_order: Maximum integrator order.
    :type max_order: int
    :return: Array containing [t_end, mean_time, std_time].
    :rtype: np.ndarray
    """
    t_end_f90 = ("%.5e" % t_end).replace("e", "d")
    print(" > t_end = " + t_end_f90 + " s")
    pcm.utils.replace(
      file_in=self.path_to_runs + "/run.bkp/inputfile.sh",
      patterns=(
        "TE",
        "DT_MAX",
        "RTOL",
        "MAX_ORDER"
      ),
      replaces=(
        "TE = " + t_end_f90 + "\n",
        "DT_MAX = " + t_end_f90 + "\n",
        "RTOL = 1.d-" + str(rtol_order) + "\n",
        "MAX_ORDER = " + str(max_order) + "\n"
      ),
      inplace=True,
      use_startswith=True
    )
    if (self.nb_workers > 1):
      timings = jl.Parallel(self.nb_workers)(
        jl.delayed(self._eval_exec_time)(
          path=self.path_to_runs,
          case_nb=i,
          init_sol=init_sol
        ) for (i, init_sol) in enumerate(tqdm(init_sols, ncols=80))
      )
    else:
      timings = []
      for (i, init_sol) in enumerate(tqdm(init_sols, ncols=80)):
        timings.append(
          self._eval_exec_time(
            path=self.path_to_runs,
            case_nb=i,
            init_sol=init_sol
          )
        )
    # Filter solutions
    timings = np.array([t for t in timings if (t is not None)])
    # Compute moments and return data
    mean, std = np.mean(timings), np.std(timings)
    return np.array([t_end, mean, std])

  def _eval_exec_time(
    self,
    path: str,
    case_nb: int,
    init_sol: np.ndarray
  ) -> float:
    """
    Run a single simulation case and extract execution time.

    :param path: Base path to run folders.
    :type path: str
    :param case_nb: Case index (used for naming).
    :type case_nb: int
    :param init_sol: Initial solution vector.
    :type init_sol: np.ndarray
    :return: Execution time in seconds, or None if failed.
    :rtype: float
    """
    # Create run folder
    case_id = str(int(case_nb+1)).zfill(6)
    case_path = shutil.copytree(
      path + "/run.bkp",
      path + f"/run.{case_id}",
      dirs_exist_ok=True
    )
    os.chdir(case_path)
    # Write initial solution for PLATO
    with open("init_sol.dat", "w") as file:
      np.savetxt(file, init_sol.reshape(-1))
    try:
      # Run 0D simulation
      subprocess.run(
        f"box inputfile.sh {self.nb_threads}",
        shell=True,
        env=self.env,
        timeout=1e1,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
      )
      # Read execution time
      timing = np.loadtxt("./output_box/exec_time.dat").squeeze()
    except subprocess.TimeoutExpired:
      timing = None
    # Delete run folder
    os.chdir(self.path_to_wd)
    shutil.rmtree(case_path)
    # Return solution
    return timing
