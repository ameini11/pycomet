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
    integrator="lsode",
    load_plato="module load plato",
    nb_workers=1,
    nb_threads=1
  ):
    self.integrator = integrator.lower()
    # Paths
    # ---------------------------------
    self.cwd = os.getcwd()
    self.path_to_data = self.cwd + f"/data/{self.integrator}/"
    self.path_to_runs = self.cwd + f"/runs/{self.integrator}/"
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
    t_ends,
    init_sols,
    rtol_order=8,
    max_order=2
  ):
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
    t_end,
    init_sols,
    rtol_order=8,
    max_order=2
  ):
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
    path,
    case_nb,
    init_sol
  ):
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
    os.chdir(self.cwd)
    shutil.rmtree(case_path)
    # Return solution
    return timing
