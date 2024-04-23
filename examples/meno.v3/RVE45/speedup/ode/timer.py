import os
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
import joblib as jl
import pycomet as pcm
import tensorflow as tf

from pyDOE import lhs
from tqdm import tqdm

from hypernet.apps import utils as hyputils


class ODESolveTimer(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    inp_space={},
    scenarios=100,
    path_to_physics="./physics",
    load_plato="module load plato",
    nb_workers=1,
    nb_threads=1
  ):
    # Paths
    # ---------------------------------
    self.cwd = os.getcwd()
    self.path_to_data = self.cwd + "/data/"
    self.path_to_runs = self.cwd + "/runs/"
    self.path_to_physics = path_to_physics
    os.makedirs(self.path_to_data, exist_ok=True)
    # Data quantities
    # ---------------------------------
    self.scenarios = scenarios
    # Input space
    self.inp_space = np.vstack(list(inp_space.values())).T
    self.inp_space_lbl = list(inp_space.keys())
    # Initializing thermophysical models
    # -----------------------------------
    with open(self.path_to_physics + "/inputs.json") as file:
      phyinp = json.load(file)
    # Species thermo
    self.sp_th = hyputils._get_specie_thermos(phyinp)
    # Mixture
    self.mix = hyputils._get_mixture(phyinp, self.sp_th)
    # Species
    # -----------------------------------
    self.species = []
    for st in self.sp_th.values():
      if (st.specie.nb_comp > 1):
        self.species.extend([
          st.specie.name + f"_{i+1}" for i in range(st.specie.nb_comp)
        ])
        self.molecule = st.specie.name
      else:
        self.species.append(st.specie.name)
        self.atom = st.specie.name
    # Labeling data
    # -----------------------------------
    # Primitive variables
    self.prim_lbl = [f"rho_{s}" for s in self.species] + ["T"]
    # Environment
    # -----------------------------------
    self.nb_workers = nb_workers
    self.nb_threads = nb_threads
    self.env = pcm.utils.load_external_module(
      load_plato,
      nb_threads=self.nb_threads
    )

  # Generate data
  # ===================================
  def __call__(
    self,
    t_ends,
    init_sols=None,
    read_init_sols=False
  ):
    # Sample initial conditions
    if (init_sols is None):
      if read_init_sols:
        print("Reading sampled input space")
        init_sols = pd.read_csv(self.path_to_data+"/init_sols.csv").to_numpy()
      else:
        print("Sampling input space")
        init_sols = self.sample_inp_space()
    # Evaluate execution time
    print("Evaluating execution time")
    timings = [self.eval_exec_time(t_end, init_sols) for t_end in t_ends]
    timings = np.vstack(timings)
    # Save timings
    timings = pd.DataFrame(
      data=timings,
      columns=["t_end [s]", "mean [s]", "std [s]"]
    )
    timings = timings.sort_values(by=["t_end [s]"])
    timings.to_csv(
      path_or_buf=self.path_to_data+"/timings.csv",
      float_format="%.8e",
      index=False
    )
    return timings

  # Sample initial conditions
  # ===================================
  def sample_inp_space(self):
    # Contruct design matrix (DM)
    d, s = self.inp_space.shape[1], self.scenarios
    x = lhs(d, s)
    # Sorting w.r.t. norm
    i = np.linalg.norm(x, axis=-1).argsort()
    x = x[i]
    # Rescale DM
    x = x * (self.inp_space[1] - self.inp_space[0]) + self.inp_space[0]
    # Restructure DM
    x = np.split(x, x.shape[-1], axis=-1)
    x = {k: x[i].reshape(-1,1) for (i, k) in enumerate(self.inp_space_lbl)}
    # From input space to primitive variables
    x = self.inp_to_prim(x)
    # Saving and returning sampled initial solutions
    prim = pd.DataFrame(data=x, columns=self.prim_lbl)
    prim.to_csv(
      path_or_buf=self.path_to_data+"/init_sols.csv",
      float_format="%.8e",
      index=False
    )
    return x

  def inp_to_prim(
    self,
    samples
  ):
    samples = tf.nest.map_structure(tf.constant, samples)
    # Set initial composition
    samples[f"X_{self.molecule}"] = 1 - samples[f"X_{self.atom}"]
    self.mix.init_mix(
      x={k: samples[f"X_{k}"] for k in self.sp_th.keys()},
      var="X",
      p_mix=samples["p"],
      T=samples["Ti"],
      flatten=False
    )
    # Collect primitive variables
    prim_var = []
    for st in self.sp_th.values():
      prim_var.append(st.specie.rho)
    prim_var.append(samples["T"])
    return tf.concat(tuple(prim_var), axis=1).numpy()

  # Evaluate execution time
  # ===================================
  def eval_exec_time(
    self,
    t_end,
    init_sols
  ):
    print("> t_end = %.5e s" % t_end)
    t_end_str = ("%.5e" % t_end).replace("e", "d")
    pcm.utils.replace(
      file_in=self.path_to_runs + "/run.bkp/inputfile.sh",
      patterns=("TE",),
      replaces=(f"TE = {t_end_str}\n",),
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
