import os
import json
import torch
import shutil
import subprocess
import numpy as np
import pandas as pd
import joblib as jl
import tensorflow as tf

from tqdm import tqdm
from pyDOE import lhs
from nitrom import utils
from nitrom import interpolant
from nitrom import backend as bkd
from silx.io.dictdump import dicttoh5
from hypernet.apps import utils as hyputils


class TrajGenerator(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    nb_traj,
    time_lim,
    time_pts,
    inp_space,
    mom_order=4,
    path_to_wdir=None,
    load_plato="module load plato",
    nb_workers=4,
    nb_threads=4
  ):
    # Paths
    # ---------------------------------
    self.cwd = path_to_wdir
    if (self.cwd is None):
      self.cwd = os.getcwd()
    self.path_to_phy = self.cwd + "/physics/"
    self.path_to_runs = self.cwd + "/runs/"
    self.path_to_trajs = self.cwd + "/trajs.hdf5"
    # Trajectories quantities
    # ---------------------------------
    self.nb_traj = nb_traj
    self.time_vec = np.logspace(
      *[np.log10(t) for t in time_lim],
      num=time_pts
    )
    self.inp_space = np.vstack(list(inp_space.values())).T
    self.inp_space_lbl = list(inp_space.keys())
    # Initializing thermophysical models
    # -----------------------------------
    with open(self.path_to_phy + "/inputs.json") as file:
      phyinp = json.load(file)
    # Species thermo
    self.sp_th = hyputils._get_specie_thermos(phyinp)
    # Mixture
    self.mix = hyputils._get_mixture(phyinp, self.sp_th)
    # Kinetics
    self.kin = hyputils._get_kinetics(phyinp, self.sp_th, self.mix)
    # Check if StS or CG
    self.is_sts = (not any([
      (st.thermo.int.__name__ == "LinCoarseGrained") \
        for st in self.sp_th.values()
    ]))
    # Method of Moments order
    self.mom_order = mom_order
    # Labeling
    # ---------------------------------
    # Species
    self.species = []
    for st in self.sp_th.values():
      if ((st.specie.nb_comp > 1) and (not self.is_sts)):
        self.species.extend([
          st.specie.name + f"_{i+1}" for i in range(st.specie.nb_comp)
        ])
        self.molecule = st.specie.name
      else:
        self.species.append(st.specie.name)
        self.atom = st.specie.name
    self.Xi_names = [f"X_{s}" for s in self.species]
    self.Yi_names = [f"Y_{s}" for s in self.species]
    # Box solution
    self.sol_lbl = ['t'] + self.Xi_names + ["T", "rho", "p", "n", "E"]
    # Environment
    # ---------------------------------
    self.env = utils.load_external_module(load_plato)
    # Multi-threading
    self.nb_workers = nb_workers
    self.nb_threads = nb_threads

  # Sample initial conditions
  # ===================================
  def sample_inp_space(
    self
  ):
    # Contruct design matrix (DM)
    d = self.inp_space.shape[1]
    x = lhs(d, self.nb_traj)
    # Rescale DM
    x = x * (self.inp_space[1] - self.inp_space[0]) + self.inp_space[0]
    # Restructure DM
    x = [xi.reshape(-1,1) for xi in np.split(x, x.shape[-1], axis=-1)]
    x = {k: bkd.to_backend(x[i]) for (i, k) in enumerate(self.inp_space_lbl)}
    # Manipulate DM
    return self._manipulate_inp_space(x)

  def _manipulate_inp_space(
    self,
    samples
  ):
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
    return torch.hstack(tuple(prim_var))

  # Generate trajectories
  # ===================================
  def generate_trajs(self, samples):
    # Computing solutions
    print('> Computing 0D cases')
    samples = bkd.to_numpy(samples)
    cases = jl.Parallel(self.nb_workers)(
      jl.delayed(self._compute_solution)(
        case_nb=i, path=self.path_to_runs, sol=sol
      ) for (i, sol) in enumerate(tqdm(samples, ncols=80))
    )
    # cases = []
    # for (i, sol) in enumerate(tqdm(samples, ncols=80)):
    #   c = self._compute_solution(
    #     case_nb=i, path=self.path_to_runs, sol=sol
    #   )
    #   cases.append(c)
    # Filter solutions
    cases = {k: case for (k, case) in cases if (case is not None)}
    # Saving computed solutions
    print('> Saving 0D cases')
    dicttoh5(
      treedict=cases,
      h5file=self.path_to_trajs,
      overwrite_data=True,
      create_dataset_args={'compression': "gzip"}
    )

  # Compute solution
  # -----------------------------------
  def _compute_solution(
    self,
    case_nb,
    path,
    sol
  ):
    # Create run folder
    case_id = str(int(case_nb+1)).zfill(6)
    case_path = shutil.copytree(
      path + f'/run.bkp',
      path + f'/run.{case_id}',
      dirs_exist_ok=True
    )
    os.chdir(case_path)
    # Write initial solution for PLATO
    with open("init_sol.dat", "w") as file:
      np.savetxt(file, sol.reshape(-1))
    try:
      # Run 0D simulation
      subprocess.run(
        f"box inputfile.sh {self.nb_threads}",
        shell=True,
        env=self.env,
        timeout=1e2,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
      )
      # Postprocess solution
      sol = self._postproc_solution("./output_box/")
    except subprocess.TimeoutExpired:
      sol = None
    # Delete run folder
    os.chdir(self.cwd)
    shutil.rmtree(case_path)
    # Return solution
    return (case_id, sol)
  
  def _postproc_solution(
    self,
    path_to_sol
  ):
    sol = self._read_solution(path_to_sol)
    sol = tf.nest.map_structure(bkd.to_backend, sol)
    sol = self._interpolate_solution(sol)
    sol = self._compute_moments(sol)
    return tf.nest.map_structure(bkd.to_numpy, sol)

  # Read solution
  # -----------------------------------
  def _read_solution(
    self,
    path_to_sol
  ):
    container = {}
    # Read box file
    box = pd.read_csv(
      path_to_sol + "/box.dat",
      skiprows=2,
      header=None,
      delimiter="  ",
      engine="python"
    )
    box.columns = self.sol_lbl
    # Get macroscopic quantities of interest
    container["qoi"] = self._read_qoi(box)
    # Get temperature, molar fractions, and derivatives
    if (not self.is_sts):
      container.update(self._read_sol_cg(box))
    else:
      container.update(self._read_sol_sts(box))
    return container

  def _read_qoi(
    self,
    box
  ):
    qoi = list(set(self.sol_lbl) - set(self.Xi_names + ['t','T']))
    qoi = box[qoi].to_dict('list')
    qoi = {k: np.array(v).reshape(-1,1) for (k, v) in qoi.items()}
    return qoi

  def _read_sol_cg(
    self,
    box
  ):
    container = {}
    # Get solution vector
    t, n, T = [box[k].to_numpy().reshape(-1,1) for k in ('t','n','T')]
    n = box[self.Xi_names].to_numpy() * n.reshape(-1,1)
    t, n, T = [bkd.to_backend(x) for x in (t, n, T)]
    # Compute source term
    w = self.kin.function(t=t, y=[n, t])
    dn = w[:,self.kin.idx_c]
    dT = w[:,self.kin.idx_T]
    # Store data
    container["t"] = t
    for (k, nx, Tx) in (('z', n, T), ("dz", dn, dT)):
      container[k] = {
        "n": {
          s: nx[:,st.specie.comp_indices] for (s, st) in self.sp_th.items()
        },
        "T": Tx.reshape(-1,1)
      }
    return container

  def _read_sol_sts(
    self,
    box
  ):
    return {}

  # def _read_int_temp(self, path_to_sol, pd_kwargs):
  #   temp = pd.read_csv(path_to_sol + "/Tint.dat", **pd_kwargs)
  #   temp.columns = ["t", "Ti"]
  #   temp = temp["Ti"].to_numpy()
  #   return temp

  # def _read_pop(self, path_to_sol, nb_steps):
  #   # Get molecule
  #   specie = self.sp_th[self.molecule].specie
  #   # Get degeneracies
  #   degen = bkd.to_numpy(specie.lev['g']).reshape(1,-1)
  #   # Read distribution function
  #   filename = path_to_sol + f'/output_box/pop_{specie.name}.dat'
  #   n_ov_g = np.loadtxt(filename, comments='&', skiprows=2, usecols=(1))
  #   n_ov_g = np.abs(n_ov_g).reshape((nb_steps, specie.nb_comp))
  #   return n_ov_g * degen

  # Interpolate solution
  # -----------------------------------
  def _interpolate_solution(
    self,
    container
  ):
    t = self.time_vec.reshape(-1,1)
    container = tf.nest.map_structure(
      lambda f: self._interpolate(
        x=torch.log(bkd.to_backend(t) + 1e-15),
        c=torch.log(container["t"] + 1e-15),
        f=f
      ),
      container
    )
    container['t'] = t
    return container

  def _interpolate(self, x, c, f, order=2, smoothing=0.0):
    phi = interpolant.get_phi(order)
    w, v = interpolant.solve(c, f, phi, smoothing)
    return interpolant.apply(x, c, phi, w, v)

  # Method of moments
  # -----------------------------------
  def _compute_moments(
    self,
    sol
  ):
    sol["mom"] = self.sp_th[self.molecule].thermo.int.compute_moments(
      n=sol['z']['n'][self.molecule],
      T=sol['z']['T'],
      order=self.mom_order
    )
    return sol
