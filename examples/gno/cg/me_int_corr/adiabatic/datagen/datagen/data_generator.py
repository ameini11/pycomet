import os
import json
import shutil
import subprocess
import numpy as np
import scipy as sp
import pandas as pd
import joblib as jl
import pycomet as pcm
import tensorflow as tf

from pyDOE import lhs
from tqdm import tqdm
from silx.io.dictdump import dicttoh5, h5todict
from pycomet.sampling import DifferentialBased

# HyperNet package
from hypernet import const
from hypernet import backend as hypbkd
from hypernet.apps import utils as hyputils
from hypernet.thermophysics import specie_thermo as st_mod


class DataGenerator(pcm.data.BaseDataGenerator):

  # Initialization
  # ===================================
  def __init__(
    self,
    time_lim=[0,1],
    time_eps=1e-15,
    inp_space={},
    svd=None,
    read_init_sols=False,
    save_sols_only=False,
    train_pts=32,
    scenarios=100,
    test_cases=10,
    validation_split=0.2,
    path_to_physics="./physics",
    grad_split=0.75,
    load_plato="module load plato",
    nb_workers=4,
    nb_threads=4,
    *args,
    **kwargs
  ):
    super(DataGenerator, self).__init__(*args, **kwargs)
    # Paths
    # ---------------------------------
    self.cwd = os.getcwd()
    self.path_to_data = self.cwd + "/data/"
    self.path_to_runs = self.cwd + "/datagen/runs/"
    self.path_to_physics = path_to_physics
    os.makedirs(self.path_to_data, exist_ok=True)
    # Data quantities
    # ---------------------------------
    self.valid_pts = round(validation_split*train_pts)
    self.train_pts = train_pts - self.valid_pts
    self.scenarios = scenarios
    self.test_cases = test_cases
    # Dynamics sampler
    self.time_lim = time_lim
    self.time_eps = time_eps
    self.dyn_sampler = DifferentialBased(
      method="source",
      grad_split=grad_split,
      xlim=np.log(self.time_lim)
    )
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
    # Kinetics
    self.kin = hyputils._get_kinetics(phyinp, self.sp_th, self.mix)
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
    # Set nodes/edges
    # -----------------------------------
    self.set_states()
    self.set_rates()
    # SVD decomposition
    # -----------------------------------
    self.svd = svd
    self.compute_svd_coeff = (self.svd is not None)
    if self.compute_svd_coeff:
      self.svd = tf.nest.map_structure(np.loadtxt, self.svd)
      self.svd_nb_coeff = {}
      for (k, ksvd) in self.svd.items():
        self.svd_nb_coeff[k] = ksvd["phi"].shape[1] + 2
    # Labeling data
    # -----------------------------------
    # Box solution
    self.xi_names = [f"X_{s}" for s in self.species]
    self.box_lbl = ["t"] + self.xi_names + ["T", "rho", "p", "n", "E"]
    # Dataset
    self.data_lbl = ["t"] \
      + [f"n_{self.atom}", f"n_{self.molecule}", "T", f"e_{self.molecule}"] \
      + [f"n_{s}_0" for s in self.species[1:]] + [f"t_p{self.atom}"]
    if self.compute_svd_coeff:
      for (k, nb) in self.svd_nb_coeff.items():
        self.data_lbl += [f"w{i+1}_{k}" for i in range(nb)]
    self.data_lbl += [f"n_{s}" for s in self.species[1:]]
    # Control variables
    # -----------------------------------
    self.read_init_sols = read_init_sols
    # Save solutions for low-rank approximation calculations
    self.save_sols_only = save_sols_only
    # Environment
    # -----------------------------------
    self.env = pcm.utils.load_external_module(load_plato)
    # Multi-threading
    self.nb_workers = nb_workers
    self.nb_threads = nb_threads

  # Set node/edge static attributes
  # ===================================
  def set_states(self):
    sp = self.sp_th[self.molecule].specie
    # Probability matrix
    P = hypbkd.to_numpy(sp.mappings["lev"])
    # StS energies
    ei = [hypbkd.to_numpy(sp.lev[k]) for k in ("ERot", "EVib")]
    ei = np.stack(ei, axis=1)
    # CG energies
    eg = (np.matmul(ei.T, P) / np.sum(P, axis=0, keepdims=True)).T
    # CG normalized energies
    eg_min = np.amin(eg[eg>0])*1e-2
    eg_norm = np.log((eg+eg_min)/const.eV_to_J)
    eg_std = np.std(eg_norm, axis=0, keepdims=True)
    eg_mean = np.mean(eg_norm, axis=0, keepdims=True)
    eg_norm = (eg_norm - eg_mean) / eg_std
    # Saving energies
    path = self.path_to_data + "/physics/states/"
    os.makedirs(path, exist_ok=True)
    np.savetxt(path + "/energies.txt", eg, fmt="%.8e")
    np.savetxt(path + "/energies_norm.txt", eg_norm, fmt="%.8e")

  def set_rates(self):
    # Define function to read parameters
    get_param = lambda reaction: {
      k: getattr(reaction.fwd, k) for k in ("A", "beta", "Ta")
    }
    # Loop over processes
    rates = {}
    for reaction in self.kin.reactions:
      if (reaction.param["process"] == 5):
        rates["kef"] = get_param(reaction)
      elif (reaction.param["process"] == 2):
        rates["kd"] = get_param(reaction)
    # rates = tf.nest.map_structure(lambda x: x.numpy().squeeze(), rates)
    rates = tf.nest.map_structure(lambda x: x.squeeze(), rates)
    # Saving rates
    os.makedirs(self.path_to_data + "/physics/", exist_ok=True)
    dicttoh5(
      treedict=rates,
      h5file=self.path_to_data + "/physics/rates.hdf5",
      overwrite_data=True
    )

  # Generate data
  # ===================================
  def get_data_src(
    self
  ):
    data_src = {}
    # Sampling input space
    if self.read_init_sols:
      pcm.utils.print_fn("Reading sampled input space", lev=1)
      init_sols = h5todict(self.path_to_runs + "/samples/init_sols.hdf5")
    else:
      pcm.utils.print_fn("Sampling input space", lev=1)
      init_sols = self.sample_inp_space()
    # Generate training/validation data
    pcm.utils.print_fn("Training/validation data", lev=1)
    data_src.update(self.training_data(init_sols))
    # Generate testing data
    data_src.update(self.testing_data(init_sols))
    return data_src

  # Sample initial conditions
  # ===================================
  def sample_inp_space(self):
    # Contruct design matrix (DM)
    d = self.inp_space.shape[1]
    s = self.scenarios * 2 + self.test_cases
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
    # Splitting sampled initial solutions
    if (self.test_cases > 0):
      indices = pcm.utils.stratified_sampling(np.arange(s), self.test_cases)
      samples = [tf.nest.map_structure(lambda z: z[i], x) for i in indices]
      samples = {
        "train": samples[0][0::2],
        "valid": samples[0][1::2],
        "test": samples[1]
      }
    else:
      samples = {
        "train": x[0::2],
        "valid": x[1::2]
      }
    # Saving sampled initial solutions
    os.makedirs(self.path_to_runs + "/samples/", exist_ok=True)
    dicttoh5(
      treedict=samples,
      h5file=self.path_to_runs + "/samples/init_sols.hdf5",
      overwrite_data=True,
      create_dataset_args={"compression": "gzip"}
    )
    # Returning sampled initial solutions
    return samples

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

  # Compute solutions
  # ===================================
  def compute_solutions(
    self,
    path,
    dataid,
    init_sols
  ):
    if (not os.path.exists(path + "/sols.hdf5")):
      pcm.utils.print_fn(f"Computing 0D cases: {dataid}", lev=2)
      # Computing solutions
      sols = jl.Parallel(self.nb_workers)(
        jl.delayed(self.compute_solution)(
          path=self.path_to_runs,
          case_nb=i,
          init_sol=init_sol
        ) for (i, init_sol) in enumerate(tqdm(init_sols, ncols=80))
      )
      # Filter solutions
      sols = {i: c for (i, c) in sols if (c is not None)}
      # Saving computed solutions
      pcm.utils.print_fn(f"Saving 0D cases: {dataid}", lev=2)
      os.makedirs(path, exist_ok=True)
      dicttoh5(
        treedict=sols,
        h5file=path + "/sols.hdf5",
        overwrite_data=True,
        create_dataset_args={"compression": "gzip"}
      )
    else:
      pcm.utils.print_fn(f"Reading 0D cases: {dataid}", lev=2)
      sols = h5todict(path + "/sols.hdf5")
    return sols

  # Compute/Read solution
  # -----------------------------------
  def compute_solution(
    self,
    path,
    case_nb,
    init_sol
  ):
    # Create run folder
    case_id = str(int(case_nb+1)).zfill(6)
    case_path = shutil.copytree(
      path + f"/run.bkp",
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
        timeout=1e2,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
      )
      # Postprocess solution
      sol = self.read_solution(path_to_sol="./output_box/")
    except subprocess.TimeoutExpired:
      sol = None
    # Delete run folder
    os.chdir(self.cwd)
    shutil.rmtree(case_path)
    # Return solution
    return (case_id, sol)

  def read_solution(
    self,
    path_to_sol=None
  ):
    # Read box file
    box = np.loadtxt(path_to_sol + "/box.dat")
    if (box < 0).any():
      return None
    box = pd.DataFrame(box, columns=self.box_lbl)
    # Get solution
    t, n, T = [box[k].to_numpy().reshape(-1,1) for k in ("t", "n", "T")]
    n = n * box[self.xi_names].to_numpy()
    y = np.hstack([n,T])
    return {"t": t, "y": y}

  # Postprocess solution
  # -----------------------------------
  def postproc_solution(self, sol, nb_pts=-1):
    sol = [sol[k] for k in ("t", "y")]
    # Get SVD coefficients for (na, T) decomposition
    if self.compute_svd_coeff:
      w = self.svd_decomp(*sol)
    # Sample solution in time
    if (nb_pts > 0):
      sol = self.sample_solution(*sol, nb_pts)
    # Get physical QoI
    sol = self.prim_to_phys(*sol)
    # Store SVD coefficients for (na, T) decomposition
    if self.compute_svd_coeff:
      sol["svd"] = w
    return sol

  def prim_to_phys(self, t, y):
    # Manipulate primitive variables
    n, T = y[:,:-1], y[:,-1:]
    # Update thermochemistry and mixture
    st_mod.update_specie_thermos(self.sp_th, T)
    self.mix.update(n)
    # Store physical variables
    phys = {
      "t": t,
      "T": T,
      "n": np.sum(n, axis=-1, keepdims=True)
    }
    phys = tf.nest.map_structure(lambda x: x.reshape(-1,1), phys)
    for (name, st) in self.sp_th.items():
      phys["X_"+name] = st.specie.X
    return phys

  def sample_solution(self, t, y, nb_pts):
    idx = self.dyn_sampler.sample(
      x=np.log(t+self.time_eps).reshape(-1,1),
      y=y,
      nb_pts=nb_pts
    )
    return t[idx], y[idx]

  # SVD decomposition
  # -----------------------------------
  def svd_decomp(self, t, y):
    sol = self.prim_to_phys(t, y)
    # Extract variables
    var = ("t", "T", "n", f"X_{self.atom}")
    t, T, n, Xa = [sol[k].reshape(-1) for k in var]
    # Compute QoI for SVD decomposition
    na = Xa * n
    pa = na * const.UKB * T
    tp = t * pa
    # Compute SVD coefficients
    w = {}
    yi = {"T": T, "na": na, "pa": pa}
    for (k, ksvd) in self.svd.items():
      w[k] = self._svd_coeff(
        t_in=tp, t_out=ksvd["tp"], y=yi[k], phi=ksvd["phi"]
      )
    # Return all the coefficients
    return w

  def _svd_coeff(self, t_in, t_out, y, phi):
    # Interpolate data onto bases grid
    y = self._svd_interpolate(t_in, t_out, y)
    # Normalize data
    m = np.mean(y).reshape(1)
    s = np.std(y).reshape(1)
    y = (y - m) / s
    # Compute SVD coefficients using OLS
    w = self._svd_ols(phi, y)
    # Return all the coefficients
    return np.concatenate([m,s,w]).reshape(-1)
  
  def _svd_interpolate(self, t_in, t_out, y):
    # Extend domain
    t_in = np.concatenate([t_in, t_in[-1:]*1e2])
    y = np.concatenate([y, y[-1:]])
    # Interpolate
    return sp.interpolate.interp1d(t_in, y, kind='cubic')(t_out)

  def _svd_ols(self, x, y):
    # Ordinary least squares (OLS) algorithm
    # > Compute Gram matrix
    gr = np.matmul(x.T, x)
    # > Inverse of Gram matrix
    gri = np.linalg.inv(gr)
    # > Moment matrix
    ym = np.matmul(x.T, y)
    # > Compute coefficients
    w = np.matmul(gri, ym)
    return w

  # Manage data
  # ===================================
  def get_data(self, sol, nb_pts=-1):
    sol = self.postproc_solution(sol, nb_pts)
    # Extract variables
    t, T, n = [sol[k].reshape(-1,1) for k in ("t", "T", "n")]
    na, nm = [sol[f"X_{s}"] * n for s in self.sp_th.keys()]
    # Update thermo
    st_mod.update_specie_thermos(self.sp_th, T)
    em = self.sp_th[self.molecule].thermo.int.e
    # Compose inputs
    # -------------
    nb_pts = len(t)
    # > Time-dependent QoI
    nt = np.sum(nm, axis=-1, keepdims=True)
    et = np.sum(nm*em, axis=-1, keepdims=True) / nt
    qt = np.hstack([t, na, nt, T, et])
    # > Initial distribution
    nm0 = np.tile(nm[0].reshape(1,-1), (nb_pts,1))
    # > Correction factor
    tp = t * na * const.UKB * T
    f = np.log(tp + 1e-10)
    if self.compute_svd_coeff:
      w = list(sol["svd"].values())
      w = np.concatenate(w).reshape(1,-1)
      w = np.tile(w, (nb_pts,1))
      f = np.hstack([f, w])
    # Compose data
    # -------------
    x, y = [qt, nm0, f], nm
    return [x, y]

  def stack_data(self, data, data_new, data_type):
    if (data_type not in data):
      data[data_type] = data_new
    else:
      data[data_type] = tf.nest.map_structure(
        lambda x, y: np.concatenate((x,y), axis=0),
        data[data_type],
        data_new
      )
    return data

  def save_data(self, filename, data, bounds=False):
    data = np.hstack(tf.nest.flatten(data))
    df = pd.DataFrame(data, columns=self.data_lbl)
    df.to_csv(filename + ".csv", index=0, float_format="{:e}".format)
    if bounds:
      bd = np.vstack([np.amin(data, axis=0), np.amax(data, axis=0)])
      df = pd.DataFrame(bd, columns=self.data_lbl)
      df.to_csv(filename + "_bounds.csv", index=0, float_format="{:e}".format)

  # Training/Validation data
  # ===================================
  def training_data(
    self,
    init_sols
  ):
    # Generating data
    data = {}
    for k in ("train", "valid"):
      # Computing 0D data
      sols = self.compute_solutions(
        path=self.path_to_runs + f"/samples/data/{k}/",
        dataid=k,
        init_sols=init_sols[k]
      )
      # Sampling 0D data
      if (not self.save_sols_only):
        pcm.utils.print_fn(f"Sampling 0D cases: {k}", lev=2)
        nb_pts = self.train_pts if (k == "train") else self.valid_pts
        for sol in tqdm(sols.values(), ncols=80):
          idata = self.get_data(sol, nb_pts)
          data = self.stack_data(data, idata, k)
        # Save data in a 'csv' file
        filename = self.path_to_data + f"/{k}"
        self.save_data(filename, data[k], bounds=True)
    return data

  # Testing data
  # ===================================
  def testing_data(
    self,
    init_sols
  ):
    data = {}
    if (("test" in init_sols) and (not self.save_sols_only)):
      path = self.path_to_data + "/test/"
      os.makedirs(path, exist_ok=True)
      # Computing 0D data
      sols = self.compute_solutions(
        path=self.path_to_runs + "/samples/data/test/",
        dataid="test",
        init_sols=init_sols["test"]
      )
      # Saving 0D data
      for (i, sol) in tqdm(sols.items(), ncols=80):
        ikey = "case_" + i
        data[ikey] = self.get_data(sol)
        self.save_data(filename=path+f"/{ikey}", data=data[ikey])
    return data
