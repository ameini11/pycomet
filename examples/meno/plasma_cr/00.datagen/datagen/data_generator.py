import os
import json
import shutil
import subprocess
import numpy as np
import scipy as sp
import pandas as pd
import joblib as jl
import dill as pickle
import pycomet as pcm
import tensorflow as tf

from tqdm import tqdm
from silx.io.dictdump import dicttoh5, h5todict

from pycomet.sampling import DifferentialBased
from pycomet.networks.elements import LinAutoencoder
from pycomet.networks.chemkin.plasma_cr.meno import const, Mixture


class DataGenerator(pcm.data.BaseDataGenerator):

  # Initialization
  # ===================================
  def __init__(
    self,
    time_lim=[0,1],
    time_eps=1e-15,
    time_eval=None,
    train_pts=32,
    validation_split=0.2,
    rom_config=None,
    path_to_dtb=None,
    path_to_inp_space=None,
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
    self.path_to_dtb = path_to_dtb
    self.path_to_inp_space = path_to_inp_space
    os.makedirs(self.path_to_data, exist_ok=True)
    # Data quantities
    # ---------------------------------
    self.valid_pts = round(validation_split*train_pts)
    self.train_pts = train_pts - self.valid_pts
    # Dynamics sampler
    self.time_lim = time_lim
    self.time_eps = time_eps
    self.time_eval = time_eval
    if (self.time_eval is not None):
      self.time_eval = np.array(self.time_eval).squeeze()
    self.dyn_sampler = DifferentialBased(
      method="source",
      grad_split=grad_split,
      xlim=np.log(self.time_lim)
    )
    # Mixture model
    # -----------------------------------
    species_order = ("em", "Ar", "Arp")
    self.mix = Mixture(
      species={k: path_to_dtb + f"/species/{k}.json" for k in species_order},
      species_order=species_order
    )
    self.set_states()
    # Linear autoencoder
    # -----------------------------------
    if isinstance(rom_config, str):
      with open(rom_config, "rb") as file:
        rom_config = pickle.load(file)
    self.rom = LinAutoencoder(**rom_config)
    # Labeling data
    # -----------------------------------
    # Box solution
    self.box_lbl = "t x_em x_Ar x_Arp Th Te rho p n E".split()
    # Dataset
    # > Species names
    self.species_names = []
    for k in self.mix.species_order:
      nc = self.mix.species[k].nb_comp
      names = [k] if (nc == 1) else [k+f"{i+1}" for i in range(nc)]
      self.species_names.extend(names)
    # > Labels
    self.data_lbl = [f"rho_{s}_0" for s in self.species_names] \
      + ["Th_0", "Te_0", "t"] \
      + [f"w_{s}" for s in self.species_names] \
      + ["Th", "Te", "pe", "n_em", "n_Ar1"] \
      + ["rho"] + [f"z{i+1}_0" for i in range(self.rom.dim)] \
      + ["Th_0", "Te_0", "ln_t_pe", "ln_t_ph"]
    # > Indices
    i = 0
    self.data_idx = {}
    for (k, ik) in (
      ("rho_0", self.mix.nb_comp),
      ("Th_0", 1),
      ("Te_0", 1),
      ("t", 1),
      ("w", self.mix.nb_comp),
      ("Th", 1),
      ("Te", 1),
      ("pe", 1),
      ("n_em", 1),
      ("n_Ar1", 1),
      ("x0_norm", 5+self.rom.dim)
    ):
      self.data_idx[k] = np.arange(i,i+ik).tolist()
      i += ik
    assert (i == len(self.data_lbl))
    with open(self.path_to_data + "/indices.json", "w") as file:
      json.dump(self.data_idx, file, indent=2, ensure_ascii=False)
    # Environment
    # -----------------------------------
    self.env = pcm.utils.load_external_module(load_plato)
    # Multi-threading
    self.nb_workers = nb_workers
    self.nb_threads = nb_threads

  # Set node static attributes
  # ===================================
  def set_states(self):
    q = []
    for k in self.mix.species_order:
      if (k != "em"):
        # Species
        sp = self.mix.species[k]
        # Properties
        es = sp.lev["E"].numpy().squeeze()
        js = sp.lev["jc"].numpy().squeeze()
        zs = np.full(sp.nb_comp, sp.Z)
        # Composing
        qs = np.vstack([es,js,zs]).T
        if (k == "Ar"):
          qs = qs[1:]
        q.append(qs)
    q = np.vstack(q)
    # Normalization
    qmin = np.amin(q, axis=0, keepdims=True)
    qmax = np.amax(q, axis=0, keepdims=True)
    qnorm = 2.0 * (q - qmin) / (qmax - qmin) - 1.0
    # Saving states
    np.savetxt(self.path_to_data + "/states.txt", qnorm, fmt="%.10e")

  # Generate data
  # ===================================
  def get_data_src(
    self
  ):
    data_src = {}
    # Input space
    pcm.utils.print_fn("Reading sampled input space", lev=1)
    init_sols = self.read_inp_space()
    # Generate training/validation data
    pcm.utils.print_fn("Training/validation data", lev=1)
    data_src.update(self.training_data(init_sols))
    # Generate testing data
    data_src.update(self.testing_data(init_sols))
    return data_src

  # Read input space
  # ===================================
  def read_inp_space(self):
    xinp, xsol = {}, {}
    for k in ("train", "valid", "test"):
      filename = self.path_to_inp_space + f"/{k}.csv"
      if os.path.exists(filename):
        data = pd.read_csv(filename)
        xinp[k] = data.values
        xsol[k] = self.inp_to_prim(data)
    # Check if test cases are in training cases
    if ("test" in xinp):
      assert (~np.isin(xinp["test"], xinp["train"]).all(-1)).all(-1)
    # Saving sampled initial solutions
    os.makedirs(self.path_to_runs + "/samples/", exist_ok=True)
    for (k, data) in (
      ("inp_space", xinp),
      ("init_sols", xsol)
    ):
      dicttoh5(
        treedict=data,
        h5file=self.path_to_runs + f"/samples/{k}.hdf5",
        overwrite_data=True,
        create_dataset_args={"compression": "gzip"}
      )
    # Returning sampled initial solutions
    return xsol

  def inp_to_prim(
    self,
    samples
  ):
    # Decode latent space
    z = samples[[f"z{i+1}" for i in range(self.rom.dim)]].values
    w = self.rom(z, method=2).numpy()
    # Enforce conservation: mass and charge
    w = self._enforce_cons_mass(w)
    w = self._enforce_cons_charge(w)
    # Collect primitive variables
    rho = w * samples[["rho"]].values
    Th, dT = [samples[[k]].values for k in ("Th", "dT")]
    Te = Th - dT
    return np.hstack([rho, Th, Te])

  def _enforce_cons_mass(self, z):
    i = self.mix.species["Ar"].indices[0]
    z[:,i] = 0.0
    z[:,i] = 1.0 - np.sum(z, axis=-1)
    return z

  def _enforce_cons_charge(self, w):
    x = self.mix.get_x(w).numpy()
    ie, ii = [self.mix.species[k].indices for k in ("em", "Arp")]
    x[:,ie] = np.sum(x[:,ii], axis=-1, keepdims=True)
    x = self._enforce_cons_mass(x)
    w = self.mix.get_w(x).numpy()
    return w

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
    if (box < 0.0).any():
      return None
    box = pd.DataFrame(box, columns=self.box_lbl)
    # Read populations
    n = {k: self._read_pop(path_to_sol, k) for k in ("Ar", "Arp")}
    n["em"] = (box["x_em"] * box["n"]).values.reshape(-1,1)
    n = np.hstack([n[k] for k in self.mix.species_order])
    # Molar fractions
    x = n / np.sum(n, axis=-1, keepdims=True)
    # Mass fractions
    w = self.mix.get_w(x).numpy()
    # Get primitive variables
    q = {k: box[k].values.reshape(-1,1) for k in ("t", "rho", "Th", "Te")}
    y = np.hstack([q["rho"]*w, q["Th"], q["Te"]])
    return {"t": q["t"], "y": y}

  def _read_pop(
    self,
    path_to_sol,
    name
  ):
    filename = path_to_sol + f"/pop_{name}.dat"
    sp = self.mix.species[name]
    ni = np.loadtxt(filename, comments="&", skiprows=2)[:,1]
    nt = len(ni) / sp.nb_comp
    ni = np.vstack(np.split(ni, nt, axis=0))
    return ni * sp.lev["g"].numpy()

  # Postprocess solution
  # -----------------------------------
  def postproc_solution(self, sol, nb_pts=-1, teval=None):
    # Limit solution in time
    sol = self.limit_solution(**sol)
    # Interpolate solution in time
    if (teval is not None):
      sol = self.interpolate_solution(*sol, teval)
    # Sample solution in time
    if (nb_pts > 0):
      sol = self.sample_solution(*sol, nb_pts)
    # Get physical QoI
    return self.prim_to_phys(*sol)

  def limit_solution(self, t, y):
    idx = np.argwhere(
      (t.squeeze() >= self.time_lim[0]) * (t.squeeze() <= self.time_lim[1])
    ).squeeze()
    return t[idx], y[idx]

  def interpolate_solution(self, t, y, teval):
    y = sp.interpolate.interp1d(t.squeeze(), y, kind="cubic", axis=0)(teval)
    return teval, y

  def sample_solution(self, t, y, nb_pts):
    idx = self.dyn_sampler.sample(
      x=np.log(t+self.time_eps).reshape(-1,1),
      y=y,
      nb_pts=nb_pts
    )
    return t[idx], y[idx]

  def prim_to_phys(self, t, y):
    # Manipulate primitive variables
    nc = self.mix.nb_comp
    rho, Th, Te = np.split(y, [nc, nc+1], axis=-1)
    n = self.mix.get_n(rho).numpy()
    # Store physical variables
    phys = {
      "t": t,
      "Th": Th,
      "Te": Te,
      "n": np.sum(n, axis=-1),
      "rho": np.sum(rho, axis=-1)
    }
    for sp in self.mix.species.values():
      phys[f"n_{sp.name}"] = np.sum(n[:,sp.indices], axis=-1)
      phys[f"rho_{sp.name}"] = np.sum(rho[:,sp.indices], axis=-1)
    phys = tf.nest.map_structure(lambda x: x.reshape(-1,1), phys)
    # Molar/mass fractions
    phys["x"] = n / phys["n"]
    phys["w"] = rho / phys["rho"]
    # Pressures
    ne = phys["n_em"]
    nh = phys["n_Ar"] + phys["n_Arp"]
    phys["pe"] = ne * const.UKB * Te
    phys["ph"] = nh * const.UKB * Th
    phys["p"] = phys["pe"] + phys["ph"]
    # Return
    return phys

  # Manage data
  # ===================================
  def get_data(self, sol, nb_pts=-1, teval=None):
    # Initial time
    sol0 = tf.nest.map_structure(lambda x: x[:1], sol)
    sol0 = self.prim_to_phys(**sol0)
    # Dynamics
    sol = self.postproc_solution(sol, nb_pts, teval)
    # Extract variables
    # > Output variables
    t, w, Th, Te = [sol[k] for k in ("t", "w", "Th", "Te")]
    # > Additional variables
    pe, n_em = [sol[k] for k in ("pe", "n_em")]
    i1 = self.mix.species["Ar"].indices[:1]
    n_Ar1 = sol["x"][:,i1] * sol["n"]
    # Inputs
    nb_pts = len(t)
    x0 = np.hstack([sol0["rho"] * sol0["w"], sol0["Th"], sol0["Te"]])
    x0 = np.tile(x0, (nb_pts,1))
    x = [x0, t]
    # > Inputs to normalize
    z0 = self.rom(sol0["w"], method=1).numpy()
    mu0 = np.hstack([sol0["rho"], z0, sol0["Th"], sol0["Te"]])
    mu0 = np.tile(mu0, (nb_pts,1))
    ln_t_pe = np.log(t * sol0["pe"])
    ln_t_ph = np.log(t * sol0["ph"])
    x_norm = np.hstack([mu0, ln_t_pe, ln_t_ph])
    # Outputs
    y = np.hstack([w, Th, Te, pe, n_em, n_Ar1])
    # Return data
    return [x, y, x_norm]

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
    df.to_csv(filename + ".csv", index=0, float_format="%.10e")
    if bounds:
      bd = np.vstack([np.amin(data, axis=0), np.amax(data, axis=0)])
      df = pd.DataFrame(bd, columns=self.data_lbl)
      df.to_csv(filename + "_bounds.csv", index=0, float_format="%.10e")

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
      pcm.utils.print_fn(f"Sampling 0D cases: {k}", lev=2)
      nb_pts = self.train_pts if (k == "train") else self.valid_pts
      for sol in tqdm(sols.values(), ncols=80):
        data_i = self.get_data(sol, nb_pts)
        data = self.stack_data(data, data_i, k)
      # Save data in a "csv" file
      filename = self.path_to_data + f"/{k}"
      self.save_data(filename, data[k], bounds=True)
      # Specific time instants to evaluate
      if (self.time_eval is not None):
        pcm.utils.print_fn(
          f"Extract solution at specific time instants: {k}", lev=2
        )
        # Prepare data container
        teval = ["{:.2e}".format(tj) for tj in self.time_eval]
        teval_data = {tj: [] for tj in teval}
        # Interpolate solution
        _teval_data = jl.Parallel(self.nb_workers)(
          jl.delayed(self.get_data)(
            sol, teval=self.time_eval
          ) for sol in tqdm(sols.values(), ncols=80)
        )
        _teval_data = [np.hstack(tf.nest.flatten(d)) for d in _teval_data]
        # Assemble interpolated data
        for data_i in _teval_data:
          for (j, data_ij) in enumerate(data_i):
            teval_data[teval[j]].append(data_ij)
        teval_data = {
          tj: np.vstack(data_j) for (tj, data_j) in teval_data.items()
        }
        # Save interpolated data
        dicttoh5(
          treedict=teval_data,
          h5file=self.path_to_data + f"/{k}_teval.hdf5",
          overwrite_data=True,
          create_dataset_args={"compression": "gzip"}
        )
    return data

  # Testing data
  # ===================================
  def testing_data(
    self,
    init_sols
  ):
    data = {}
    if ("test" in init_sols):
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
