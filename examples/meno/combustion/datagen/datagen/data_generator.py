import os
import json
import numpy as np
import scipy as sp
import pandas as pd
import joblib as jl
import pycomet as pcm
import tensorflow as tf

from pyDOE import lhs
from tqdm import tqdm
from scipy.constants import Boltzmann as UKB
from silx.io.dictdump import dicttoh5, h5todict
from pycomet.sampling import DifferentialBased

from reactor import POLLUReactor


class DataGenerator(pcm.data.BaseDataGenerator):

  # Initialization
  # ===================================
  def __init__(
    self,
    species_file,
    reactions_file,
    time_pts=200,
    time_lim=[0,1],
    time_eps=1e-15,
    time_eval=None,
    inp_space=None,
    read_init_sols=False,
    train_pts=32,
    scenarios=100,
    test_cases=10,
    validation_split=0.2,
    grad_split=0.75,
    nb_workers=4,
    *args,
    **kwargs
  ):
    super(DataGenerator, self).__init__(*args, **kwargs)
    # Paths
    # ---------------------------------
    self.cwd = os.getcwd()
    self.path_to_data = self.cwd + "/data/"
    self.path_to_samples = self.cwd + "/datagen/samples/"
    os.makedirs(self.path_to_data, exist_ok=True)
    # Control variables
    # -----------------------------------
    self.read_init_sols = read_init_sols
    # Data quantities
    # ---------------------------------
    self.valid_pts = round(validation_split*train_pts)
    self.train_pts = train_pts - self.valid_pts
    self.scenarios = scenarios
    self.test_cases = test_cases
    # Time grid
    self.time_pts = time_pts
    self.time_lim = time_lim
    self.time_eps = time_eps
    self.time_eval = time_eval
    if (self.time_eval is not None):
      self.time_eval = np.array(self.time_eval).squeeze()
    self.tvec = self.get_tvec(*self.time_lim, num=self.time_pts)
    # Dynamics sampler
    self.dyn_sampler = DifferentialBased(
      method="source",
      grad_split=grad_split,
      xlim=np.log(self.time_lim)
    )
    # Input space
    with open(inp_space) as file:
      inp_space = json.load(file)
    self.inp_space = np.vstack(list(inp_space.values())).T
    self.inp_space_lbl = list(inp_space.keys())
    # Initializing plasma reactor
    # -----------------------------------
    self.reactor = POLLUReactor(
      species_file=species_file,
      reactions_file=reactions_file
    )
    self.set_species_and_rates()
    # Labeling data
    # -----------------------------------
    # Dataset
    species_names = self.reactor.species_to_index.keys()
    self.data_lbl = [f"y_{s}_0" for s in species_names] + ["t"] \
      + [f"y_{s}" for s in species_names]
    # Multi-threading
    # -----------------------------------
    self.nb_workers = nb_workers

  def get_tvec(self, start, stop, num):
    t = np.geomspace(start, stop, num=num-1)
    t = np.insert(t, 0, 0.0)
    return t

  # Set node/edge static attributes
  # ===================================
  def set_species_and_rates(self):
    path = self.path_to_data + "/physics/"
    os.makedirs(path, exist_ok=True)
    # Save species features (one-hot encoding)
    nb_comp = len(self.reactor.species_indices["lin"])
    np.savetxt(path + "/species_feat.txt", np.eye(nb_comp), fmt="%.8e")
    # Save species indices and rates
    for k in ("species_indices", "rates"):
      dicttoh5(
        treedict=getattr(self.reactor, k),
        h5file=path + f"/{k}.hdf5",
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
      init_sols = h5todict(self.path_to_samples + "/samples/init_sols.hdf5")
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
    os.makedirs(self.path_to_samples, exist_ok=True)
    dicttoh5(
      treedict=samples,
      h5file=self.path_to_samples + "/init_sols.hdf5",
      overwrite_data=True,
      create_dataset_args={"compression": "gzip"}
    )
    # Returning sampled initial solutions
    return samples

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
        jl.delayed(self.reactor.solve)(
          t=self.tvec,
          y0=init_sol
        ) for init_sol in tqdm(init_sols, ncols=80)
      )
      t = self.tvec.reshape(-1,1)
      name = lambda i: str(int(i+1)).zfill(6)
      sols = {name(i): {"t": t, "y": yi} for (i, yi) in enumerate(sols)}
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

  # Manage data
  # ===================================
  def get_data(self, sol, nb_pts=-1, teval=None):
    t, y = [sol[k] for k in ("t", "y")]
    y0 = y[0]
    # Interpolate solution in time
    if (teval is not None):
      t, y = self.interpolate_solution(t, y, teval)
    # Sample solution in time
    if (nb_pts > 0):
      t, y = self.sample_solution(t[1:], y[1:], nb_pts)
    # Return data
    y0 = np.tile(y0.reshape(1,-1), (len(t),1))
    return [[y0, t], y]

  def interpolate_solution(self, t, y, teval):
    t, y = t.squeeze(), np.log(y)
    y = sp.interpolate.interp1d(t, y.T, kind="cubic")(teval).T
    return teval, np.exp(y)

  def sample_solution(self, t, y, nb_pts):
    idx = self.dyn_sampler.sample(
      x=np.log(t+self.time_eps).reshape(-1,1),
      y=y,
      nb_pts=nb_pts
    )
    return t[idx], y[idx]

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
        path=self.path_to_samples + f"/data/{k}/",
        dataid=k,
        init_sols=init_sols[k]
      )
      # Sampling 0D data
      pcm.utils.print_fn(f"Sampling 0D cases: {k}", lev=2)
      nb_pts = self.train_pts if (k == "train") else self.valid_pts
      for sol in tqdm(sols.values(), ncols=80):
        data_i = self.get_data(sol, nb_pts)
        data = self.stack_data(data, data_i, k)
      # Save data in a 'csv' file
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
        path=self.path_to_samples + "/data/test/",
        dataid="test",
        init_sols=init_sols["test"]
      )
      # Saving 0D data
      for (i, sol) in tqdm(sols.items(), ncols=80):
        ikey = "case_" + i
        data[ikey] = self.get_data(sol)
        self.save_data(filename=path+f"/{ikey}", data=data[ikey])
    return data
