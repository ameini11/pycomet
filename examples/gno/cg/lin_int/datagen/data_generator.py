import os
import json
import numpy as np
import pycomet as pcm
import tensorflow as tf

from pyDOE import lhs
from tqdm import tqdm
from silx.io.dictdump import dicttoh5
from hypernet import backend as hypbkd
from hypernet.apps import utils as hyputils
from pycomet.sampling import DifferentialBased


class DataGenerator(pcm.data.BaseDataGenerator):

  # Initialization
  # ===================================
  def __init__(
    self,
    time_lim=[0,1],
    time_eps=1e-15,
    inp_space={},
    train_pts=32,
    scenarios=100,
    test_cases=10,
    validation_split=0.2,
    path_to_physics="./physics",
    grad_split=0.75,
    *args,
    **kwargs
  ):
    super(DataGenerator, self).__init__(*args, **kwargs)
    # Paths
    # ---------------------------------
    self.cwd = os.getcwd()
    self.path_to_data = self.cwd + "/data/"
    self.path_to_physics = path_to_physics
    os.makedirs(self.path_to_data, exist_ok=True)
    # Data quantities
    # ---------------------------------
    self.train_pts = train_pts
    self.valid_pts = round(validation_split*train_pts)
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
    with open(self.path_to_phy + "/inputs.json") as file:
      phyinp = json.load(file)
    # Species thermo
    self.sp_th = hyputils._get_specie_thermos(phyinp)
    # Mixture
    self.mix = hyputils._get_mixture(phyinp, self.sp_th)
    # Kinetics
    self.kin = hyputils._get_kinetics(phyinp, self.sp_th, self.mix)
    # Solver
    self.solver = hyputils._get_solver(phyinp, self.kin.fun, self.kin.jac)
    # Control variables
    # -----------------------------------
    self._edges_set = False

  # Generate data
  # ===================================
  def get_data_src(
    self
  ):
    data_src = {}
    # Sample input space
    pcm.utils.print_fn("Sample input space", lev=1)
    inp_space = self.sample_inp_space()
    # Generate training/validation data
    pcm.utils.print_fn("Training/validation data", lev=1)
    data_src.update(self.training_data(inp_space[0]))
    # Generate testing data
    if (self.test_cases > 0):
      pcm.utils.print_fn("Testing data", lev=1)
      data_src["test"] = self.testing_data(inp_space[1])
    # Saving edges
    pcm.utils.print_fn("Saving edges", lev=1)
    filename = self.path_to_data + "/edge_index.txt"
    np.savetxt(filename, self.edge_index, fmt="%d")
    return data_src

  # Initial conditions
  # -----------------------------------
  def sample_inp_space(self):
    dim = self.inp_space.shape[1]
    samples = self.scenarios * 2 + self.test_cases
    # Contructing design matrix
    x_norm = lhs(dim, samples)
    x = x_norm * (self.inp_space[1] - self.inp_space[0]) + self.inp_space[0]
    # Sorting w.r.t. the projection norm
    proj = np.dot(x_norm, np.ones((dim,1))).squeeze()
    inp_space = x[proj.argsort()]
    # Returning initial conditions
    if (self.test_cases > 0):
      idx_train, idx_test = pcm.utils.stratified_sampling(
        np.arange(samples), self.test_cases
      )
      return [inp_space[idx_train], inp_space[idx_test]]
    else:
      return [inp_space]

  # Training/Validation data
  # -----------------------------------
  def training_data(
    self,
    inp_space
  ):
    data = {}
    for (i, inp) in enumerate(tqdm(inp_space, ncols=80)):
      is_train = ((i % 2) == 0)
      # Generate data
      nb_pts = self.train_pts if is_train else self.valid_pts
      data_i = self.get_data(inp, nb_pts)
      # Collect data
      data_type = "train" if is_train else "valid"
      data = self.stack_data(data, data_i, data_type)
    # Save data
    for data_type in data:
      filename = self.path_to_data + f"/{data_type}.hdf5"
      dicttoh5(data[data_type], filename)
    return data

  def get_data(self, inp, nb_pts=-1):
    # Extract number densities
    sol = self.compute_solution(inp, nb_pts)
    # Initializing edges
    self.set_edges()
    # Extract number densities
    na, nm = [sol[f"X_{s}"] * sol["n"] for s in self.sp_th.keys()]
    # Compose edge attributes
    edge_t = sol["t"].reshape(-1,1,1)
    edge_k = (self.rates * na[0]).reshape(1,-1,1)
    edge_k = np.tile(edge_k, (nb_pts,1,1))
    edge = np.concatenate([edge_k, edge_t], axis=-1)
    edge = np.log(edge + self.time_eps)
    # Compose node attributes
    node = np.tile(nm[0].reshape(1,-1,1), (nb_pts,1,1))
    # Compose target
    targ = np.expand_dims(nm, axis=-1)
    # Compose data
    return {
      "node_attr": node,
      "edge_attr": edge,
      "target": targ
    }

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

  # Testing data
  # -----------------------------------
  def testing_data(
    self,
    inp_space
  ):
    data = {}
    for (i, inp) in enumerate(tqdm(inp_space, ncols=80)):
      data["case_" + str(i+1).zfill(3)] = self.get_data(inp)
    dicttoh5(data, self.path_to_data + "/test.hdf5")
    return data

  # Compute solution
  # -----------------------------------
  def compute_solution(self, inp, nb_pts):
    # Preprocess input space: [X_atom, Ti, T, p]
    inp = {k: inp[i] for (i, k) in self.inp_space_lbl}
    comp = {
      self.atom: inp[f"X_{self.atom}"],
      self.molecule: 1 - inp[f"X_{self.atom}"]
    }
    # Initialize solution
    self.mix.init_mix(
      x=comp,
      var="X",
      p_mix=inp["p"],
      T=inp["Ti"]
    )
    y0 = self.kin.get_init_sol(T=inp["T"])
    # Integrate
    sol = self.solver.solve(y0)
    sol = [hypbkd.to_numpy(x) for x in sol]
    # Sample solution in time
    if (nb_pts > 0):
      sol = self.sample_solution(*sol, nb_pts)
    # Get physical QoI
    phys = self.kin.prim_to_phys(*sol)
    phys["E"] = phys.pop("he")
    return phys

  def sample_solution(self, t, y, nb_pts):
    idx = self.dyn_sampler.sample(
      x=np.log(t+self.time_eps),
      y=y,
      nb_pts=nb_pts
    )
    if (0 not in idx):
      idx = np.insert(idx, 0, 0)
    return t[idx], y[idx]

  def set_edges(self):
    if (not self._edges_set):
      # Loop over reactions
      kf, kb = 0, 0
      for reaction in self.kin.reactions:
        kf = kf + hypbkd.to_numpy(reaction.fwd.k).squeeze()[0]
        kb = kb + hypbkd.to_numpy(reaction.bwd.k).squeeze()[0]
      # Compose adjacency matrix
      k = kf + kb.T
      # Identify non-zero links
      k_idx = np.where(k > 0)
      # Construct edges
      self.rates = k[k_idx]
      self.edge_index = np.array(k_idx)
      self._edges_set = True
