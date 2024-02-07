import os
import json
import numpy as np
import pycomet as pcm
import tensorflow as tf

from pyDOE import lhs
from tqdm import tqdm
from silx.io.dictdump import dicttoh5
from pycomet.sampling import DifferentialBased

# HyperNet package
os.environ.update({"HYPNET_CONFIGFILE": "./hypcfg.json"})
from hypernet import const
from hypernet import backend as hypbkd
from hypernet.apps import utils as hyputils


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
    # Solver
    self.solver = hyputils._get_solver(phyinp, self.kin.fun, self.kin.jac)
    # Species
    for (name, st) in self.sp_th.items():
      if st.specie.nb_comp > 1:
        self.molecule = name
      else:
        self.atom = name
    # Control variables
    # -----------------------------------
    self.is_edge_set = False
    self.is_node_set = False

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
    # Edges normalization
    self.compute_norm_qoi(data, "cond_attr")
    for k in ("train", "valid"):
      data[k] = self.apply_norm_qoi(data[k], "cond_attr")
    # Save data
    for data_type in data:
      filename = self.path_to_data + f"/{data_type}.hdf5"
      dicttoh5(data[data_type], filename)
    return data

  def get_data(self, inp, nb_pts=-1):
    # Extract number densities
    sol = self.compute_solution(inp, nb_pts)
    # Initializing edges/nodes
    self.set_edges()
    self.set_nodes()
    # Extract number densities and temperature
    na, nm = [sol[f"X_{s}"] * sol["n"] for s in self.sp_th.keys()]
    # Set dimensions
    nb_pts, nb_nodes = nm.shape
    # Compose node attributes
    node_n = np.tile(nm[0].reshape(1,-1), (nb_pts,1))
    node_T = np.tile(sol["T"].reshape(-1,1), (1,nb_nodes))
    node = np.stack([node_n, node_T], axis=-1)
    # Compose edge attributes
    edge = (self.rates * na[0]).reshape(1,-1,1)
    edge = np.tile(edge, (nb_pts,1,1))
    # Compose conditioning input attributes
    cond_t = sol["t"].reshape(-1,1)
    cond_t = np.tile(cond_t, (1,nb_nodes))
    cond_n = na[0].reshape(1,1)
    cond_n = np.tile(cond_n, (nb_pts,nb_nodes))
    cond = np.stack([cond_t, cond_n], axis=-1)
    cond = np.log(cond + self.time_eps)
    # Compose target
    targ = np.expand_dims(nm, axis=-1)
    # Compose data
    return {
      "node_attr": node,
      "edge_attr": edge,
      "cond_attr": cond,
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

  # Normalization
  # -----------------------------------
  def compute_norm_qoi(self, data, identifier):
    path = self.path_to_data + "/norm/"
    os.makedirs(path, exist_ok=True)
    # Compute normalizing QoI
    data = [data[k][identifier] for k in ("train", "valid")]
    data = np.concatenate(data, axis=0)
    qois = (
      (f"{identifier}_mean", np.mean),
      (f"{identifier}_std", np.std)
    )
    for (qoi, fun) in qois:
      value = fun(data, axis=(0,1))
      setattr(self, qoi, value.reshape(1,1,-1))
      np.savetxt(path + f"/{qoi}.txt", value, fmt='%.10e')

  def apply_norm_qoi(self, data, identifier):
    data[identifier] -= getattr(self, f"{identifier}_mean")
    data[identifier] /= getattr(self, f"{identifier}_std")
    return data

  # Testing data
  # -----------------------------------
  def testing_data(
    self,
    inp_space
  ):
    data = {}
    for (i, inp) in enumerate(tqdm(inp_space, ncols=80)):
      datai = self.get_data(inp)
      datai = self.apply_norm_qoi(datai, "cond_attr")
      data["case_" + str(i+1).zfill(3)] = datai
    dicttoh5(data, self.path_to_data + "/test.hdf5")
    return data

  # Compute solution
  # -----------------------------------
  def compute_solution(self, inp, nb_pts):
    # Preprocess input space: [X_atom, Ti, T, p]
    inp = {k: inp[i] for (i, k) in enumerate(self.inp_space_lbl)}
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
    T = np.array(inp["T"]).reshape(1)
    y0 = self.kin.get_init_sol(T=T)
    # Integrate
    sol = self.solver._solve(y0)
    sol = [hypbkd.atleast_2d(x, force_numpy=True) for x in sol]
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
    return t[idx], y[idx]

  # Nodes/edges static attributes
  # -----------------------------------
  def set_edges(self):
    if (not self.is_edge_set):
      # Compose adjacency matrix
      k = 0
      for reaction in self.kin.reactions:
        k = k + hypbkd.to_numpy(reaction.fwd.k).squeeze()
      # Identify non-zero links
      k_idx = np.where(k > 0)
      # Construct edges
      self.rates = k[k_idx].squeeze()
      self.fwd_edge_index = np.array(k_idx)
      # Saving edges
      filename = self.path_to_data + "/fwd_edge_index.txt"
      np.savetxt(filename, self.fwd_edge_index, fmt="%d")
      # Edge is set
      self.is_edge_set = True

  def set_nodes(self):
    if (not self.is_node_set):
      sp = self.sp_th[self.molecule].specie
      # Probability matrix
      P = hypbkd.to_numpy(sp.mappings["lev"])
      # StS energies
      ei = [hypbkd.to_numpy(sp.lev[k]) for k in ("ERot", "EVib")]
      ei = np.stack(ei, axis=1) / const.eV_to_J
      # CG energies
      eg = (np.matmul(ei.T, P) / np.sum(P, axis=0, keepdims=True)).T
      # Saving nodes
      filename = self.path_to_data + "/node_energies.txt"
      np.savetxt(filename, eg, fmt="%.8e")
      # Node is set
      self.is_node_set = True
