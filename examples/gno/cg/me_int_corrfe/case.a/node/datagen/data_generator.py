import os
import json
import numpy as np
import scipy as sp
import pandas as pd
import pycomet as pcm
import tensorflow as tf

from pyDOE import lhs
from tqdm import tqdm
from pycomet.sampling import DifferentialBased

# HyperNet package
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
    processes=("exc",),
    free_atom=False,
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
    # Type of system
    self.free_atom = free_atom
    self.processes = processes
    # Labeling data
    mol_comp = self.sp_th[self.molecule].specie.nb_comp
    mol_lbl = [self.molecule + f"_{i+1}" for i in range(mol_comp)]
    self.data_lbl = ["t", self.atom+"_0"]
    self.data_lbl += [imol+"_0" for imol in mol_lbl]
    # self.data_lbl += ["t", self.atom+"_0"] + mol_lbl
    self.data_lbl += [f"t_p{self.atom}_0"] + mol_lbl
    # Get rates
    # -----------------------------------
    # Control variables
    # -----------------------------------
    self.is_state_set = False
    self.is_rate_set = False

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
    # Save data in a 'csv' file
    for (typek, datak) in data.items():
      filename = self.path_to_data + f"/{typek}"
      self.save_data(filename, datak, bounds=True)
    return data

  def get_data(self, inp, nb_pts=-1):
    # Extract number densities
    sol = self.compute_solution(inp, nb_pts)
    # Initializing edges/nodes
    self.set_states()
    # Extract variables
    t, T, n = [sol[k].reshape(-1,1) for k in ("t", "T", "n")]
    na, nm = [sol[f"X_{s}"] * n for s in self.sp_th.keys()]
    # Inputs
    n0 = np.concatenate([na[0], nm[0]]).reshape(1,-1)
    n0 = np.tile(n0, (len(t),1))
    if self.free_atom:
      raise NotImplementedError
    else:
      na0 = np.tile(na[0].reshape(1,1), (len(t),1))
      # f = np.hstack([t, na0])
      pa = na0 * const.UKB * T
      f = t * pa
    x = [t, n0, f]
    # Target
    y = nm
    # Compose data
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
    df.to_csv(filename+".csv", index=0, float_format='{:e}'.format)
    if bounds:
      bd = np.vstack([np.amin(data, axis=0), np.amax(data, axis=0)])
      df = pd.DataFrame(bd, columns=self.data_lbl)
      df.to_csv(filename+"_bounds.csv", index=0, float_format='{:e}'.format)

  # Testing data
  # -----------------------------------
  def testing_data(
    self,
    inp_space
  ):
    path = self.path_to_data + '/test/'
    os.makedirs(path, exist_ok=True)
    data = {}
    for (i, inp) in enumerate(tqdm(inp_space, ncols=80)):
      keyi = "case_" + str(i+1).zfill(3)
      data[keyi] = self.get_data(inp)
      self.save_data(filename=path+f"/{keyi}", data=data[keyi])
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
    self.kin._update_thermochem(T=T)
    self.set_rates()
    # Integrate
    t = self.solver.t_eval
    y = self.solve(t, y0)
    sol = [t, y]
    # Sample solution in time
    if (nb_pts > 0):
      sol = self.sample_solution(*sol, nb_pts)
    # Get physical QoI
    phys = self.kin.prim_to_phys(*sol)
    phys["E"] = phys.pop("he")
    return phys

  def sample_solution(self, t, y, nb_pts):
    idx = self.dyn_sampler.sample(
      x=np.log(t+self.time_eps).reshape(-1,1),
      y=y,
      nb_pts=nb_pts
    )
    return t[idx], y[idx]

  def solve(self, t, y0):
    res = sp.integrate.solve_ivp(
      fun=self.fun,
      t_span=[t[0],t[-1]],
      t_eval=t.squeeze(),
      y0=y0,
      method="LSODA",
      atol=1e-15,
      rtol=1e-12
    )
    return res.y.T

  def fun(self, t, y):
    f = np.zeros_like(y)
    # Extract temperature and number densities
    T = y[-1:]
    n = y[:-1] * const.UNA
    na, nm = n[0:1], n[1:]
    # Construct source terms
    wm = 0.0
    if ("exc" in self.processes):
      wm = wm + np.matmul(self.rates["ke"], nm*na)
    if ("dis" in self.processes):
      wd = self.rates["kd"]*nm*na - self.rates["kr"]*na**3
      wm = wm - wd
    # Compose source terms
    if self.free_atom:
      f[0] = 2*np.sum(wd)
    f[1:-1] = wm
    return f / const.UNA

  # State/Group-specific static attributes
  # -----------------------------------
  def set_states(self):
    if (not self.is_state_set):
      sp = self.sp_th[self.molecule].specie
      # Probability matrix
      P = hypbkd.to_numpy(sp.mappings["lev"])
      # StS energies
      ei = [hypbkd.to_numpy(sp.lev[k]) for k in ("ERot", "EVib")]
      ei = np.stack(ei, axis=1)
      # CG energies
      eg = (np.matmul(ei.T, P) / np.sum(P, axis=0, keepdims=True)).T
      # Saving states
      path = self.path_to_data + '/states/'
      os.makedirs(path, exist_ok=True)
      np.savetxt(path+"/energies.txt", eg, fmt="%.8e")
      # State is set
      self.is_state_set = True

  def set_rates(self):
    if (not self.is_rate_set):
      kef, keb = 0.0, 0.0
      for reaction in self.kin.reactions:
        if (reaction.param['process'] in (5,6)):
          kef = kef + reaction.fwd.k.squeeze()
          keb = keb + reaction.bwd.k.squeeze()
        elif (reaction.param['process'] == 2):
          kd = reaction.fwd.k.squeeze()
          kr = reaction.bwd.k.squeeze()
      ke = kef + keb.T
      ke = (ke.T - np.diag(np.sum(ke, axis=-1)))
      # Storing rates
      self.rates = {"ke": ke, "kef": kef, "keb": keb, "kd": kd, "kr": kr}
      # Saving rates
      path = self.path_to_data + '/rates/'
      os.makedirs(path, exist_ok=True)
      for (k, rate) in self.rates.items():
        np.savetxt(path+f"/{k}.txt", rate.squeeze(), fmt="%.8e")
      # Rate is set
      self.is_rate_set = True
