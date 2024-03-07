import os
import json
import numpy as np
import scipy as sp
import pandas as pd
import pycomet as pcm
import tensorflow as tf

from pyDOE import lhs
from tqdm import tqdm
from silx.io.dictdump import dicttoh5
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
    svd=None,
    free_atom=False,
    save_sol_only=False,
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
    self.processes = processes
    self.free_atom = free_atom
    self.svd, self.nb_coeff_svd = svd, 0
    if (self.svd is not None):
      self.svd = tf.nest.map_structure(np.loadtxt, self.svd)
      self.nb_coeff_svd = self.svd["phi"].shape[1] + 2
    # Save solutions for low-rank approximation calculations
    self.save_sol_only = save_sol_only
    # Control variables
    # -----------------------------------
    self._compute_svd_coeff = (self.free_atom and (self.svd is not None))
    self._is_state_set = False
    self._is_rate_set = False
    self._is_ols_set = False
    self._is_pf_set = False
    # Labeling data
    # -----------------------------------
    mol_comp = self.sp_th[self.molecule].specie.nb_comp
    mol_lbl = [self.molecule + f"_{i+1}" for i in range(mol_comp)]
    self.data_lbl = ["t", self.atom, self.molecule] \
      + [imol+"_0" for imol in mol_lbl] + [f"t_p{self.atom}"]
    if self._compute_svd_coeff:
      self.data_lbl += [f"w{i+1}" for i in range(self.nb_coeff_svd)]
    self.data_lbl += mol_lbl

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
      isol = self.get_solution(inp=inp, case_nb=i, nb_pts=nb_pts)
      if (not self.save_sol_only):
        idata = self.get_data(isol)
        # Collect data
        data_type = "train" if is_train else "valid"
        data = self.stack_data(data, idata, data_type)
    if (not self.save_sol_only):
      # Save data in a 'csv' file
      for (typek, datak) in data.items():
        filename = self.path_to_data + f"/{typek}"
        self.save_data(filename, datak, bounds=True)
    return data

  def get_solution(self, inp, case_nb=-1, nb_pts=-1):
    if self.save_sol_only:
      nb_pts = -1
      path = self.path_to_data + "/sols/"
      os.makedirs(path, exist_ok=True)
    sol = self.compute_solution(inp, nb_pts)
    if self.save_sol_only:
      case_nb = str(case_nb+1).zfill(6)
      dicttoh5(
        treedict=sol,
        h5file=path + f"/{case_nb}.hdf5",
        overwrite_data=True,
        create_dataset_args={"compression": "gzip"}
      )
    return sol

  def get_data(self, sol):
    # Initializing edges/nodes
    self.set_states()
    # Extract variables
    t, T, n = [sol[k].reshape(-1,1) for k in ("t", "T", "n")]
    na, nm = [sol[f"X_{s}"] * n for s in self.sp_th.keys()]
    # Inputs
    # -------------
    nb_pts = len(t)
    # > Time-dependent QoI
    qt = [t, na, np.sum(nm, axis=-1, keepdims=True)]
    qt = np.hstack(qt)
    # > Initial distribution
    nm0 = np.tile(nm[0].reshape(1,-1), (nb_pts,1))
    # > Correction factor
    tp = t * na * const.UKB * T
    tp = np.log(tp + 1e-10)
    if self._compute_svd_coeff:
      w = sol[f"n_{self.atom}_svd"].reshape(1,-1)
      w = np.tile(w, (nb_pts,1))
      f = np.hstack([tp, w])
    else:
      f = tp
    # > Compose inputs
    x = [qt, nm0, f]
    # Targets
    # -------------
    y = nm
    # Return data
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

  # Testing data
  # -----------------------------------
  def testing_data(
    self,
    inp_space
  ):
    path = self.path_to_data + "/test/"
    os.makedirs(path, exist_ok=True)
    data = {}
    for (i, inp) in enumerate(tqdm(inp_space, ncols=80)):
      ikey = "case_" + str(i+1).zfill(3)
      isol = self.get_solution(inp)
      data[ikey] = self.get_data(isol)
      self.save_data(filename=path+f"/{ikey}", data=data[ikey])
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
    # Set partition functions and rates
    self.set_pf()
    self.set_rates()
    # Integrate
    t = self.solver.t_eval
    y = self.solve(t, y0)
    sol = [t, y]
    # Get SVD coefficients for na decomposition
    if self._compute_svd_coeff:
      w = self.svd_coeff(*sol)
    # Sample solution in time
    if (nb_pts > 0):
      sol = self.sample_solution(*sol, nb_pts)
    # Get physical QoI
    sol = self.kin.prim_to_phys(*sol)
    sol["E"] = sol.pop("he")
    # Store SVD coefficients for na decomposition
    if self._compute_svd_coeff:
      sol[f"n_{self.atom}_svd"] = w
    return sol

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

  def sample_solution(self, t, y, nb_pts):
    idx = self.dyn_sampler.sample(
      x=np.log(t+self.time_eps).reshape(-1,1),
      y=y,
      nb_pts=nb_pts
    )
    return t[idx], y[idx]

  # SVD decomposition
  # -----------------------------------
  def svd_coeff(self, t, y):
    sol = self.kin.prim_to_phys(t, y)
    # Extract variables
    var = ("t", "T", "n", f"X_{self.atom}")
    t, T, n, Xa = [sol[k].reshape(-1) for k in var]
    # Compute QoI for SVD decomposition
    na = Xa * n
    pa = na * const.UKB * T
    tp = t * pa
    # Interpolate data onto bases grid
    y = self._svd_interpolate(tp, na) / const.UNA
    # Normalize data
    m = np.mean(y).reshape(1)
    s = np.std(y).reshape(1)
    y = (y - m) / s
    # Compute SVD coefficients using OLS
    w = self._svd_ols(y)
    # Return all the coefficients
    return np.concatenate([m,s,w]).reshape(-1)

  def _svd_interpolate(self, tp, na):
    f = sp.interpolate.interp1d(tp, na, kind='cubic')
    return f(self.svd["tp"])

  def _svd_ols(self, y):
    # Ordinary least squares (OLS) algorithm
    # > Set Gram matrix
    if (not self._is_ols_set):
      x = self.svd["phi"]
      # Compute Gram matrix
      gr = np.matmul(x.T, x)
      # Inverse of Gram matrix
      self.svd["gri"] = np.linalg.inv(gr)
      # OLS is set
      self._is_ols_set = True
    # > Moment matrix
    ym = np.matmul(self.svd["phi"].T, y)
    # > Compute coefficients
    w = np.matmul(self.svd["gri"], ym)
    return w

  # State/Group-specific static attributes
  # -----------------------------------
  def set_states(self):
    if (not self._is_state_set):
      sp = self.sp_th[self.molecule].specie
      # Probability matrix
      P = hypbkd.to_numpy(sp.mappings["lev"])
      # StS energies
      ei = [hypbkd.to_numpy(sp.lev[k]) for k in ("ERot", "EVib")]
      ei = np.stack(ei, axis=1)
      # CG energies
      eg = (np.matmul(ei.T, P) / np.sum(P, axis=0, keepdims=True)).T
      # CG normalized energies
      eg_norm = np.log(eg/const.eV_to_J)
      eg_std = np.std(eg_norm, axis=0, keepdims=True)
      eg_mean = np.mean(eg_norm, axis=0, keepdims=True)
      eg_norm = (eg_norm - eg_mean) / eg_std
      # Saving energies
      path = self.path_to_data + "/physics/states/"
      os.makedirs(path, exist_ok=True)
      np.savetxt(path + "/energies.txt", eg, fmt="%.8e")
      np.savetxt(path + "/energies_norm.txt", eg_norm, fmt="%.8e")
      # State is set
      self._is_state_set = True

  def set_rates(self):
    if (not self._is_rate_set):
      kef, keb = 0.0, 0.0
      for reaction in self.kin.reactions:
        if (reaction.param["process"] in (5,6)):
          kef = kef + reaction.fwd.k.squeeze()
          keb = keb + reaction.bwd.k.squeeze()
        elif (reaction.param["process"] == 2):
          kd = reaction.fwd.k.squeeze()
          kr = reaction.bwd.k.squeeze()
      ke = kef + keb.T
      ke = (ke.T - np.diag(np.sum(ke, axis=-1)))
      # Storing rates
      self.rates = {"ke": ke, "kef": kef, "keb": keb, "kd": kd, "kr": kr}
      # Saving rates
      path = self.path_to_data + "/physics/rates/"
      os.makedirs(path, exist_ok=True)
      for (k, rate) in self.rates.items():
        np.savetxt(path + f"/{k}.txt", rate.squeeze(), fmt="%.8e")
      # Rate is set
      self._is_rate_set = True

  def set_pf(self):
    if (not self._is_pf_set):
      # Saving partition functions
      path = self.path_to_data + "/physics/pf/"
      os.makedirs(path, exist_ok=True)
      for (name, st) in self.sp_th.items():
        Qi = st.pf.tot.Q.reshape(-1)
        np.savetxt(path + f"/{name}.txt", Qi, fmt="%.8e")
      # Rate is set
      self._is_pf_set = True
