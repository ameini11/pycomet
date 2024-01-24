import os
import numpy as np
import pycomet as pcm
import tensorflow as tf

from pyDOE import lhs
from scipy.integrate import solve_ivp
from silx.io.dictdump import dicttoh5


class DataGenerator(pcm.data.BaseDataGenerator):

  # Initialization
  # ===================================
  def __init__(
    self,
    scenarios,
    nb_time_pts,
    validation_split,
    test_cases,
    inp_space,
    inp_space_xform,
    autoregressive,
    *args,
    **kwargs
  ):
    super(DataGenerator, self).__init__()
    self.data_dir = "./data/"
    os.makedirs(self.data_dir, exist_ok=True)
    # Data quantities
    # -----------------------
    # Training/Validation
    self.scenarios = scenarios
    self.nb_time_pts = nb_time_pts
    self.validation_split = validation_split
    # Testing
    self.test_cases = test_cases
    # Input space variables
    self.inp_space = np.vstack(list(inp_space.values())).T
    self.inp_space_xform = inp_space_xform
    # Model
    # -----------------------
    self.autoregressive = autoregressive
    # Integrator options
    # -----------------------
    self.options = {
      "atol": 1e-9,
      "rtol": 1e-6,
      "inith": 1e-5,
      "maxh": 1e3
    }
    self.rates = np.array([4e-2, 3e7, 1e4])
    self.t_span = [0, 1e5]
    self.t_eval = np.logspace(-5, 5, num=self.nb_time_pts)
    # Chemical network
    # -----------------------
    # Log transform + Normalization
    k = np.log(self.rates)
    k = (k - np.mean(k)) / np.std(k)
    self.edges = np.array([
      [0, 1, k[0],    0],
      [1, 0, k[2],    0],
      [1, 2, k[1], k[2]],
      [2, 0, k[2],    0]
    ])

  def get_data_src(
    self
  ):
    data_src = {}
    # Sample input space
    inp_space = self.compute_inp_space()
    # Generate training/validation data
    data_src.update(self.training_data(inp_space[0]))
    # Generate testing data
    if (self.test_cases > 0):
      data_src['test'] = self.testing_data(inp_space[1])
    # Saving edges
    filename = self.data_dir + "/edge_index.txt"
    np.savetxt(filename, self.edges[:,:2], fmt='%d')
    return data_src

  # Initial conditions
  # ===================================
  def compute_inp_space(
    self
  ):
    dim = self.inp_space.shape[1]
    samples = self.scenarios * 2 + self.test_cases
    # Transform input space - Forward
    inp_space_lim = np.zeros_like(self.inp_space)
    for (i, xform) in enumerate(self.inp_space_xform.values()):
      inp_space_lim[:,i] = xform[0](self.inp_space[:,i])
    # Contructing design matrix
    x_norm = lhs(dim, samples)
    x = x_norm * (inp_space_lim[1]-inp_space_lim[0]) + inp_space_lim[0]
    # Transform input space - Backward
    for (i, xform) in enumerate(self.inp_space_xform.values()):
      x[:,i] = xform[1](x[:,i])
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

  # Solution
  # ===================================
  def solution(self, y0):
    sol = solve_ivp(
      fun=self.fun,
      t_span=self.t_span,
      y0=y0,
      method="LSODA",
      t_eval=self.t_eval,
      args=self.rates,
      first_step=self.options["inith"],
      max_step=self.options["maxh"],
      atol=self.options["atol"],
      rtol=self.options["rtol"],
      jac=self.jac,
      vectorized=False
    )
    return sol.t.reshape(-1,1), sol.y.T

  def fun(self, t, y, *arg):
    dy = np.zeros(3)
    dy[0] = - arg[0]*y[0] + arg[2]*y[1]*y[2]
    dy[1] = arg[0]*y[0] - arg[1]*y[1]**2 - arg[2]*y[1]*y[2]
    dy[2] = arg[1]*y[1]**2
    return dy

  def jac(self, t, y, *arg):
    J = [
      [- arg[0],                   arg[2]*y[2],   arg[2]*y[1]],
      [  arg[0], - 2*arg[1]*y[1] - arg[2]*y[2], - arg[2]*y[1]],
      [       0,                 2*arg[1]*y[1],             0]
    ]
    return np.array(J)

  # Training/Validation data
  # ===================================
  def training_data(
    self,
    inp_space
  ):
    data = {}
    for (i, y0) in enumerate(inp_space):
      # Generate data
      data_i = self.get_data(y0)
      # Collect data
      data_type = 'train' if ((i % 2) == 0) else 'valid'
      data = self.stack_data(data, data_i, data_type)
    # Normalization
    dref = data['train'][0][0]
    self.xmean, self.xstd = np.mean(dref, axis=0), np.std(dref, axis=0)
    np.savetxt(self.data_dir + "/xmean.txt", self.xmean, fmt='%.10e')
    np.savetxt(self.data_dir + "/xstd.txt",  self.xstd,  fmt='%.10e')
    # Save data
    for data_type in data:
      data[data_type] = self.store_data(data[data_type])
      filename = self.data_dir + data_type + ".hdf5"
      dicttoh5(data[data_type], filename)
    return data

  def get_data(self, y0):
    t, y = self.solution(y0)
    n = len(t)
    # Compose edge attributes
    edge = np.expand_dims(self.edges[:,2:],0)
    # Compose data
    if self.autoregressive:
      edge = np.tile(edge, (n-1,1,1))
      return [[y[:-1], edge], y[1:]]
    else:
      edge = np.tile(edge, (n,1,1))
      return [[np.tile(y0, (n,1)), edge, t], y]

  def stack_data(self, data, data_new, data_type):
    if (data_type not in data):
      data[data_type] = data_new
    else:
      data[data_type] = tf.nest.map_structure(
        lambda x, y: np.vstack((x,y)),
        data[data_type],
        data_new
      )
    return data

  def store_data(self, data):
    x = (data[0][0] - self.xmean) / self.xstd
    y = (data[1] - self.xmean) / self.xstd
    if self.autoregressive:
      return {
        "node_attr": x[...,None],     # Input node attributes
        "edge_attr": data[0][1],      # Input edge attributes
        "target": y[...,None]         # Output node attributes
      }
    else:
      return {
        "node_attr": x[...,None],     # Input node attributes
        "edge_attr": data[0][1],      # Input edge attributes
        "cond_attr": data[0][2],      # Conditioning inputs - Time
        "target": y[...,None]         # Output node attributes
      }

  # Testing data
  # ===================================
  def testing_data(
    self,
    inp_space
  ):
    data = {}
    for (i, y0) in enumerate(inp_space):
      data['case_' + str(i+1).zfill(3)] = self.store_data(self.get_data(y0))
    dicttoh5(data, self.data_dir + "/test.hdf5")
    return data
