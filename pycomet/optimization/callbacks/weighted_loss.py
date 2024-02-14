import sys
import copy
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.callbacks as cb
import tensorflow.python.keras.initializers as tf_init

from tqdm import tqdm
from pycomet import utils
from pycomet import backend as bkd
from pycomet import operations as ops
from tensorflow.python.eager import backprop
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.callbacks.BaseWeightsAdapter')
class BaseWeightsAdapter(cb.Callback):

  # Initialization
  # ===================================
  def __init__(
    self,
    freq=1,
    data=None,
    log_dir=None,
    multi_obj_loss_weights=None,
    **kwargs
  ):
    super(BaseWeightsAdapter, self).__init__(**kwargs)
    self.freq = int(np.clip(freq, 1, None))
    self.name = utils.get_class_name(self)
    self.multi_obj_loss_weights = multi_obj_loss_weights
    # Define data container
    self.data = data
    if (not self.data.multi_obj_loss):
      utils.raise_err(
        ValueError,
        f"The callback '{self.name}' is valid " \
        "only with multi-objective loss function."
      )
    # Define 'logs' directory
    self.log_dir = log_dir
    # Control variables
    self._print = True

  # Callback methods
  # ===================================
  def set_model(self, model):
    super(BaseWeightsAdapter, self).set_model(model)
    # Set the writer
    self._weights_writer = None
    c1 = (self.log_dir is not None)
    c2 = (self.multi_obj_loss_weights is not None)
    if (c1 and c2):
      self._weights_writer = {}
      for obj in self.multi_obj_loss_weights:
        _logdir = self.log_dir + '/scalars/multi_obj_loss_weights/' + obj
        self._weights_writer[obj] = tf.summary.create_file_writer(_logdir)

  def on_epoch_end(self, epoch, logs={}):
    if ((self.multi_obj_loss_weights is not None) and self._print):
      self._log_weights(epoch, logs)

  # Handling weights methods
  # ===================================
  def _init_weights(self):
    if (self.multi_obj_loss_weights is None):
      utils.warning(
        "No 'multi_obj_loss_weights' have been provided to " \
        f"'{self.name}': the initial value of 1 will be applied."
      )
      self.multi_obj_loss_weights = {}
      for obj in self.model.objectives['train']:
        self.multi_obj_loss_weights[obj] = tf.ones(1, dtype=bkd.get_floatx())
    elif isinstance(self.multi_obj_loss_weights, (list,tuple)):
      self.multi_obj_loss_weights = utils.make_dict(
        self.model.objectives['train'], self.multi_obj_loss_weights
      )
    elif isinstance(self.multi_obj_loss_weights, dict):
      pass
    else:
      utils.raise_err(
        ValueError,
        f"'multi_obj_loss_weights' passed into '{self.name}' " \
        "has to be a list or a dictionary."
      )
    for (k, w) in self.multi_obj_loss_weights.items():
      w = tf.reshape(ops.to_tensor(w), shape=(-1,1))
      self.multi_obj_loss_weights[k] = w

  def _log_weights(self, epoch, logs):
    weights = tf.nest.map_structure(
      lambda x: np.squeeze(x.numpy()), self.multi_obj_loss_weights
    )
    logs.update({obj + '_loss_weight': w for (obj, w) in weights.items()})
    if (self._weights_writer is not None):
      for (obj, writer) in self._weights_writer.items():
        with writer.as_default():
          tf.summary.scalar(
            name='epoch_' + obj + '_loss_weight',
            data=weights[obj],
            step=epoch
          )

  def _check_weights(self, weights=None):
    if (weights is None):
      weights = self.multi_obj_loss_weights
    nan = [tf.math.is_nan(w) for w in tf.nest.flatten(weights)]
    inf = [tf.math.is_inf(w) for w in tf.nest.flatten(weights)]
    if (any(nan) or any(inf)):
      print("")
      utils.warning(
        "One or multiple weights evaluated in " \
        f"'{self.name}' class are 'inf' or 'nan'."
      )
      if (self.multi_obj_loss_weights is not None):
        utils.print_fn("Here the 'multi_obj_loss_weights' values:", lev=1)
        for (k, v) in self.multi_obj_loss_weights.items():
          utils.print_fn(f"  - '{k}': {v}", lev=1)
        utils.print_fn("The training will be stopped.", lev=1)
        self.model.stop_training = True

  # Gradient descent methods
  # ===================================
  def on_grad_step_begin(self, sample_weight=None):
    return self.multi_obj_loss_weights

  def on_grad_step_eval(self, sample_weight=None):
    return self.multi_obj_loss_weights

  def on_grad_step_end(
    self,
    loss=0,
    sample_weight=None,
    sample_index=None,
    tape=None
  ):
    pass


@keras_export('keras.callbacks.ConstantWeightsAdapter')
class ConstantWeightsAdapter(BaseWeightsAdapter):

  # Initialization
  # ===================================
  def __init__(
    self,
    freq=1,
    data=None,
    log_dir=None,
    multi_obj_loss_weights=None,
    **kwargs
  ):
    super(ConstantWeightsAdapter, self).__init__(
      freq=freq,
      data=data,
      log_dir=log_dir,
      multi_obj_loss_weights=multi_obj_loss_weights,
      **kwargs
    )
    self._print = False

  # Callback methods
  # ===================================
  def on_train_begin(self, logs={}):
    super(ConstantWeightsAdapter, self).on_train_begin(logs=logs)
    self._init_weights()


@keras_export('keras.callbacks.EmpiricalWeightsAdapter')
class EmpiricalWeightsAdapter(BaseWeightsAdapter):
  """Adaptive loss weights technique for PINNs based on an empirical learning
  rate annealing algorithm.

  See:
    Sifan Wang, Yujun Teng, Paris Perdikaris
    'Understanding and mitigating gradient pathologies in physics-informed
    neural networks'
    https://arxiv.org/abs/2001.04536"""

  # Initialization
  # ===================================
  def __init__(
    self,
    freq=50,
    data=None,
    log_dir=None,
    multi_obj_loss_weights=None,
    alpha=0.8,
    max_samples=20000,
    method="mean",
    ref_obj="res",
    **kwargs
  ):
    super(EmpiricalWeightsAdapter, self).__init__(
      freq=freq,
      data=data,
      log_dir=log_dir,
      multi_obj_loss_weights=multi_obj_loss_weights,
      **kwargs
    )
    self.alpha = tf.cast(tf.clip_by_value(alpha, 0, 1), bkd.get_floatx())
    self.gradients = {}
    self.dataset = {}
    self.max_samples = max_samples
    # Gradients aggregation
    if (method == "max"):
      self.reduction = tf.math.reduce_max
    else:
      self.reduction = tf.math.reduce_mean
    # Reference objective data
    self.ref_obj = ref_obj
    # Control variables
    self._check_ref_obj = True
    self._update = True

  # Callback methods
  # ===================================
  def on_train_begin(self, logs={}):
    super(EmpiricalWeightsAdapter, self).on_train_begin(logs=logs)
    self._init_weights()
    cond = (self.ref_obj not in self.model.objectives['train'])
    if (cond and self._check_ref_obj):
      utils.warning(
        f"Network '{self.model.name}' hasn't any '{self.ref_obj}' objective "
        f"loss to be evaluated: '{self.name}' can't be applied."
      )
      self._update = False
      self.model.weighted_loss = None

  def on_epoch_begin(self, epoch, logs={}):
    if (self._update and (epoch % self.freq == 0)):
      self._sample_dataset()
      self._eval_loss_gradients()
      self._update_multi_obj_loss_weights()
      self._check_weights(self.multi_obj_loss_weights)

  # Handling weights methods
  # ===================================
  def _sample_dataset(self):
    self.dataset = self.data.sample_data(
      identifier="train", max_samples=self.max_samples
    )

  def _eval_loss_gradients(self):
    losses = {}
    with backprop.GradientTape(persistent=True) as tape:
      for obj in self.model.objectives['train']:
        x, y = self.dataset[obj][:2]
        # Get predictions
        y_pred = self.model.multi_obj_pred(x, objective=obj)
        # Evaluate each objective function loss
        losses[obj] = self.model.compiled_loss[obj](y, y_pred)
    # Evaluate each objective function loss
    for obj in self.model.objectives['train']:
      self.gradients[obj] = tape.gradient(
        losses[obj], self.model.trainable_variables
      )

  def _update_multi_obj_loss_weights(self):
    # Aggragate reference gradients
    ref_grads = self._aggr_grads(self.gradients[self.ref_obj])
    # Evaluate new multi-objective loss weights
    new_weights = {}
    for (obj, grads) in self.gradients.items():
      if (obj == self.ref_obj):
        new_weights[obj] = self.multi_obj_loss_weights[self.ref_obj]
        continue
      new_weights[obj] = ref_grads / self._aggr_grads(grads)
      new_weights[obj] *= self.multi_obj_loss_weights[self.ref_obj]
    # Update multi-objective loss weights
    self.multi_obj_loss_weights = tf.nest.map_structure(
      lambda old, new: self.alpha*new + (1 - self.alpha)*old,
      self.multi_obj_loss_weights,
      new_weights
    )

  def _aggr_grads(self, gradients):
    gradients = tf.nest.map_structure(
      lambda g: tf.reshape(tf.math.abs(g), -1), gradients
    )
    gradients = tf.concat(tf.nest.flatten(gradients), axis=0)
    return self.reduction(gradients)


@keras_export('keras.callbacks.InverseDirichletWeightsAdapter')
class InverseDirichletWeightsAdapter(EmpiricalWeightsAdapter):
  """Adaptive loss weights technique for PINNs based on Inverse-Dirichlet
  weighting."""

  # Initialization
  # ===================================
  def __init__(
    self,
    freq=50,
    data=None,
    log_dir=None,
    multi_obj_loss_weights=None,
    alpha=0.8,
    max_samples=20000,
    method="std",
    **kwargs
  ):
    super(InverseDirichletWeightsAdapter, self).__init__(
      freq=freq,
      data=data,
      log_dir=log_dir,
      multi_obj_loss_weights=multi_obj_loss_weights,
      alpha=alpha,
      max_samples=max_samples,
      method=method,
      ref_obj=None,
      **kwargs
    )
    if (method == "var"):
      self.reduction = tf.math.reduce_variance
    else:
      self.reduction = tf.math.reduce_std
    self._check_ref_obj = False

  # Handling weights methods
  # ===================================
  def _update_multi_obj_loss_weights(self):
    # Aggragate gradients
    sigmas = {}
    for (obj, grad) in self.gradients.items():
      sigmas[obj] = self._aggr_grads(grad)
    sigma_max = max(tf.nest.flatten(sigmas))
    # Evaluate new multi-objective loss weights
    new_weights = {}
    for (obj, sigma) in sigmas.items():
      new_weights[obj] = sigma_max/sigma
    # Update multi-objective loss weights
    self.multi_obj_loss_weights = tf.nest.map_structure(
      lambda old, new: self.alpha*new + (1 - self.alpha)*old,
      self.multi_obj_loss_weights,
      new_weights
    )

  def _aggr_grads(self, gradients):
    gradients = tf.nest.map_structure(lambda g: tf.reshape(g, -1), gradients)
    gradients = tf.concat(tf.nest.flatten(gradients), axis=0)
    return self.reduction(gradients)


# @keras_export('keras.callbacks.NTKWeightsAdapter')
# class NTKWeightsAdapter(EmpiricalWeightsAdapter):
#   """Adaptive loss weights technique for PINNs based on Neural Tangent Kernel
#   (NTK) theory.

#   See:
#     Sifan Wang, Xinling Yu, Paris Perdikaris
#     'When and why PINNs fail to train: A neural tangent kernel perspective'
#     https://arxiv.org/abs/2007.14527"""

#   def __init__(
#     self,
#     ode_loss_weights=None,
#     data=None,
#     freq=None,
#     log_dir=None,
#     max_samples=1000,
#     **kwargs
#   ):
#     super(NTKWeightsAdapter, self).__init__(
#       ode_loss_weights=ode_loss_weights,
#       data=data,
#       freq=freq,
#       log_dir=log_dir,
#       max_samples=max_samples,
#       **kwargs
#     )
#     self.pred_fn = {}

#   def on_train_begin(self, logs={}):
#     self._init_weights()
#     self.set_pred_fn()

#   def update(self):
#     self._sample_dataset()
#     ntk = self.get_ntk()
#     self.update_loss_weights(ntk)
#     self._check_weights()

#   # Handling predictions
#   ###########################################################################
#   def set_pred_fn(self):
#     for data_id in self.model.data_ids:
#       if data_id == 'res':
#         _pred_fn = self.model.residual
#       else:
#         _pred_fn = self.model.call
#       self.pred_fn[data_id] = _pred_fn

#   # Handling loss gradients
#   ###########################################################################
#   def flatten(self, x):
#     # Flatten array or list of arrays into 1D vector
#     if isinstance(x, (list,tuple)):
#       return list(itertools.chain(
#         *[tf.split(tf.reshape(i,[-1]), tf.size(i).numpy()) for i in x]
#       ))
#     else:
#       return tf.split(tf.reshape(x,[-1]), tf.size(x).numpy())

#   def get_ntk(self):
#     pred = {}
#     with backprop.GradientTape(persistent=True) as tape:
#       for data_id in self.model.data_ids:
#         x = self.dataset[data_id][0]
#         pred[data_id] = self.flatten(
#           self.pred_fn[data_id](
#             self.dataset[data_id][0], training=False
#           )
#         )
#     return self.eval_diag_ntk(pred, tape)

#   def eval_diag_ntk(self, pred, tape):
#     utils.print_fn("From 'NTKWeightsAdapter' callback:", lev=1)
#     ntk = {}
#     for data_id, pred_i in pred.items():
#       utils.print_fn(
#         "  - Evaluating NTK matrix for '{data_id}' data points", lev=1
#       )
#       ta = tf.TensorArray(
#         dtype=bkd.get_floatx(),
#         size=0,
#         dynamic_size=True,
#         clear_after_read=False
#       )
#       for i, elem in enumerate(tqdm(pred_i, file=sys.stdout)):
#         grad = tape.gradient(
#           elem,
#           self.model.trainable_variables,
#           unconnected_gradients='zero'
#         )
#         tr = tf.add_n([tf.reduce_sum(tf.multiply(g, g)) for g in grad])
#         ta = ta.write(i, tr)
#       ntk[data_id] = ta.stack()
#     return ntk

#   # Handling loss weights
#   ###########################################################################
#   def update_loss_weights(self, ntk):
#     trace = {
#       data_id: tf.reduce_sum(ntk_i).numpy() \
#         for data_id, ntk_i in ntk.items()
#     }
#     norm_grad = tf.reduce_sum(list(trace.values())).numpy()
#     for data_id in self.model.data_ids:
#       self.model.ode_loss_weights[data_id] = norm_grad / trace[data_id]


# @keras_export('keras.callbacks.SoftAttentionWeightsAdapter')
# class SoftAttentionWeightsAdapter(BaseWeightsAdapter):
#   """Base class for adaptive loss weights techniques."""

#   def __init__(
#     self,
#     data=None,
#     trainable=True,
#     rank=1,
#     limits=[1.,1.],
#     mask=None,
#     q=None,
#     freq=1,
#     log_dir=None,
#     **kwargs
#   ):
#     super(SoftAttentionWeightsAdapter, self).__init__(
#       data=data,
#       freq=freq,
#       log_dir=log_dir,
#       **kwargs
#     )
#     # Characterize weights tensors
#     self.trainable = trainable
#     self.rank = rank

#     # Define weights initializers
#     self.limits = limits
#     if isinstance(limits, list):
#       self.limits = {
#         obj: limits for obj in self.model.objectives['train']
#       }
#     if not isinstance(self.limits, dict):
#       utils.raise_err(ValueError,
#         "'limits' in '{}' has to be a 'dict' of [min, max] value "
#         "list for each objecitve function.".format(self.name)
#       )
#     self.initializer = {
#       obj: tf_init.RandomUniform(*l) for obj, l in self.limits.items()
#     }

#     # Define the self-adaptation mask function
#     if q is not None:
#       if any([q_i <= 0. for q_i in q]):
#         utils.raise_err(ValueError,
#           "'q' coefficents for the mask function in '{}' have "
#           "to be positive.".format(self.name)
#         )
#     else:
#       q = [1., 1.]
#     if mask == 'poly':
#       self.mask_fn = lambda x: q[0]*tf.pow(tf.math.abs(x), q[1])
#     elif mask == 'sigmoid':
#       self.mask_fn = lambda x: q[0]*tf.math.sigmoid(q[1]*x)
#     elif mask == 'exp':
#       self.mask_fn = lambda x: q[0]*tf.math.exp(q[1]*x)
#     else:
#       self.mask_fn = tf.square

#     self._update = True
#     self._init_weights()

#   # Callback methods --------------------------------------------------------
#   def on_train_begin(self, logs={}):
#     super(SoftAttentionWeightsAdapter, self).on_train_begin(logs=logs)
#     # Get indipendent model optimizer for each objective function
#     self.optimizer = {
#       obj: copy.deepcopy(self.model.optimizer) \
#         for obj in self.model.objectives['train']
#     }

#   def on_epoch_begin(self, epoch, logs={}):
#     if epoch % self.freq == 0:
#       self._update = True
#     else:
#       self._update = False

#   # Handling weights methods ------------------------------------------------
#   def _init_weights(self):
#     # Build sample weights
#     self.data.init_sample_weight(
#       self.initializer, self.trainable, self.rank
#     )
#     # Shuffle/Batch data
#     self.data.keep_indices = True
#     self.data.on_epoch_end()

#   # Gradient descent methods ------------------------------------------------
#   def on_grad_step_begin(self, sample_weight=None):
#     if self._update and sample_weight is not None:
#       return tf.nest.map_structure(
#         lambda sw: tf.Variable(sw, trainable=self.trainable),
#         sample_weight
#       )

#   def on_grad_step_eval(self, sample_weight=None):
#     if sample_weight is not None:
#       return tf.nest.map_structure(self.mask_fn, sample_weight)

#   def on_grad_step_end(
#     self,
#     loss=0.,
#     sample_weight=None,
#     sample_index=None,
#     tape=None
#   ):
#     if self._update and self.trainable and sample_weight is not None:
#       for obj, sw in sample_weight.items():
#         # Update batch sample weights
#         sw = tf.nest.flatten(sw)
#         grads = tf.nest.map_structure(
#           lambda x: -tape.gradient(loss, x), sw
#         )
#         self.optimizer[obj].apply_gradients(zip(grads, sw))
#         # Assign the updated sample weights to data container
#         tf.nest.map_structure(
#           lambda sw_old, sw_new: sw_old.scatter_nd_update(
#             sample_index[obj], sw_new
#           ),
#           self.data[obj][-1],
#           sw
#         )
