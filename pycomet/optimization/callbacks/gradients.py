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
    data_container=None,
    multi_obj_loss_weights=None,
    freq=1,
    log_dir=None,
    **kwargs
  ):
    super(BaseWeightsAdapter, self).__init__()
    self.__name__ = utils.get_class_name(self)
    self.multi_obj_loss_weights = multi_obj_loss_weights
    self.freq = int(np.clip(freq, 1, None))
    self.print = True
    # Define data container
    self.data_container = data_container
    assert self.data_container is data_container
    # Define `logs` directory
    self.log_dir = log_dir

  # Callback methods
  # ===================================
  def on_train_begin(self, logs={}):
    utils.print_fn(
      f"The '{self.__name__}' callback is used " \
      "to balance the multi-objective loss.",
      lev=1
    )
    # Assign `weighted_loss` class to Keras model
    self.model.weighted_loss = self

  def on_epoch_end(self, epoch, logs={}):
    if ((self.multi_obj_loss_weights is not None) and self.print):
      self.print_weights(epoch, logs)

  # Handling weights methods
  # ===================================
  def init_weights(self):
    if (self.multi_obj_loss_weights is None):
      utils.warning(
        "No 'multi_obj_loss_weights' have been provided to " \
        f"'{self.__name__}': the initial value of 1 will be applied."
      )
      self.multi_obj_loss_weights = {
        obj: tf.ones(1, dtype=bkd.get_floatx()) \
          for obj in self.model.objectives['train']
      }
    elif isinstance(self.multi_obj_loss_weights, (list,tuple)):
      self.multi_obj_loss_weights = dict(
        zip(
          self.model.objectives['train'],
          self.multi_obj_loss_weights
        )
      )
    elif isinstance(self.multi_obj_loss_weights, dict):
      pass
    else:
      utils.raise_err(
        ValueError,
        f"'multi_obj_loss_weights' passed into '{self.__name__}' " \
        "has to be a list or a dictionary."
      )
    self.multi_obj_loss_weights = ops.convert_struct_to_tensor(
      self.multi_obj_loss_weights
    )
    for (k, w) in self.multi_obj_loss_weights.items():
      w = tf.reshape(w, shape=[-1,1]) if (w.shape.rank < 1) else w
      self.multi_obj_loss_weights[k] = w
    # Create file writer
    if (self.log_dir is not None):
      self.logs = {
        obj: tf.summary.create_file_writer(
          self.log_dir + '/scalars/multi_obj_loss_weights/' + obj
        ) for obj in self.multi_obj_loss_weights
      }

  def print_weights(self, epoch, logs):
    weights = tf.nest.map_structure(
      lambda x: np.squeeze(x.numpy()), self.multi_obj_loss_weights
    )
    logs.update({obj + '_loss_weight': w for (obj, w) in weights.items()})
    if (self.log_dir is not None):
      for (obj, log) in self.logs.items():
        with log.as_default():
          tf.summary.scalar(
            name='epoch_' + obj + '_loss_weight',
            data=weights[obj],
            step=epoch
          )

  def check_weights(self, weights=None):
    if (weights is None):
      weights = self.multi_obj_loss_weights
    nan = [tf.math.is_nan(w) for w in tf.nest.flatten(weights)]
    inf = [tf.math.is_inf(w) for w in tf.nest.flatten(weights)]
    if (any(nan) or any(inf)):
      print("")
      utils.warning(
        "One or multiple weights evaluated in " \
        "'{self.__name__}' class are 'inf' or 'nan'."
      )
      if (self.multi_obj_loss_weights is not None):
        utils.print_fn("Here the 'multi_obj_loss_weights' values:", lev=1)
        for (k, v) in self.multi_obj_loss_weights.items():
          utils.print_fn("  - '{}': {}".format(k,v), lev=1)
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
    data_container=None,
    multi_obj_loss_weights=None,
    freq=1,
    log_dir=None,
    **kwargs
  ):
    super(ConstantWeightsAdapter, self).__init__(
      data_container=data_container,
      multi_obj_loss_weights=multi_obj_loss_weights,
      freq=freq,
      log_dir=log_dir,
      **kwargs
    )
    self.print = False

  # Callback methods
  # ===================================
  def on_train_begin(self, logs={}):
    super(ConstantWeightsAdapter, self).on_train_begin(logs=logs)
    self.init_weights()


@keras_export('keras.callbacks.EmpiricalWeightsAdapter')
class EmpiricalWeightsAdapter(BaseWeightsAdapter):
  """Adaptive loss weights technique for PINNs based on an empirical learning
  rate annealing algorithm.

  See:
    Sifan Wang, Yujun Teng, Paris Perdikaris
    `Understanding and mitigating gradient pathologies in physics-informed
    neural networks`
    https://arxiv.org/abs/2001.04536"""

  # Initialization
  # ===================================
  def __init__(
    self,
    data_container=None,
    multi_obj_loss_weights=None,
    freq=1000,
    alpha=0.8,
    max_samples=20000,
    method='mean',
    log_dir=None,
    ref_obj="res",
    **kwargs
  ):
    super(EmpiricalWeightsAdapter, self).__init__(
      data_container=data_container,
      multi_obj_loss_weights=multi_obj_loss_weights,
      freq=freq,
      log_dir=log_dir,
      **kwargs
    )
    self.alpha = tf.cast(tf.clip_by_value(alpha, 0, 1), bkd.get_floatx())
    self.gradients = {}
    self.dataset = {}
    self.max_samples = max_samples
    # Gradients aggregation
    self._mean = lambda x: tf.reduce_mean(
      [tf.reduce_mean(tf.math.abs(i)) for i in x]
    )
    self._max = lambda x: tf.reduce_max(
      [tf.reduce_max(tf.math.abs(i)) for i in x]
    )
    self.aggr_grads = self._mean if (method == 'mean') else self._max
    # Control variables
    self._update = True
    # Reference objective data
    self.ref_obj = ref_obj
    self.check_ref_obj = True

  # Callback methods
  # ===================================
  def on_train_begin(self, logs={}):
    super(EmpiricalWeightsAdapter, self).on_train_begin(logs=logs)
    self.init_weights()
    c1 = (self.ref_obj not in self.model.objectives['train'])
    c2 = self.check_ref_obj
    if (c1 and c2):
      utils.warning(
        f"Network '{self.model.name}' hasn't any '{self.ref_obj}' objective "
        f"loss to be evaluated: '{self.__name__}' can't be applied."
      )
      self._update = False
      self.model.weighted_loss = None

  def on_epoch_begin(self, epoch, logs={}):
    if (self._update and (epoch % self.freq == 0)):
      self.update_weights()

  # Handling weights methods
  # ===================================
  def update_weights(self):
    self.sample_dataset()
    self.eval_loss_gradients()
    self.update_multi_obj_loss_weights()
    self.check_weights(self.multi_obj_loss_weights)

  def sample_dataset(self):
    for (obj, dset) in self.data_container.data.items():
      nb_samples = self.data_container.nb_samples[obj]
      indices = np.random.permutation(nb_samples)
      if (nb_samples > self.max_samples):
        indices = np.random.choice(
          indices, self.max_samples, replace=False
        )
      self.dataset[obj] = tf.nest.map_structure(
        lambda z: tf.gather(z, indices, axis=0), dset
      )

  def eval_loss_gradients(self):
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


for g, v in grads_and_vars:
  tf.summary.histogram(v.name, v)
  tf.summary.histogram(v.name + '_grad', g)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('train_log_layer', tf.get_default_graph())

...

_, summary = sess.run([train_op, merged], feed_dict={I: 2*np.random.rand(1, 1)-1})
if i % 10 == 0:
  writer.add_summary(summary, global_step=i)


    def _log_weights(self, epoch):
        """Logs the weights of the Model to TensorBoard."""
        with self._train_writer.as_default():
            with tf.summary.record_if(True):
                for layer in self.model.layers:
                    for weight in layer.weights:
                        weight_name = weight.name.replace(":", "_")
                        # Add a suffix to prevent summary tag name collision.
                        histogram_weight_name = weight_name + "/histogram"
                        tf.summary.histogram(
                            histogram_weight_name, weight, step=epoch
                        )
                self._train_writer.flush()




  def compute_gradients(self):
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(self.dataset)
    # Run forward pass
    with tf.GradientTape() as tape:
      y_pred = self.model(x, training=True)
      loss = self.model.compute_loss(x, y, y_pred, sample_weight)
    self.model._validate_target_and_loss(y, loss)
    # Run backwards pass
    return self.model.optimizer.compute_gradients(
      loss=loss, var_list=self.trainable_variables, tape=tape
    )



w = tf.summary.create_file_writer('test/logs')
with w.as_default():
    tf.summary.histogram("activations", tf.random.uniform([100, 50]), step=0)
    tf.summary.histogram("initial_weights", tf.random.normal([1000]), step=0)



import time
import json
import tensorflow.python.keras.callbacks as cb
from tensorflow.keras import backend as tf_bkd

from keras.utils import io_utils
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.callbacks.GradientsTracker')
class GradientsTracker(cb.Callback):

  # Initialization
  # ===================================
  def __init__(
    self,
    available_time
  ):
    super(GradientsTracker, self).__init__()
    self.threshold = available_time # seconds
    self.t_start = None


    self._compute_gradients = self.make_compute_gradients()

  # Callback methods
  # ===================================
  def on_train_begin(self, logs={}):
    self._compute_gradients = self.make_compute_gradients()

  def on_epoch_end(self, epoch, logs={}):
    if (self._update and (epoch % self.freq == 0)):
      self.sample_dataset()
      grads = self.compute_gradients()

  def compute_gradients(self):
      data = data_adapter.unpack_x_y_sample_weight(self.dataset)
      grads = self._compute_gradients(data)
      return utils.make_dict(self.wnames, grads)

  def make_compute_gradients(self):
      grads = tf_bkd.gradients(
        self.model.total_loss, self.model.trainable_weights
      )
      inputs = self.model._feed_inputs \
        + self.model._feed_targets \
        + self.model._feed_sample_weights
      return tf_bkd.function(inputs, grads)
