import tensorflow as tf

from pycomet import utils
from tensorflow.python.keras import callbacks as cb
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.callbacks.VariablesTracker')
class VariablesTracker(cb.CSVLogger):

  # Initialization
  # ===================================
  def __init__(
    self,
    var_names=[],
    log_dir=None,
    filename=''
  ):
    super(VariablesTracker, self).__init__(filename)
    self.var_names = tf.nest.flatten(var_names)
    self.log_dir = log_dir
    self.tracked_variables = {}

  # Callback methods
  # ===================================
  def set_model(self, model):
    super(VariablesTracker, self).set_model(model)
    # Set the writer
    self._vars_writer = None
    if (self.log_dir is not None):
      _logdir = self.log_dir + '/scalars/tracked_variables'
      self._vars_writer = tf.summary.create_file_writer(_logdir)
    # Set the variables to be tracked
    variables = {}
    for var in self.model.trainable_variables:
      variables[var.name.split(':')[0]] = var
    for name in self.var_names:
      if (name not in variables.keys()):
        utils.warning(
          f"Not found 'tf.Variable' with name '{name}' in " \
          "'model.trainable_variables': it can't be tracked."
        )
        self.var_names.remove(name)
      else:
        self.tracked_variables[name] = variables[name]

  def on_epoch_end(self, epoch, logs={}):
    logs = {}
    for (k, v) in self.tracked_variables.items():
      logs[k] = v.numpy()
      self._log_variable(epoch, k, v.numpy())
    super(VariablesTracker, self).on_epoch_end(epoch, logs)

  def _log_variable(self, epoch, name, value):
    if (self._vars_writer is not None):
      with self._vars_writer.as_default():
        tf.summary.scalar(name=name, data=value, step=epoch)
