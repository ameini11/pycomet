import numpy as np
import tensorflow as tf
import tensorflow.python.keras.callbacks as cb

from pycomet import utils
from silx.io.dictdump import dicttoh5
from tensorflow.keras import backend as tf_bkd
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.callbacks.GradientsTracker')
class GradientsTracker(cb.Callback):

  # Initialization
  # ===================================
  def __init__(
    self,
    data=None,
    max_samples=20000,
    freq=50,
    histogram_freq=0,
    save=False,
    log_dir=None,
    train_dir=None,
    h5filename="grads",
    **kwargs
  ):
    super(GradientsTracker, self).__init__(**kwargs)
    # Dataset
    self.data = data
    self.max_samples = max_samples
    # Computing and saving/writing gradients
    self.freq = freq
    self.histogram_freq = histogram_freq
    self.save = save
    self.log_dir = log_dir
    self.train_dir = train_dir
    self.h5filename = self.train_dir + "/" + h5filename + ".h5"

  def set_model(self, model):
    super(GradientsTracker, self).set_model(model)
    self._weights_names = tf.nest.map_structure(
      lambda w: w.name.replace(":", "_"), self.model.trainable_weights
    )
    if (self.log_dir is not None):
      self._writer = tf.summary.create_file_writer(self.log_dir + "/train")
    else:
      self.histogram_freq = 0

  # Callback methods
  # ===================================
  # 'on_train_begin' method
  # -----------------------------------
  def on_train_begin(self, logs={}):
    self._compute_gradients = self._make_compute_gradients()

  def _make_compute_gradients(self):
    grads = tf_bkd.gradients(
      self.model.total_loss, self.model.trainable_weights
    )
    inputs = self.model._feed_inputs \
      + self.model._feed_targets \
      + self.model._feed_sample_weights
    return tf_bkd.function(inputs, grads)

  # 'on_epoch_end' method
  # -----------------------------------
  def on_epoch_end(self, epoch, logs={}):
    if (epoch % self.freq == 0):
      # Sample dataset
      data = utils.unpack_list(self._sample_dataset(), nb_elem=3)
      # Compute gradients
      grads = self._compute_gradients(data)
      grads = utils.make_dict(self._weights_names, grads)
      # Compute gradients mean
      grads_mean = tf.nest.map_structure(self._aggr_grads, grads)
      # Save gradients to hdf5
      if self.save:
        self._save_grads(epoch, grads, grads_mean)
      # Write gradients to TensorBoard
      if (self.histogram_freq and (epoch % self.histogram_freq == 0)):
        self._log_grads(epoch, grads, grads_mean)

  def _sample_dataset(self):
    return self.data.sample_data(
      identifier="train", max_samples=self.max_samples
    )

  def _aggr_grads(self, gradients):
    return np.mean(np.abs(gradients))

  def _save_grads(
    self,
    epoch,
    gradients,
    gradients_mean
  ):
    dicttoh5(
      treedict={
        str(epoch).zfill(6): {
          "tensors": gradients,
          "mean": gradients_mean
        }
      },
      h5file=self.h5filename,
      mode='a'
    )

  def _log_grads(
    self, 
    epoch,
    gradients,
    gradients_mean
  ):
    with self._writer.as_default():
      with tf.summary.record_if(True):
        for name in self._weights_names:
          tf.summary.histogram(
            name="grad_" + name,
            data=gradients[name],
            step=epoch
          )
          tf.summary.scalar(
            name="grad_mean_" + name,
            data=gradients_mean[name],
            step=epoch
          )
        self._writer.flush()
