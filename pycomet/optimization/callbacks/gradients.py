import os
import h5py
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.callbacks as cb

from pycomet import utils
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.callbacks.GradientsTracker")
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
    h5filename="./grads.h5",
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
    self.h5filename = h5filename
    if os.path.exists(self.h5filename):
      os.remove(self.h5filename)

  # Callback methods
  # ===================================
  def set_model(self, model):
    super(GradientsTracker, self).set_model(model)
    self._weights_names = tf.nest.map_structure(
      lambda w: "_".join(w.name.replace(":", "_").split("/")[-2:]),
      self.model.trainable_weights
    )
    self._grads_writer = None
    if ((self.log_dir is not None) and (self.histogram_freq > 0)):
      _log_dir = self.log_dir + "/train/grads"
      self._grads_writer = tf.summary.create_file_writer(_log_dir)
    # Make function to compute gradients
    self._compute_gradients = tf.function(
      func=self._compute_gradients,
      autograph=True,
      jit_compile=True,
      reduce_retracing=True
    )

  def _compute_gradients(self, x, y, sample_weight):
    loss = self.model.compute_total_loss(x, y, sample_weight)
    return tf.gradients(loss, self.model.trainable_weights)

  def on_epoch_end(self, epoch, logs={}):
    if (epoch % self.freq == 0):
      # Sample dataset
      data = utils.unpack_list(self._sample_dataset(), nb_elem=3)
      # Compute gradients
      grads = self._compute_gradients(*data)
      grads = utils.make_dict(self._weights_names, grads)
      # Compute gradients mean
      grads_mean = tf.nest.map_structure(self._aggr_grads, grads)
      # Save gradients to hdf5
      if self.save:
        self._save_grads(epoch, grads, grads_mean)
      # Write gradients to TensorBoard
      if (self._grads_writer and (epoch % self.histogram_freq == 0)):
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
    with h5py.File(self.h5filename, "a") as file:
      for name in self._weights_names:
        self._save_grads_to_h5(file, "tensor", epoch, name, gradients[name])
        self._save_grads_to_h5(file, "mean", epoch, name, gradients_mean[name])

  def _save_grads_to_h5(self, file, group, epoch, name, grads):
    group = file.require_group(group)
    if (epoch == 0):
      shape = tuple([None] + list(grads.shape))
      grads = np.expand_dims(grads, axis=0)
      group.create_dataset(name, data=grads, maxshape=shape)
    else:
      group[name].resize(epoch+1, axis=0)
      group[name][-1] = grads

  def _log_grads(
    self, 
    epoch,
    gradients,
    gradients_mean
  ):
    with self._grads_writer.as_default():
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
        self._grads_writer.flush()
