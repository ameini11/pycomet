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
    save=False,
    h5filename="res",
    log_dir=None,
    **kwargs
  ):
    super(GradientsTracker, self).__init__(**kwargs)
    # Dataset
    self.data = data
    self.max_samples = max_samples
    # Computing and saving/writing gradients
    self.freq = freq
    self.save = save
    self.h5filename = h5filename
    self.log_dir = log_dir
    if (self.log_dir is not None):
      self.writer = tf.summary.create_file_writer(self.log_dir + '/gradients')

  # Callback methods
  # ===================================
  # 'on_train_begin' method
  # -----------------------------------
  def on_train_begin(self, logs={}):
    self.weights_names = tf.nest.map_structure(
      lambda w: w.name, self.model.trainable_weights
    )
    self.compute_gradients = self.make_compute_gradients()

  def make_compute_gradients(self):
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
      data = utils.unpack_list(self.sample_dataset(), nb_elem=3)
      # Compute gradients
      grads = self.compute_gradients(data)
      grads = utils.make_dict(self.weights_names, grads)
      # Compute gradients mean
      grads_mean = tf.nest.map_structure(self.aggr_grads, grads)
      # Store gradients
      self.store_grads(epoch, grads, grads_mean)

  def sample_dataset(self):
    return self.data.sample_data(
      identifier="train", max_samples=self.max_samples
    )

  def aggr_grads(self, gradients):
    return tf.math.reduce_mean(tf.math.abs(gradients))

  def store_grads(self, epoch, gradients, gradients_mean):
    # Save gradients to hdf5
    if self.save:
      data = {
        str(epoch).zfill(6): {
          "tensors": gradients,
          "mean": gradients_mean
        }
      }
      dicttoh5(treedict=data, h5file=self.h5filename, mode='a')
    # Write gradients to TensorBoard
    if (self.log_dir is not None):
      with self.writer.as_default():
        for name in self.weights_names:
          tf.summary.histogram(
            name=name, data=gradients[name], step=epoch
          )
          tf.summary.scalar(
            name=name+'_mean', data=gradients_mean[name], step=epoch
          )
