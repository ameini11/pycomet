import time
import json
import tensorflow.python.keras.callbacks as cb

from keras.utils import io_utils
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.callbacks.Timer")
class Timer(cb.Callback):
  """Stop training when training time reaches the threshold.
  This Timer starts after the first call of `on_train_begin`."""

  # Initialization
  # ===================================
  def __init__(
    self,
    available_time
  ):
    super(Timer, self).__init__()
    self.threshold = available_time # seconds
    self.t_start = None

  # Callback methods
  # ===================================
  def on_train_begin(self, logs=None):
    if (self.t_start is None):
      self.t_start = time.time()

  def on_epoch_end(self, epoch, logs=None):
    if ((time.time() - self.t_start) > self.threshold):
      time_used = (time.time() - self.t_start) / 60
      io_utils.print_msg(
        "Stop training as time used up:"
        "- Time used: {:.1f} mins"
        "- Epoch used: {}".format(time_used, epoch)
      )
      self.model.stop_training = True


@keras_export("keras.callbacks.Terminator")
class Terminator(cb.Callback):

  # Initialization
  # ===================================
  def __init__(
    self,
    filename,
    freq=1
  ):
    super(Terminator, self).__init__()
    self.filename = filename
    # Generate file
    train_ctrl = {"stop_training": False}
    with open(self.filename, "w") as file:
      json.dump(train_ctrl, file, indent=4)
    # Frequency
    self.freq = freq

  # Callback methods
  # ===================================
  def on_epoch_end(self, epoch, logs=None):
    if (epoch % self.freq == 0):
      with open(self.filename) as file:
        train_ctrl = json.load(file)
      if train_ctrl["stop_training"]:
        io_utils.print_msg("\nTraining manually stopped.")
        self.model.stop_training = True
