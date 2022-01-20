import time
import tensorflow.python.keras.callbacks as CB

from tensorflow.python.util.tf_export import keras_export
from prode.utilities import utils


@keras_export('keras.callbacks.Timer')
class Timer(CB.Callback):
    """Stop training when training time reaches the threshold.
    This Timer starts after the first call of `on_train_begin`."""

    def __init__(
        self,
        available_time
    ):
        super(Timer, self).__init__()

        self.threshold = available_time * 60  # convert to seconds
        self.t_start = None

    def on_train_begin(self, logs=None):
        if self.t_start is None:
            self.t_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if (time.time() - self.t_start) > self.threshold:
            time_used = (time.time() - self.t_start) / 60
            utils.print_main(
                "Stop training as time used up:"
                "- Time used: {:.1f} mins"
                "- Epoch used: {}".format(time_used, epoch)
            )
            self.model.stop_training = True
