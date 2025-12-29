from pycomet import utils
from keras.utils import io_utils
from keras.utils import tf_utils
from keras.distribute import distributed_file_utils
from tensorflow.python.keras import callbacks as cb
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.platform import tf_logging as logging


@keras_export("keras.callbacks.ModelCheckpoint")
class ModelCheckpoint(cb.ModelCheckpoint):

  # Initialization
  # ===================================
  def __init__(
    self,
    filepath,
    monitor="val_loss",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    options=None,
    initial_value_threshold=None,
    **kwargs
  ):
    super(ModelCheckpoint, self).__init__(
      filepath=filepath,
      monitor=monitor,
      verbose=verbose,
      save_best_only=save_best_only,
      save_weights_only=save_weights_only,
      mode=mode,
      save_freq=save_freq,
      options=options,
      initial_value_threshold=initial_value_threshold
    )

  # Callback methods
  # ===================================
  def _save_model(self, epoch, batch=None, logs={}):
    logs = logs or {}
    c1 = isinstance(self.save_freq, int)
    c2 = (self.epochs_since_last_save >= self.period)
    if (c1 or c2):
      # Block only when saving interval is reached.
      logs = tf_utils.sync_to_numpy_or_python_type(logs)
      self.epochs_since_last_save = 0
      filepath = self._get_file_path(epoch, batch, logs)
      try:
        if self.save_best_only:
          current = logs.get(self.monitor)
          if (current is None):
            logging.warning(
              "Can save best model only with %s available, skipping.",
              self.monitor
            )
          else:
            if self.monitor_op(current, self.best):
              if (self.verbose > 0):
                io_utils.print_msg(
                  f"\nEpoch {epoch + 1}: {self.monitor} improved " \
                  f"from {self.best:.5f} to {current:.5f}, " \
                  f"saving model to {filepath}"
                )
              self.best = current
              if self.save_weights_only:
                self.model.save_weights(
                  filepath,
                  overwrite=True,
                  options=self._options,
                )
              else:
                self.model.save(
                  filepath,
                  overwrite=True,
                  include_optimizer=False,
                  save_format="tf",
                  signatures=None,
                  options=self._options,
                  save_traces=True
                )
            else:
              if (self.verbose > 0):
                io_utils.print_msg(
                  f"\nEpoch {epoch + 1}: {self.monitor} did not improve " \
                  f"from {self.best:.5f}"
                )
        else:
          if (self.verbose > 0):
            io_utils.print_msg(
              f"\nEpoch {epoch + 1}: saving model to {filepath}"
            )
          if self.save_weights_only:
            self.model.save_weights(
              filepath,
              overwrite=True,
              options=self._options,
            )
          else:
            self.model.save(
              filepath,
              overwrite=True,
              include_optimizer=False,
              save_format="tf",
              signatures=None,
              options=self._options,
              save_traces=True
            )
        self._maybe_remove_file()
      except IsADirectoryError:
        utils.raise_err(
          IOError,
          "Please specify a non-directory filepath for " \
          "'ModelCheckpoint'. Filepath used is an existing " \
          f"directory: {filepath}"
        )
      except IOError as e:
        if ("is a directory" in str(e.args[0]).lower()):
          utils.raise_err(
            IOError,
            "Please specify a non-directory filepath for " \
            "'ModelCheckpoint'. Filepath used is an existing " \
            f"directory: {filepath}"
          )
        # Re-throw the error for any other causes.
        raise e

  def _get_file_path(self, epoch, batch, logs):
    """Returns the file path for checkpoint."""
    try:
      # `filepath` may contain placeholders such as
      # `{epoch:02d}`,`{batch:02d}` and `{mape:.2f}`. A mismatch between
      # logged metrics and the path's placeholders can cause formatting to
      # fail.
      if ((batch is None) or ("batch" in logs)):
        file_path = self.filepath.format(epoch=epoch+1, **logs)
      else:
        file_path = self.filepath.format(
          epoch=epoch+1, batch=batch+1, **logs
        )
    except KeyError as e:
      utils.raise_err(
        KeyError,
        f"Failed to format this callback filepath: '{self.filepath}'. " \
        f"Reason: {e}"
      )
    self._write_filepath = distributed_file_utils.write_filepath(
      file_path, self.model.distribute_strategy
    )
    return self._write_filepath
