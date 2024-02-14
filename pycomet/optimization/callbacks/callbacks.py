import tensorflow as tf
import tensorflow.python.keras.callbacks as cb

from pycomet import utils
from . import _CB_LIST
from . import early_stopping
from . import gradients
from . import learning_rate
from . import model_ckpt
from . import plotting
from . import variable
from . import weighted_loss


def get_cb_list(model, callbacks=None):
  cb_list = {}
  if (callbacks is not None):
    for (identifier, kwargs) in callbacks.items():
      if (kwargs is not None):
        cb_list[identifier] = get_callback(model, identifier, kwargs)
  if ("terminator" not in cb_list.keys()):
    cb_list["terminator"] = get_callback(model, "terminator", {})
  return list(cb_list.values())

def get_callback(model, identifier, kwargs):
  if isinstance(identifier, str):
    if (identifier == "early_stopping"):
      return cb.EarlyStopping(**kwargs)
    elif (identifier == "lr_scheduler"):
      return utils.get_class(learning_rate, kwargs=kwargs)
    elif (identifier == "reduce_lr_on_plateau"):
      return learning_rate.ReduceLROnPlateau(**kwargs)
    elif (identifier == "lr_tracker"):
      return learning_rate.LearningRateTracker(
        filename=model.dir['train']+'/lr.csv',
        log_dir=model.dir['logs'],
        **kwargs
      )
    elif (identifier == "grads_tracker"):
      return gradients.GradientsTracker(
        data=model.data,
        log_dir=model.dir['logs'],
        train_dir=model.dir['train'],
        **kwargs
      )
    elif (identifier == "plotter"):
      return plotting.Plotter(
        test_data=model.data.test,
        train_dir=model.dir['train'],
        **kwargs
      )
    elif (identifier == "model_ckpt"):
      mc = []
      if ('formats' in kwargs):
        formats = tf.nest.flatten(kwargs['formats'])
      else:
        formats = ['h5']
      for k in formats:
        if (k in ('h5', 'hdf5')):
          kwargs.update({
            'filepath': model.dir['ckpt'] + "/{epoch:06d}.weights.hdf5",
            'save_weights_only': True
          })
        else:
          kwargs.update({
            'filepath': model.dir['ckpt'] + "/model.ckpt",
          })
        mc.append(model_ckpt.ModelCheckpoint(**kwargs))
      return mc
    elif (identifier == "tensorboard"):
      return cb.TensorBoard(**kwargs, log_dir=model.dir['logs'])
    elif (identifier == "terminator"):
      return early_stopping.Terminator(
        filename=model.dir['train']+'/train_ctrl.json',
        **kwargs
      )
    elif (identifier == "timer"):
      return early_stopping.Timer(**kwargs)
    elif (identifier == "variable"):
      return variable.VariablesTracker(
        filename=model.dir['train']+'/variables.csv',
        log_dir=model.dir['logs'],
        **kwargs
      )
    elif (identifier == "weighted_loss"):
      kwargs.update(
        dict(
          data=model.data,
          log_dir=model.dir['logs']
        )
      )
      return utils.get_class(weighted_loss, kwargs=kwargs)
  elif callable(identifier):
    return identifier
  else:
    utils.raise_err(
      ValueError,
      f"Could not interpret callback identifier: '{identifier}'. " \
      f"The available ones are {_CB_LIST}"
    )
