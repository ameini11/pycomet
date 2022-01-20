import tensorflow.python.keras.callbacks as cb

from prode.utilities.callbacks import early_stopping
from prode.utilities.callbacks import learning_rate
from prode.utilities.callbacks import plotting
from prode.utilities.callbacks import variable
from prode.utilities.callbacks import weighted_loss
from prode.utilities import utils


def get_cb_list(model, callbacks):
    if callbacks:
        return [
            get_callback(model, identifier, kwargs) for identifier, kwargs in \
                callbacks.items() if kwargs is not None
        ]

def get_callback(model, identifier, kwargs):
    cb_identifier = {
        "early_stopping": cb.EarlyStopping(**kwargs),
        "lr_scheduler":   utils.get_class(learning_rate, kwargs=kwargs),
        "lr_tracker":     learning_rate.LearningRateTracker(**kwargs),
        "plotter":        plotting.Plotter(**kwargs),
        "model_ckpt":     cb.ModelCheckpoint(
                              filepath=model.ckpt_dir+"/weights_{epoch:06d}.h5",
                              **kwargs
                          ),
        "tensorboard":    cb.TensorBoard(**kwargs, log_dir=model.TB_dir),
        "timer":          early_stopping.Timer(**kwargs),
        "variable":       variable.VariableValue(**kwargs),
        "weighted_loss":  utils.get_class(weighted_loss, kwargs=kwargs)
    }

    if isinstance(identifier, str):
        return cb_identifier[identifier]
    elif callable(identifier):
        return identifier
    else:
        utils.raise_value_err(
            f"Could not interpret callback identifier: '{identifier}'."
        )
