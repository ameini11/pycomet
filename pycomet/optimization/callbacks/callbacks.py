import tensorflow.python.keras.callbacks as cb

from pycomet.optimization.callbacks import early_stopping
from pycomet.optimization.callbacks import learning_rate
from pycomet.optimization.callbacks import plotting
from pycomet.optimization.callbacks import variable
from pycomet.optimization.callbacks import weighted_loss
from pycomet import utils


def get_cb_list(model, callbacks=None):
    if callbacks is not None:
        return [
            get_callback(model, identifier, kwargs) for identifier, kwargs \
                in callbacks.items() if kwargs is not None
        ]

def get_callback(model, identifier, kwargs):
    if isinstance(identifier, str):
        if identifier == "early_stopping":
            return cb.EarlyStopping(**kwargs)
        elif identifier == "lr_scheduler":
            return utils.get_class(learning_rate, kwargs=kwargs)
        elif identifier == "reduce_lr_on_plateau":
            return learning_rate.ReduceLROnPlateau(
                **kwargs
            )
        elif identifier == "lr_tracker":
            return learning_rate.LearningRateTracker(
                file=model.dir['train']+'/lr.csv',
                log_dir=model.dir['logs'],
                **kwargs
            )
        elif identifier == "plotter":
            return plotting.Plotter(
                test_data=model.data.test,
                train_dir=model.dir['train'],
                **kwargs
            )
        elif identifier == "model_ckpt":
            return cb.ModelCheckpoint(
                filepath=model.dir['ckpt']+"/weights_{epoch:06d}.h5", **kwargs
            )
        elif identifier == "tensorboard":
            return cb.TensorBoard(**kwargs, log_dir=model.dir['logs'])
        elif identifier == "timer":
            return early_stopping.Timer(**kwargs)
        elif identifier == "variable":
            return variable.VariableValue(**kwargs)
        elif identifier == "weighted_loss":
            kwargs.update(
                dict(
                    data_container=model.data.train,
                    log_dir=model.dir['train']
                )
            )
            return utils.get_class(weighted_loss, kwargs=kwargs)
    elif callable(identifier):
        return identifier
    else:
        utils.raise_value_err(
            f"Could not interpret callback identifier: '{identifier}'."
        )
