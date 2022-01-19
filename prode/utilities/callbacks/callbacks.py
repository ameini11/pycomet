import os
import sys
import time
import copy
import tensorflow.keras.callbacks as cb

from tensorflow.python.util.tf_export import keras_export
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
    if identifier == "early_stopping":
        return cb.EarlyStopping(**kwargs)
    elif identifier == "model_ckpt":
        return cb.ModelCheckpoint(
            filepath=model.ckpt_dir+"/weights_{epoch:06d}.h5", **kwargs
        )
    elif identifier == "tensorboard":
        return cb.TensorBoard(**kwargs, log_dir=model.TB_dir)
    elif identifier == "timer":
        return early_stopping.Timer(**kwargs)
    elif identifier == "lr_scheduler":
        return get_lr_scheduler(kwargs)
    elif identifier == "weighted_loss":
        return get_weighted_loss(model, kwargs)
    elif identifier == "lr_tracker":
        return learning_rate.LearningRateTracker(
            **kwargs, file=model.train_dir+'/lr.csv', log_dir=model.TB_dir
        )
    elif identifier == "plotter":
        return plotting.Plotter(
            test_data=model.data.test,
            train_dir=model.train_dir,
            **kwargs
        )
    elif identifier == "variable":
        return variable.VariableValue(**kwargs)
    elif callable(identifier):
        return identifier
    else:
        utils.raise_value_err(
            "Could not interpret callback identifier: '{}'.".format(identifier)
        )

def get_weighted_loss(model, kwargs):
    # Add loss weights to the network class
    # model.net.add_loss_weights(model.loss_weights)
    # Manipulate arguments for `weighted_loss` callback
    name = kwargs.pop('name')
    # kwargs.update(
    #     {
    #         'data_container': model.data.train,
    #         'log_dir': model.TB_dir
    #     }
    # )
    return utils.get_class(weighted_loss, name)(**kwargs)

def get_lr_scheduler(kwargs):
    # Manipulate arguments for `learning_rate` callback
    name = kwargs.pop('name')
    return utils.get_class(learning_rate, name)(**kwargs)
