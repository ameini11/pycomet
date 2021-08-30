import os
import sys
import time
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
            get_callback(model, identifier, args) for identifier, args in \
                callbacks.items() if args is not None
        ]

def get_callback(model, identifier, args):
    if identifier == "early_stopping":
        return cb.EarlyStopping(**args)
    elif identifier == "model_ckpt":
        return cb.ModelCheckpoint(**args, filepath=model.ckpt_dir+"/weights_{epoch:06d}.h5")
    elif identifier == "tensorboard":
        return cb.TensorBoard(**args, log_dir=model.TB_dir)
    elif identifier == "timer":
        return early_stopping.Timer(**args)
    elif identifier == "lr_scheduler":
        return get_lr_scheduler(args)
    elif identifier == "weighted_loss":
        return get_weighted_loss(model, args)
    elif identifier == "lr_tracker":
        return learning_rate.LearningRateTracker(**args, file=model.train_dir+'/lr.csv', log_dir=model.TB_dir)
    elif identifier == "plotter":
        return plotting.Plotter(**args, train_dir=model.train_dir)
    elif identifier == "variable":
        return variable.VariableValue(**args)
    elif callable(identifier):
        return identifier
    else:
        utils.raise_value_err("Could not interpret callback identifier: ", identifier)

def get_weighted_loss(model, args):
    # Add loss weights to the network class
    # model.net.add_loss_weights(model.loss_weights)
    # Manipulate arguments for `weighted_loss` callback
    name = args.pop('name')
    args.update(
        {
            'model':          model.net,
            'data_generator': model.data.train,
            'log_dir':        model.TB_dir
        }
    )
    return utils.get_class(weighted_loss, name)(**args)

def get_lr_scheduler(args):
    name = args.pop('name')
    return utils.get_class(learning_rate, name)(**args)
