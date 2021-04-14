import pickle
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from . import display
from . import losses as losses_module
from . import metrics as metrics_module
from . import train as train_module
from .backend import tf
from .callbacks import CallbackList
from .utils import get_num_args, guarantee_initialized_variables, timing


class Model(object):
    """The ``Model`` class trains a ``Net`` on a ``Data``.

    Args:
        data: ``deepxde.data.Data`` instance.
        net: ``deepxde.maps.Map`` instance.
    """

    def __init__(self, data, net):
        self.data = data
        self.net  = net

        self.optimizer  = None
        self.batch_size = None

        self.losses    = None
        self.totalloss = None
        self.train_op  = None
        self.metrics   = None

        self.saver       = None
        self.train_state = TrainState()
        self.losshistory = LossHistory()
        self.stop_training = False
        self.callbacks = None

    @timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=[],
        decay=None,
        loss_weights=None,
    ):
        """Configures the model for training.

        Args:
            optimizer: String. Name of optimizer.
            lr: A Tensor or a floating point value. The learning rate.
            loss: If the same loss is used for all errors, then `loss` is a String (name of objective function) or
                objective function. If different errors use different losses, then `loss` is a list whose size is equal
                to the number of errors.
            metrics: List of metrics to be evaluated by the model during training.
            decay: Tuple. Name and parameters of decay to the initial learning rate. One of the following options:

                - `inverse time decay <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/inverse_time_decay>`_: ("inverse time", decay_steps, decay_rate)
                - `cosine decay <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/cosine_decay>`_: ("cosine", decay_steps, alpha)

            loss_weights: A list specifying scalar coefficients (Python floats)
                to weight the loss contributions. The loss value that will be minimized by the model
                will then be the weighted sum of all individual losses,
                weighted by the loss_weights coefficients.
        """
        print("Compiling model...")

        # Initializing the optimizer
        self.optimizer = get_train_opt(optimizer, lr=lr, decay=decay)

        # Initializing the loss functions
        self.loss_pts = losses_module.get(loss)
        if isinstance(self.loss, (list, tuple)):
            self.loss_pts = self.loss[0]
            self.loss_res = self.loss[1]

        # Defining metrics
        self.metrics = [ metrics_module.get(m) for m in metrics ]

        self.net.compile(optimizer=self.optimizer, loss=self.loss_pts, metrics=self.metrics, loss_weights=loss_weights)
        # if data.num_coll > 0
        #     self.net.add_loss(data.residual_loss(self.loss_res, self.net.inputs))
        self.net.summary()


    @timing
    def train(
        self,
        epochs=None,
        batch_size=None,
        display_every=1000,
        disregard_previous_best=False,
        callbacks=None,
        model_restore_path=None,
        model_save_path=None,
        print_model=False,
    ):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            epochs: Integer. Number of epochs to train the model.
            batch_size: Integer or ``None``. Not fully supported yet.
            display_every: Integer. Print the loss and metrics every this steps.
            uncertainty: Boolean. If ``True``, use Monte-Carlo Dropout to estimate uncertainty.
            disregard_previous_best: If ``True``, disregard the previous saved best model.
            callbacks: List of ``deepxde.callbacks.Callback`` instances.
                List of callbacks to apply during training.
            model_restore_path: String. Path where parameters were previously saved.
                See ``save_path`` in `tf.train.Saver.restore <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#restore>`_.
            model_save_path: String. Prefix of filenames created for the checkpoint.
                See ``save_path`` in `tf.train.Saver.save <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#save>`_.
            print_model: If ``True``, print the values of all variables.
        """
        self.batch_size = batch_size

        ### Initializing a list of callbacks
        self.callbacks = CallbackList(callbacks=callbacks)
        #   Initializing the model for callbacks operations
        self.callbacks.set_model(self)

        ### Neglecting previous best loss (best model saving)
        if disregard_previous_best:
            self.train_state.disregard_best()

        ### Recovering previous model
        if model_restore_path is not None:
            #TODO with hdf5 files
            pass 
        #     print("Restoring model from {} ...".format(model_restore_path))
        #     self.saver.restore(self.sess, model_restore_path)

        ### Training the model
        print("Training model...\n")
        self.stop_training = False
        #   Update the training/test state
        self.train_state.set_data_train(*self.data.train_next_batch(self.batch_size))
        self.train_state.set_data_test(*self.data.test())
        #   Callbacks operations
        self.callbacks.on_train_begin()
        #   Gradient Descent Steps over epochs
        self.GradDescent(epochs, display_every)
        #   Callbacks operations
        self.callbacks.on_train_end()

        print("")
        # display.training_display.summary(self.train_state)
        if print_model:
            self.print_model()
        if model_save_path is not None:
            self.save(model_save_path, verbose=1)
        return self.losshistory, self.train_state

    def evaluate(self, x, y, callbacks=None):
        """Returns the loss values & metrics values for the model in test mode."""
        raise NotImplementedError(
            "Model.evaluate to be implemented. Alternatively, use Model.predict."
        )

    @timing
    def predict(self, x, operator=None, callbacks=None):
        """Generates output predictions for the input samples."""
        print("Predicting...")
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)
        self.callbacks.on_predict_begin()
        if operator is None:
            y = self.sess.run(
                self.net.outputs, feed_dict=self.net.feed_dict(False, False, 2, x)
            )
        else:
            if get_num_args(operator) == 2:
                op = operator(self.net.inputs, self.net.outputs)
            elif get_num_args(operator) == 3:
                op = operator(self.net.inputs, self.net.outputs, x)
            y = self.sess.run(
                op,
                feed_dict=self.net.feed_dict(False, False, 2, x),
            )
        self.callbacks.on_predict_end()
        return y

    def GradDescent(self, epochs, display_every):
        for i in range(epochs):
            self.callbacks.on_epoch_begin()
            
            for data_i in self.data.get_batch_data(self.batch_size):
                self.callbacks.on_batch_begin()
                self.train_state.set_data_train(data_i)

                self.sess.run(
                    self.train_op,
                    feed_dict=self.net.feed_dict(
                        True,
                        True,
                        0,
                        self.train_state.X_train,
                        self.train_state.y_train,
                        self.train_state.train_aux_vars,
                    ),
                )

                self.callbacks.on_batch_end()

            self.train_state.epoch += 1
            self.train_state.step  += 1

            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

    def state_dict(self):
        """Returns a dictionary containing all variables."""
        destination = OrderedDict()
        var_names = [v.name    for v in self.net.trainable_variables]
        values    = [v.value() for v in self.net.trainable_variables]
        for k, v in zip(var_names, values):
            destination[k] = v
        return destination

    def print_model(self):
        """Prints all trainable variables."""
        var_names = [v.name    for v in self.net.trainable_variables]
        values    = [v.value() for v in self.net.trainable_variables]
        for k, v in zip(var_names, values):
            print("\nVariable: {}, Shape: {}".format(k, v.shape))
            tf.print(v)

    def save(self, save_path, protocol="tf.train.Saver", verbose=0):
        """Saves all variables to a disk file.

        Args:
            protocol (string): If `protocol` is "tf.train.Saver", save using `tf.train.Save <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#attributes>`_.
                If `protocol` is "pickle", save using the Python pickle module.
                Only "tf.train.Saver" protocol supports ``restore()``.
        """
        if verbose > 0:
            print(
                "Epoch {}: saving model to {}-{} ...\n".format(
                    self.train_state.epoch, save_path, self.train_state.epoch
                )
            )
        if protocol == "tf.train.Saver":
            self.saver.save(self.sess, save_path, global_step=self.train_state.epoch)
        elif protocol == "pickle":
            with open("{}-{}.pkl".format(save_path, self.train_state.epoch), "wb") as f:
                pickle.dump(self.state_dict(), f)

    def restore(self, save_path, verbose=0):
        if verbose > 0:
            print("Restoring model from {} ...\n".format(save_path))
        self.saver.restore(self.sess, save_path)


class TrainState(object):
    def __init__(self):
        self.epoch, self.step = 0, 0

        # Data
        self.X_train, self.y_train = None, None
        self.X_test,  self.y_test  = None, None

        # Results of current step
        self.y_pred_train = None
        self.loss_train,    self.loss_test    = None, None
        self.y_pred_test,   self.y_std_test   = None, None
        self.metrics_train, self.metrics_test = None, None

        # The best results correspond to the min train loss
        self.best_step = 0
        self.best_loss_train, self.best_loss_test = np.inf, np.inf
        self.best_y, self.best_ystd = None, None
        self.best_metrics_train, self.best_metrics_test = None, None

    def set_data_train(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train

    def set_data_test(self, X_test, y_test):
        self.X_test, self.y_test = X_test, y_test

    def update_best(self):
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_test = np.sum(self.loss_test)
            self.best_y, self.best_ystd = self.y_pred_test, self.y_std_test
            self.best_metrics_train, self.best_metrics_test = self.metrics_train, self.metrics_test

    def disregard_best(self):
        self.best_loss_train = np.inf

    def packed_data(self):
        def merge_values(values):
            if values is None:
                return None
            return np.hstack(values) if isinstance(values, (list, tuple)) else values

        X_train = merge_values(self.X_train)
        y_train = merge_values(self.y_train)
        X_test = merge_values(self.X_test)
        y_test = merge_values(self.y_test)
        best_y = merge_values(self.best_y)
        best_ystd = merge_values(self.best_ystd)
        return X_train, y_train, X_test, y_test, best_y, best_ystd


class LossHistory(object):
    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.metrics_train = []
        self.loss_test = []
        self.metrics_test = []
        self.loss_weights = 1

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def append(self, step, loss_train, loss_test, metrics_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        self.metrics_train.append(metrics_train)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        if metrics_test is None:
            metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
        self.metrics_test.append(metrics_test)
