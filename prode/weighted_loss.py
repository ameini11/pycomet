import time
import numpy as np

import tensorflow                   as tf
import tensorflow.keras.backend     as K
import tensorflow.keras.callbacks   as CB

from tensorflow.python.eager          import backprop
from tensorflow.python.util.tf_export import keras_export

from prode import utils


###############################################################################
@keras_export('keras.callbacks.ConstantWeightsAdapter')
class ConstantWeightsAdapter(CB.Callback):
    """Base class for adaptive loss weights techniques."""

    def __init__(
        self,
        model,
        data_generator=None,
        freq=1,
        log_dir=None,
        name='constant_weights_adapter',
        **kwargs):
        super(ConstantWeightsAdapter, self).__init__()

        self.model = model
        if self.model.loss_weights is None:
            utils.warning("No loss weights have been provided to the network `{}`: "
                "the default value of 1 will be applied.".format(self.model.name))
            self.model.loss_weights = {}
            for data_id in self.model.data_ids:
                self.model.loss_weights[data_id] = 1.

        self.data_generator = data_generator
        self.freq = np.clip(freq, 1, None)
        self.name = name

        self.log_dir = log_dir
        if self.log_dir is not None:
            self.logs = { k: tf.summary.create_file_writer(self.log_dir + \
                '/scalars/loss_weights/' + k) for k in self.model.loss_weights }

    def print_weights(self, epoch, logs):
        log_weights = {'weight_'+k+'_loss': v for k,v in self.model.loss_weights.items()}
        logs.update(log_weights)
        if self.log_dir is not None:
            for k, log in self.logs.items():
                with log.as_default():
                    tf.summary.scalar(name='epoch_weight_'+k+'_loss', \
                        data=self.model.loss_weights[k], step=epoch)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.print_weights(epoch, logs)


###############################################################################
@keras_export('keras.callbacks.EmpiricalWeightsAdapter')
class EmpiricalWeightsAdapter(ConstantWeightsAdapter):
    """Base class for adaptive loss weights techniques."""

    def __init__(
        self,
        model,
        data_generator=None,
        freq=1,
        alpha=0.8,
        log_dir=None,
        name='empirical_weights_adapter',
        **kwargs):
        super(EmpiricalWeightsAdapter, self).__init__(
            model=model,
            data_generator=data_generator,
            freq=freq,
            log_dir=log_dir,
            name=name)

        self.alpha = tf.cast(tf.clip_by_value(alpha, 0., 1.), K.floatx())
        self.loss_gradients = {}
        self.dataset = {}
        self.max_samples = 20000

    def on_train_begin(self, logs={}):
        if 'res' not in self.model.data_ids:
            utils.warning("Network `{}` hasn't any residual loss to be evaluated: "
                "`EmpiricalWeightsAdapter` can't be applied.".format(self.model.name))
        else:
            self.update()

    def on_epoch_end(self, epoch, logs={}):
        if 'res' in self.model.data_ids and (epoch+1) % self.freq == 0:
            self.update()
        logs = logs or {}
        self.print_weights(epoch, logs)

    def update(self):
        self.sample_dataset()
        self.eval_loss_gradients()
        self.update_loss_weights()


    # Sampling data
    ###########################################################################
    def sample_dataset(self):
        for data_id, dset in self.data_generator.data.items():
            n_samples = self.get_cardinality(dset)
            indeces   = np.random.permutation(n_samples)
            if n_samples > self.max_samples:
                indeces = np.random.choice(indeces, self.max_samples, replace=False)
            self.dataset[data_id] = self.get_shuffled_data(dset, indeces)

    def get_cardinality(self, x):
        if isinstance(x, (list,tuple)):
            return self.get_cardinality(x[0])
        else:
            return x.shape[0]

    def get_shuffled_data(self, dset, indeces):
        dset_shuffled = []
        for xy in dset:
            if isinstance(xy, (list,tuple)):
                dset_shuffled.append([tf.convert_to_tensor(i[indeces], \
                    dtype=K.floatx()) for i in xy])
            else:
                dset_shuffled.append(tf.convert_to_tensor(xy[indeces], \
                    dtype=K.floatx()))
        return dset_shuffled


    # Handling loss gradients
    ###########################################################################
    def eval_loss_gradients(self):
        pred, losses = {}, {}
        with backprop.GradientTape(persistent=True) as tape:
            for data_id in self.model.data_ids:
                pred[data_id], losses[data_id] = self.get_pred_loss( \
                    self.dataset[data_id], data_id)
        for data_id in self.model.data_ids:
            self.loss_gradients[data_id] = tape.gradient(losses[data_id], \
                self.model.trainable_variables)

    def get_pred_loss(self, data, data_id):
        x, y = data
        loss_fn = self.model.loss_fns[data_id]
        if data_id == 'res':
            return loss_fn(self.model, x, training=False)
        else:
            y_pred = self.model(x, training=False)
            loss   = loss_fn(y, y_pred)
            return y_pred, loss

    # Handling loss weights
    ###########################################################################
    def update_loss_weights(self):
        new_weights = self.eval_loss_weights()
        for data_id, loss in self.model.loss_weights.items():
            if data_id == 'res':
                continue
            self.model.loss_weights[data_id] = self.alpha * new_weights[data_id] + \
                (1. - self.alpha) * self.model.loss_weights[data_id]

    def eval_loss_weights(self):
        # Get the maximum gradient value from residual loss
        grad_res_max = self.max_grads(self.loss_gradients['res'])

        # Get the mean of the gradient magnitudes for each remaining loss term
        new_weights = {}
        for data_id, grad in self.loss_gradients.items():
            if data_id == 'res':
                new_weights[data_id] = self.model.loss_weights[data_id]
            # Get the mean of the gradient magnitudes for each remaining loss term
            grads_mean = self.mean_grads(grad)
            # Update the loss weights
            new_weights[data_id] = grad_res_max / grads_mean

        return new_weights

    def max_grads(self, grads):
        m = [ tf.reduce_max(tf.math.abs(grad)) for grad in grads ]
        return tf.reduce_max(m)

    def mean_grads(self, grads):
        mean = [ tf.reduce_mean(tf.math.abs(grad)) for grad in grads ]
        return tf.reduce_mean(mean)


###############################################################################
@keras_export('keras.callbacks.EmpiricalWeightsAdapter')
class EmpiricalWeightsAdapter(ConstantWeightsAdapter):
    """Base class for adaptive loss weights techniques."""

    def __init__(
        self,
        model,
        data_generator=None,
        freq=1,
        alpha=0.8,
        log_dir=None,
        name='empirical_weights_adapter',
        **kwargs):
        super(EmpiricalWeightsAdapter, self).__init__(
            model=model,
            data_generator=data_generator,
            freq=freq,
            log_dir=log_dir,
            name=name)

        self.alpha = tf.cast(tf.clip_by_value(alpha, 0., 1.), K.floatx())
        self.loss_gradients = {}
        self.dataset = {}
        self.max_samples = 20000

    def on_train_begin(self, logs={}):
        if 'res' not in self.model.data_ids:
            utils.warning("Network `{}` hasn't any residual loss to be evaluated: "
                "`EmpiricalWeightsAdapter` can't be applied.".format(self.model.name))
        else:
            self.update()

    def on_epoch_end(self, epoch, logs={}):
        if 'res' in self.model.data_ids and (epoch+1) % self.freq == 0:
            self.update()
        logs = logs or {}
        self.print_weights(epoch, logs)

    def update(self):
        self.sample_dataset()
        self.eval_loss_gradients()
        self.update_loss_weights()


    # Sampling data
    ###########################################################################
    def sample_dataset(self):
        for data_id, dset in self.data_generator.data.items():
            n_samples = self.get_cardinality(dset)
            indeces   = np.random.permutation(n_samples)
            if n_samples > self.max_samples:
                indeces = np.random.choice(indeces, self.max_samples, replace=False)
            self.dataset[data_id] = self.get_shuffled_data(dset, indeces)

    def get_cardinality(self, x):
        if isinstance(x, (list,tuple)):
            return self.get_cardinality(x[0])
        else:
            return x.shape[0]

    def get_shuffled_data(self, dset, indeces):
        dset_shuffled = []
        for xy in dset:
            if isinstance(xy, (list,tuple)):
                dset_shuffled.append([tf.convert_to_tensor(i[indeces], \
                    dtype=K.floatx()) for i in xy])
            else:
                dset_shuffled.append(tf.convert_to_tensor(xy[indeces], \
                    dtype=K.floatx()))
        return dset_shuffled


    # Handling loss gradients
    ###########################################################################
    def eval_loss_gradients(self):
        pred, losses = {}, {}
        with backprop.GradientTape(persistent=True) as tape:
            for data_id in self.model.data_ids:
                pred[data_id], losses[data_id] = self.get_pred_loss( \
                    self.dataset[data_id], data_id)
        for data_id in self.model.data_ids:
            self.loss_gradients[data_id] = tape.gradient(losses[data_id], \
                self.model.trainable_variables)

    def get_pred_loss(self, data, data_id):
        x, y = data
        loss_fn = self.model.loss_fns[data_id]
        if data_id == 'res':
            return loss_fn(self.model, x, training=False)
        else:
            y_pred = self.model(x, training=False)
            loss   = loss_fn(y, y_pred)
            return y_pred, loss

    # Handling loss weights
    ###########################################################################
    def update_loss_weights(self):
        new_weights = self.eval_loss_weights()
        for data_id, loss in self.model.loss_weights.items():
            if data_id == 'res':
                continue
            self.model.loss_weights[data_id] = self.alpha * new_weights[data_id] + \
                (1. - self.alpha) * self.model.loss_weights[data_id]

    def eval_loss_weights(self):
        # Get the maximum gradient value from residual loss
        grad_res_max = self.max_grads(self.loss_gradients['res'])

        # Get the mean of the gradient magnitudes for each remaining loss term
        new_weights = {}
        for data_id, grad in self.loss_gradients.items():
            if data_id == 'res':
                new_weights[data_id] = self.model.loss_weights[data_id]
            # Get the mean of the gradient magnitudes for each remaining loss term
            grads_mean = self.mean_grads(grad)
            # Update the loss weights
            new_weights[data_id] = grad_res_max / grads_mean

        return new_weights

    def max_grads(self, grads):
        m = [ tf.reduce_max(tf.math.abs(grad)) for grad in grads ]
        return tf.reduce_max(m)

    def mean_grads(self, grads):
        mean = [ tf.reduce_mean(tf.math.abs(grad)) for grad in grads ]
        return tf.reduce_mean(mean)

