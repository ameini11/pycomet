import time
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as CB

from tqdm import tqdm
from tensorflow.python.eager import backprop
from tensorflow.python.util.tf_export import keras_export
from prode.utilities import utils



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
        self.freq = np.clip(freq, 1, None) if freq is not None else np.inf
        self.name = name

        self.log_dir = log_dir
        if self.log_dir is not None:
            self.logs = { k: tf.summary.create_file_writer(self.log_dir + \
                '/scalars/loss_weights/' + k) for k in self.model.loss_weights }

    def on_train_begin(self, logs={}):
        self.print_weights(0, logs)

    def on_epoch_end(self, epoch, logs={}):
        self.print_weights(epoch, logs)

    def print_weights(self, epoch, logs):
        log_weights = {k+'_loss_weight': v for k,v in self.model.loss_weights.items()}
        logs.update(log_weights)
        if self.log_dir is not None:
            for k, log in self.logs.items():
                with log.as_default():
                    tf.summary.scalar(name='epoch_'+k+'_loss_weight', \
                        data=self.model.loss_weights[k], step=epoch)


@keras_export('keras.callbacks.EmpiricalWeightsAdapter')
class EmpiricalWeightsAdapter(ConstantWeightsAdapter):
    """Adaptive loss weights technique for PINNs based on an empirical learning
    rate annealing algorithm.

    See:
        Sifan Wang, Yujun Teng, Paris Perdikaris
        `Understanding and mitigating gradient pathologies in physics-informed
        neural networks`
        https://arxiv.org/abs/2001.04536"""

    def __init__(
        self,
        model,
        data_generator=None,
        freq=1000,
        alpha=0.8,
        log_dir=None,
        max_samples=20000,
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
        self.max_samples = max_samples

    def on_train_begin(self, logs={}):
        if 'res' not in self.model.data_ids:
            utils.warning("Network `{}` hasn't any residual loss to be evaluated: "
                "`EmpiricalWeightsAdapter` can't be applied.".format(self.model.name))
        else:
            self.update()
        self.print_weights(0, logs)

    def on_epoch_end(self, epoch, logs={}):
        if 'res' in self.model.data_ids and (epoch+1) % self.freq == 0:
            self.update()
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
            weight = self.alpha * new_weights[data_id] + \
                (1. - self.alpha) * self.model.loss_weights[data_id]
            self.model.loss_weights[data_id] = weight.numpy()

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


@keras_export('keras.callbacks.EmpiricalMeanWeightsAdapter')
class EmpiricalMeanWeightsAdapter(EmpiricalWeightsAdapter):
    """Adaptive loss weights technique for PINNs based on an empirical learning
    rate annealing algorithm.

    See:
        Sifan Wang, Yujun Teng, Paris Perdikaris
        `Understanding and mitigating gradient pathologies in physics-informed
        neural networks`
        https://arxiv.org/abs/2001.04536"""

    def __init__(
        self,
        model,
        data_generator=None,
        freq=1,
        alpha=0.8,
        log_dir=None,
        max_samples=20000,
        name='empirical_mean_weights_adapter',
        **kwargs):
        super(EmpiricalWeightsAdapter, self).__init__(
            model=model,
            data_generator=data_generator,
            freq=freq,
            alpha=alpha,
            log_dir=log_dir,
            max_samples=max_samples,
            name=name)

    # Handling loss weights
    ###########################################################################
    def eval_loss_weights(self):
        # Get the maximum gradient value from residual loss
        grad_res_mean = self.mean_grads(self.loss_gradients['res'])

        # Get the mean of the gradient magnitudes for each remaining loss term
        new_weights = {}
        for data_id, grad in self.loss_gradients.items():
            if data_id == 'res':
                new_weights[data_id] = self.model.loss_weights[data_id]
            # Get the mean of the gradient magnitudes for each remaining loss term
            grads_mean = self.mean_grads(grad)
            # Update the loss weights
            new_weights[data_id] = grad_res_mean / grads_mean

        return new_weights


@keras_export('keras.callbacks.NTKWeightsAdapter')
class NTKWeightsAdapter(EmpiricalWeightsAdapter):
    """Adaptive loss weights technique for PINNs based on Neural Tangent Kernel
    (NTK) theory.

    See:
        Sifan Wang, Xinling Yu, Paris Perdikaris
        `When and why PINNs fail to train: A neural tangent kernel perspective`
        https://arxiv.org/abs/2007.14527"""

    def __init__(
        self,
        model,
        data_generator=None,
        freq=None,
        log_dir=None,
        max_samples=100,
        name='ntk_weights_adapter',
        **kwargs):
        super(NTKWeightsAdapter, self).__init__(
            model=model,
            data_generator=data_generator,
            freq=freq,
            log_dir=log_dir,
            max_samples=max_samples,
            name=name)

    def on_train_begin(self, logs={}):
        self.update()
        self.print_weights(0, logs)

    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1) % self.freq == 0:
            self.update()
        self.print_weights(epoch, logs)

    def update(self):
        self.sample_dataset()
        ntk = self.get_ntk()
        self.update_loss_weights(ntk)

    # Handling loss gradients
    ###########################################################################
    def get_ntk(self):
        pred, losses = {}, {}
        with backprop.GradientTape(persistent=True) as tape:
            for data_id in self.model.data_ids:
                pred[data_id], losses[data_id] = self.get_pred_loss( \
                    self.dataset[data_id], data_id)
                # Flatten prediction matrix into 1D vector
                pred[data_id] = tf.split(tf.reshape(pred[data_id], [-1]), \
                    tf.size(pred[data_id]).numpy())
        return self.eval_diag_ntk(pred, tape)

    def eval_diag_ntk(self, pred, tape):
        utils.print_submain("From `NTKWeightsAdapter` callback:")
        ntk = {}
        for data_id, pred_i in pred.items():
            utils.print_submain("  - Evaluating Neural Tangent Kernel matrix "
                "for `{}` data points.".format(data_id))
            ntk[data_id] = []
            for elem in tqdm(pred_i, file=sys.stdout):
                grad = tape.gradient(elem, self.model.trainable_variables)
                # Flatten all gradient matrices into one unique 1D vector
                grad_vec = np.concatenate([ np.abs(g.numpy()).flatten() for g in grad ])
                ntk[data_id].append(np.dot(grad_vec,grad_vec))
        return ntk

    # Handling loss weights
    ###########################################################################
    def update_loss_weights(self, ntk):
        normalization_grad = np.sum([ np.sum(ntk_i) for ntk_i in ntk.values()])
        for data_id in self.model.data_ids:
            self.model.loss_weights[data_id] = normalization_grad / np.sum(ntk[data_id])


# @keras_export('keras.callbacks.NTKWeightsAdapter')
# class NTKWeightsAdapter(EmpiricalWeightsAdapter):
#     """Adaptive loss weights technique for PINNs based on Neural Tangent Kernel
#     (NTK) theory.

#     See:
#         Sifan Wang, Xinling Yu, Paris Perdikaris
#         `When and why PINNs fail to train: A neural tangent kernel perspective`
#         https://arxiv.org/abs/2007.14527"""

#     def __init__(
#         self,
#         model,
#         data_generator=None,
#         freq=1000,
#         log_dir=None,
#         max_samples=10000,
#         name='ntk_weights_adapter',
#         **kwargs):
#         super(NTKWeightsAdapter, self).__init__(
#             model=model,
#             data_generator=data_generator,
#             freq=freq,
#             log_dir=log_dir,
#             max_samples=max_samples,
#             name=name)

#     def on_train_begin(self, logs={}):
#         self.update()
#         self.print_weights(0, logs)

#     def on_epoch_end(self, epoch, logs={}):
#         if (epoch+1) % self.freq == 0:
#             self.update()
#         self.print_weights(epoch, logs)

#     @tf.function()
#     def update(self):
#         self.sample_dataset()
#         ntk = self.get_ntk()
#         self.update_loss_weights(ntk)

#     # Handling loss gradients
#     ###########################################################################
#     def get_ntk(self):
#         pred, losses = {}, {}
#         with backprop.GradientTape(persistent=True) as tape:
#             for data_id in self.model.data_ids:
#                 pred[data_id], losses[data_id] = self.get_pred_loss( \
#                     self.dataset[data_id], data_id)
#                 # Flatten prediction matrix into 1D vector
#                 pred[data_id] = tf.split(tf.reshape(pred[data_id], [-1]), \
#                     tf.size(pred[data_id]))
#         return self.eval_diag_ntk(pred, tape)

#     def eval_diag_ntk(self, pred, tape):

#         utils.print_submain("From `NTKWeightsAdapter` callback:")
#         ntk = {}
#         for data_id, pred_i in pred.items():
#             utils.print_submain("  - Evaluating Neural Tangent Kernel matrix "
#                 "for `{}` data points.".format(data_id))
#             pred_iter = iter(pred_i)

#             # Create a tf.while_loop to iterate over `pred_i` elements
#             j    = 0
#             elem = next(pred_iter)
#             ntk_ta = tf.TensorArray(dtype=K.floatx(), size=len(pred_i), \
#                 name='ntk_'+data_id)
#             init_state = (j, elem, ntk_ta)

#             def cond(j, elem, ntk_ta):
#                 return tf.less(j, len(pred_i))

#             def body(j, elem, ntk_ta):
#                 grad     = tape.gradient(elem, self.model.trainable_variables, \
#                     unconnected_gradients='zero')
#                 # Flatten all gradient matrices into one unique 1D vector
#                 grad_vec = tf.concat([ tf.reshape(tf.math.abs(g),[-1]) for g in grad ], 0)
#                 ntk_ta   = ntk_ta.write(j, tf.experimental.numpy.dot(grad_vec,grad_vec))
#                 return j+1, next(pred_iter), ntk_ta

#             ntk[data_id] = tf.while_loop(cond, body, loop_vars=init_state, \
#                 parallel_iterations=len(pred_i))[2].stack()

#         return ntk

#     # Handling loss weights
#     ###########################################################################
#     def update_loss_weights(self, ntk):
#         normalization_grad = tf.math.reduce_sum([ tf.math.reduce_sum(ntk_i) for ntk_i in ntk.values()])
#         for data_id in self.model.data_ids:
#             self.model.loss_weights[data_id] = normalization_grad / tf.math.reduce_sum(ntk[data_id])
#         print(self.model.loss_weights)
