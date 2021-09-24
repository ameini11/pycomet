import sys
import itertools
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
        ode_loss_weights=None,
        data_generator=None,
        freq=1,
        log_dir=None,
        **kwargs
    ):
        super(ConstantWeightsAdapter, self).__init__()

        self.name = utils.get_class_name(self)
        self.data_generator = data_generator
        self.ode_loss_weights = ode_loss_weights
        self.freq = np.clip(freq, 1, None) if freq is not None else np.inf
        self.log_dir = log_dir

    def on_train_begin(self, logs={}):
        self.init_weights()
        self.print_weights(0, logs)

    def on_epoch_end(self, epoch, logs={}):
        self.print_weights(epoch, logs)

    def init_weights(self):
        if self.ode_loss_weights is None:
            utils.warning("No `ode_loss_weights` have been provided to `{}`: "
                "the initial value of 1 will be applied.".format(self.name))
            _ode_loss_weights = dict(zip(self.model.data_ids, [1.]*len(self.model.data_ids)))
        elif isinstance(self.ode_loss_weights, (list,tuple)):
            _ode_loss_weights = dict(zip(self.model.data_ids, ode_loss_weights))
        elif isinstance(ode_loss_weights, dict):
            _ode_loss_weights = self.ode_loss_weights
        else:
            utils.raise_value_err("`ode_loss_weights` passed into `{}` "
                "has to be a list or a dictionary:\n"
                "  - 'ics': weight for initial condition pts loss;\n"
                "  - 'pts': weight for anchor pts loss;\n"
                "  - 'res': weight for residual loss.".format(self.name))
        self.model.ode_loss_weights = _ode_loss_weights
        
        if self.log_dir is not None:
            self.logs = { k: tf.summary.create_file_writer(self.log_dir + \
                '/scalars/ode_loss_weights/' + k) for k in self.model.ode_loss_weights }

    def print_weights(self, epoch, logs):
        log_weights = {k+'_loss_weight': v for k,v in self.model.ode_loss_weights.items()}
        logs.update(log_weights)
        if self.log_dir is not None:
            for k, log in self.logs.items():
                with log.as_default():
                    tf.summary.scalar(name='epoch_'+k+'_loss_weight', \
                        data=self.model.ode_loss_weights[k], step=epoch)


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
        ode_loss_weights=None,
        data_generator=None,
        freq=1000,
        alpha=0.8,
        log_dir=None,
        max_samples=20000,
        method='mean',
        **kwargs
    ):
        super(EmpiricalWeightsAdapter, self).__init__(
            ode_loss_weights=ode_loss_weights,
            data_generator=data_generator,
            freq=freq,
            log_dir=log_dir
        )

        self.alpha = tf.cast(tf.clip_by_value(alpha, 0., 1.), K.floatx())
        self.loss_gradients = {}
        self.dataset = {}
        self.max_samples = max_samples

        self.grads_eval = self.mean_grads if method == 'mean' else self.max_grads

    def on_train_begin(self, logs={}):
        self.init_weights()
        if 'res' not in self.model.data_ids:
            utils.warning("Network `{}` hasn't any residual loss to be evaluated: "
                "`{}` can't be applied.".format(self.model.name, self.name))
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
        self.check_weights()

    def check_weights(self):
        weights = self.model.ode_loss_weights.values()
        if any(tf.math.is_nan(w) for w in weights) \
            or any(tf.math.is_inf(w) for w in weights):
            print("")
            utils.warning(
                "One or multiple `ode_loss_weights` evaluated in `{}` class "
                "are `inf` or `nan`.".format(self.name)
            )
            utils.print_submain("Here the `ode_loss_weights` values:")
            for k,v in self.model.ode_loss_weights.items():
                utils.print_submain("  - '{}': {}".format(k,v))
            utils.print_submain("The training will be stopped.")
            self.model.stop_training = True


    # Sampling data
    ###########################################################################
    def sample_dataset(self):
        for data_id, dset in self.data_generator.data.items():
            n_samples = utils.get_cardinality(dset)
            indeces = np.random.permutation(n_samples)
            if n_samples > self.max_samples:
                indeces = np.random.choice(
                    indeces, self.max_samples, replace=False
                )
            self.dataset[data_id] = self.shuffle_data(dset, indeces)

    def shuffle_data(self, dset, indeces):
        _dset = []
        for xy in dset:
            if isinstance(xy, (list,tuple)):
                _dset.append([tf.convert_to_tensor(i[indeces], \
                    dtype=K.floatx()) for i in xy])
            else:
                _dset.append(tf.convert_to_tensor(xy[indeces], \
                    dtype=K.floatx()))
        return _dset

    # Handling loss gradients
    ###########################################################################
    def eval_loss_gradients(self):
        pred, losses = {}, {}
        with backprop.GradientTape(persistent=True) as tape:
            for data_id in self.model.data_ids:
                pred[data_id], losses[data_id] = self.get_pred_loss(
                    *self.dataset[data_id], data_id=data_id
                )
        for data_id in self.model.data_ids:
            self.loss_gradients[data_id] = tape.gradient(
                losses[data_id], self.model.trainable_variables
            )

    def get_pred_loss(self, x, y, data_id):
        if data_id == 'res':
            y_pred = self.model.residual(x, training=False)
        else:
            y_pred = self.model(x, training=False)
        loss = self.model.compiled_loss[data_id](y, y_pred)
        return y_pred, loss

    # Handling loss weights
    ###########################################################################
    def update_loss_weights(self):
        new_weights = self.eval_loss_weights()
        for data_id, loss in self.model.ode_loss_weights.items():
            if data_id == 'res':
                continue
            weight = self.alpha * new_weights[data_id] + \
                (1. - self.alpha) * self.model.ode_loss_weights[data_id]
            self.model.ode_loss_weights[data_id] = weight.numpy()

    def eval_loss_weights(self):
        # Get the maximum/mean gradient value from residual loss (see `method')
        grad_res = self.grads_eval(self.loss_gradients['res'])
        new_weights = {}
        for data_id, grad in self.loss_gradients.items():
            if data_id == 'res':
                continue
            # Get the mean of the gradients for each remaining loss term
            grads_mean = self.mean_grads(grad)
            # Update the loss weights
            new_weights[data_id] = grad_res / grads_mean \
                * self.model.ode_loss_weights['res']
        return new_weights

    def max_grads(self, grads):
        m = [ tf.reduce_max(tf.math.abs(grad)) for grad in grads ]
        return tf.reduce_max(m)

    def mean_grads(self, grads):
        m = [ tf.reduce_mean(tf.math.abs(grad)) for grad in grads ]
        return tf.reduce_mean(m)


@keras_export('keras.callbacks.InverseDirichletWeightsAdapter')
class InverseDirichletWeightsAdapter(EmpiricalWeightsAdapter):
    """Adaptive loss weights technique for PINNs based on Inverse-Dirichlet
    weighting."""

    def __init__(
        self,
        ode_loss_weights=None,
        data_generator=None,
        freq=1000,
        alpha=0.8,
        log_dir=None,
        max_samples=20000,
        **kwargs
    ):
        super(InverseDirichletWeightsAdapter, self).__init__(
            ode_loss_weights=ode_loss_weights,
            data_generator=data_generator,
            freq=freq,
            alpha=alpha,
            log_dir=log_dir,
            max_samples=max_samples,
            **kwargs
        )

    # Handling loss weights
    ###########################################################################
    def update_loss_weights(self):
        new_weights = self.eval_loss_weights()
        for data_id, loss in self.model.ode_loss_weights.items():
            weight = self.alpha * new_weights[data_id] + \
                (1. - self.alpha) * self.model.ode_loss_weights[data_id]
            self.model.ode_loss_weights[data_id] = weight.numpy()

    def eval_loss_weights(self):
        _var_grads = self.var_grads()
        return {data_id: max(_var_grads.values()) / var \
            for data_id, var in _var_grads.items()}

    def var_grads(self):
        _var = {}
        for data_id, grad in self.loss_gradients.items():
            _grad = np.concatenate([ g.numpy().flatten() for g in grad ])
            _var[data_id] = _grad.var()
        return _var


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
        ode_loss_weights=None,
        data_generator=None,
        freq=None,
        log_dir=None,
        max_samples=20000,
        **kwargs
    ):
        super(NTKWeightsAdapter, self).__init__(
            ode_loss_weights=ode_loss_weights,
            data_generator=data_generator,
            freq=freq,
            log_dir=log_dir,
            max_samples=max_samples,
            **kwargs
        )

    def on_train_begin(self, logs={}):
        self.init_weights()
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
        self.check_weights()

    # Handling loss gradients
    ###########################################################################
    def get_ntk(self):
        pred, losses = {}, {}
        with backprop.GradientTape(persistent=True) as tape:
            for data_id in self.model.data_ids:
                pred[data_id], losses[data_id] = self.get_pred_loss( \
                    *self.dataset[data_id], data_id)
                # Flatten prediction matrix into 1D vector
                if isinstance(pred[data_id], (list,tuple)):
                    pred[data_id] = list(itertools.chain(
                        *[ tf.split(tf.reshape(p, [-1]), tf.size(p).numpy()) \
                            for p in pred[data_id] ])
                    )
                else:
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
            self.model.ode_loss_weights[data_id] = normalization_grad / np.sum(ntk[data_id])


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
#             self.model.ode_loss_weights[data_id] = normalization_grad / tf.math.reduce_sum(ntk[data_id])
#         print(self.model.ode_loss_weights)
