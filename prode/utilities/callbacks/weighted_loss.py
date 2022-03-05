import sys
import copy
import itertools
import numpy as np
import joblib as jl
import tensorflow as tf
import tensorflow.python.keras.backend as K
import tensorflow.python.keras.callbacks as CB
import tensorflow.python.keras.initializers as tf_init

from tqdm import tqdm
from tensorflow.python.eager import backprop
from tensorflow.python.util.tf_export import keras_export
from prode.utilities import utils


@keras_export('keras.callbacks.BaseWeightsAdapter')
class BaseWeightsAdapter(CB.Callback):
    """Base class for adaptive multi-objective loss weights techniques."""

    def __init__(
        self,
        data_container=None,
        multi_obj_loss_weights=None,
        freq=1,
        log_dir=None,
        **kwargs
    ):
        super(BaseWeightsAdapter, self).__init__()

        self.name = utils.get_class_name(self)
        self.multi_obj_loss_weights = multi_obj_loss_weights
        self.freq = int(np.clip(freq, 1, None))
        self._print = True

        # Define data container
        self.data_container = data_container
        assert self.data_container is data_container
        # Define `logs` directory
        self.log_dir = log_dir

    # Callback methods --------------------------------------------------------
    def on_train_begin(self, logs={}):
        utils.print_submain(
            "The `{}` callback is used to balance "
            "the multi-objective loss.".format(self.name)
        )
        # Assign `weighted_loss` class to Keras model
        self.model.weighted_loss = self

    def on_epoch_end(self, epoch, logs={}):
        if self.multi_obj_loss_weights is not None and self._print:
            self.print_weights(epoch, logs)

    # Handling weights methods ------------------------------------------------
    def init_weights(self):
        if self.multi_obj_loss_weights is None:
            utils.warning(
                "No `multi_obj_loss_weights` have been provided to `{}`: "
                "the initial value of 1 will be applied.".format(self.name)
            )
            self.multi_obj_loss_weights = {
                obj: tf.ones(1, dtype=K.floatx()) \
                    for obj in self.model.objectives['train']
            }
        elif isinstance(self.multi_obj_loss_weights, (list,tuple)):
            self.multi_obj_loss_weights = dict(
                zip(
                    self.model.objectives['train'],
                    self.multi_obj_loss_weights
                )
            )
        elif isinstance(self.multi_obj_loss_weights, dict):
            pass
        else:
            utils.raise_value_err(
                "`multi_obj_loss_weights` passed into `{}` "
                "has to be a list or a dictionary:\n"
                "  - 'ics': weight for initial condition pts loss;\n"
                "  - 'pts': weight for anchor pts loss;\n"
                "  - 'res': weight for residual loss.".format(self.name)
            )
        self.multi_obj_loss_weights = utils.convert_struct_to_tf(
            self.multi_obj_loss_weights
        )
        for k, w in self.multi_obj_loss_weights.items():
            w = tf.reshape(w, shape=[-1,1]) if w.shape.rank < 1 else w
            self.multi_obj_loss_weights[k] = w
        
        if self.log_dir is not None:
            self.logs = {
                obj: tf.summary.create_file_writer(
                    self.log_dir + '/scalars/multi_obj_loss_weights/' + obj
                ) for obj in self.multi_obj_loss_weights
            }

    def print_weights(self, epoch, logs):
        weights = tf.nest.map_structure(
            lambda x: np.squeeze(x.numpy()), self.multi_obj_loss_weights
        )
        logs.update({obj + '_loss_weight': w for obj, w in weights.items()})
        if self.log_dir is not None:
            for obj, log in self.logs.items():
                with log.as_default():
                    tf.summary.scalar(
                        name='epoch_' + obj + '_loss_weight',
                        data=weights[obj],
                        step=epoch
                    )

    def check_weights(self, weights=None):
        if weights is None:
            weights = self.multi_obj_loss_weights
        nan = [tf.math.is_nan(w) for w in tf.nest.flatten(weights)]
        inf = [tf.math.is_inf(w) for w in tf.nest.flatten(weights)]
        if any(nan) or any(inf):
            print("")
            utils.warning(
                "One or multiple weights evaluated in `{}` class "
                "are `inf` or `nan`.".format(self.name)
            )
            if self.multi_obj_loss_weights is not None:
                utils.print_submain(
                    "Here the `multi_obj_loss_weights` values:"
                )
                for k, v in self.multi_obj_loss_weights.items():
                    utils.print_submain("  - '{}': {}".format(k,v))
                utils.print_submain("The training will be stopped.")
                self.model.stop_training = True

    # Gradient descent methods ------------------------------------------------
    def on_grad_step_begin(self, sample_weight=None):
        return self.multi_obj_loss_weights

    def on_grad_step_eval(self, sample_weight=None):
        return self.multi_obj_loss_weights

    def on_grad_step_end(
        self,
        loss=0.,
        sample_weight=None,
        sample_index=None,
        tape=None
    ):
        pass


@keras_export('keras.callbacks.ConstantWeightsAdapter')
class ConstantWeightsAdapter(BaseWeightsAdapter):
    """Constant loss weights techniques."""

    def __init__(
        self,
        data_container=None,
        multi_obj_loss_weights=None,
        freq=1,
        log_dir=None,
        **kwargs
    ):
        super(ConstantWeightsAdapter, self).__init__(
            data_container=data_container,
            multi_obj_loss_weights=multi_obj_loss_weights,
            freq=freq,
            log_dir=log_dir,
            **kwargs
        )
        self._print = False

    # Callback methods --------------------------------------------------------
    def on_train_begin(self, logs={}):
        super(ConstantWeightsAdapter, self).on_train_begin(logs=logs)
        self.init_weights()


@keras_export('keras.callbacks.EmpiricalWeightsAdapter')
class EmpiricalWeightsAdapter(BaseWeightsAdapter):
    """Adaptive loss weights technique for PINNs based on an empirical learning
    rate annealing algorithm.

    See:
        Sifan Wang, Yujun Teng, Paris Perdikaris
        `Understanding and mitigating gradient pathologies in physics-informed
        neural networks`
        https://arxiv.org/abs/2001.04536"""

    def __init__(
        self,
        data_container=None,
        multi_obj_loss_weights=None,
        freq=1000,
        alpha=0.8,
        max_samples=20000,
        method='mean',
        log_dir=None,
        **kwargs
    ):
        super(EmpiricalWeightsAdapter, self).__init__(
            data_container=data_container,
            multi_obj_loss_weights=multi_obj_loss_weights,
            freq=freq,
            log_dir=log_dir,
            **kwargs
        )

        self.alpha = tf.cast(tf.clip_by_value(alpha, 0., 1.), K.floatx())
        self.gradients = {}
        self.dataset = {}
        self.max_samples = max_samples


        self.mean = lambda x: tf.reduce_mean(
            [tf.reduce_mean(tf.math.abs(i)) for i in x]
        )
        self.max = lambda x: tf.reduce_max(
            [tf.reduce_max(tf.math.abs(i)) for i in x]
        )
        self.grads_eval = self.mean if method == 'mean' else self.max

        self._update = True
        self.check_residual = True

    # Callback methods --------------------------------------------------------
    def on_train_begin(self, logs={}):
        super(EmpiricalWeightsAdapter, self).on_train_begin(logs=logs)
        self.init_weights()
        if 'res' not in self.model.objectives['train'] and self.check_residual:
            utils.warning(
                "Network `{}` hasn't any residual loss to be evaluated: "
                "`{}` can't be applied.".format(self.model.name, self.name)
            )
            self._update = False
            self.model.weighted_loss = None

    def on_epoch_begin(self, epoch, logs={}):
        if self._update and epoch % self.freq == 0:
            self.update_weights()

    # Handling weights methods ------------------------------------------------
    def update_weights(self):
        self.sample_dataset()
        self.eval_loss_gradients()
        self.update_multi_obj_loss_weights()
        self.check_weights(self.multi_obj_loss_weights)

    def sample_dataset(self):
        for obj, dset in self.data_container.data.items():
            n_samples = self.data_container.n_samples[obj]
            indeces = np.random.permutation(n_samples)
            if n_samples > self.max_samples:
                indeces = np.random.choice(
                    indeces, self.max_samples, replace=False
                )
            self.dataset[obj] = tf.nest.map_structure(
                lambda z: tf.convert_to_tensor(z[indeces], dtype=K.floatx()),
                dset
            )

    def eval_loss_gradients(self):
        losses = {}
        with backprop.GradientTape(persistent=True) as tape:
            for obj in self.model.objectives['train']:
                x, y = self.dataset[obj][:2]
                # Get predictions
                y_pred = self.model.multi_obj_pred(x, objective=obj)
                # Evaluate each objective function loss
                losses[obj] = self.model.compiled_loss[obj](y, y_pred)
        # Evaluate each objective function loss
        for obj in self.model.objectives['train']:
            self.gradients[obj] = tape.gradient(
                losses[obj], self.model.trainable_variables
            )

    def update_multi_obj_loss_weights(self):
        # Manipulate residual gradients
        grad_res = self.grads_eval(self.gradients['res'])
        new_weights = {}
        for obj, grad in self.gradients.items():
            if obj == 'res':
                new_weights[obj] = self.multi_obj_loss_weights['res']
                continue
            # Manipulate gradients of the remaining objective loss functions
            grads_obj = self.mean(grad)
            # Evaluate new multi-objective loss weights
            new_weights[obj] = grad_res / grads_obj \
                * self.multi_obj_loss_weights['res']
        # Update multi-objective loss weights
        self.multi_obj_loss_weights = tf.nest.map_structure(
            lambda old, new: self.alpha*new + (1. - self.alpha)*old,
            self.multi_obj_loss_weights,
            new_weights
        )


@keras_export('keras.callbacks.InverseDirichletWeightsAdapter')
class InverseDirichletWeightsAdapter(EmpiricalWeightsAdapter):
    """Adaptive loss weights technique for PINNs based on Inverse-Dirichlet
    weighting."""

    def __init__(
        self,
        data_container=None,
        multi_obj_loss_weights=None,
        freq=1000,
        alpha=0.8,
        max_samples=20000,
        method='std',
        log_dir=None,
        **kwargs
    ):
        super(InverseDirichletWeightsAdapter, self).__init__(
            data_container=data_container,
            multi_obj_loss_weights=multi_obj_loss_weights,
            freq=freq,
            alpha=alpha,
            max_samples=max_samples,
            log_dir=log_dir,
            **kwargs
        )

        self.grads_eval = np.var if method == 'var' else np.std
        self.check_residual = False

    # Handling weights methods ------------------------------------------------
    def update_multi_obj_loss_weights(self):
        # Evaluate gradients `var` or `std`
        sigmas = {}
        for obj, grad in self.gradients.items():
            sigmas[obj] = self.grads_eval(
                np.concatenate([ g.numpy().flatten() for g in grad ])
            )
        # Evaluate new multi-objective loss weights
        new_weights = {
            obj: max(sigmas.values())/sigma for obj, sigma in sigmas.items()
        }
        # Update multi-objective loss weights
        self.multi_obj_loss_weights = tf.nest.map_structure(
            lambda old, new: self.alpha*new + (1. - self.alpha)*old,
            self.multi_obj_loss_weights,
            new_weights
        )

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
#         ode_loss_weights=None,
#         data_container=None,
#         freq=None,
#         log_dir=None,
#         max_samples=1000,
#         **kwargs
#     ):
#         super(NTKWeightsAdapter, self).__init__(
#             ode_loss_weights=ode_loss_weights,
#             data_container=data_container,
#             freq=freq,
#             log_dir=log_dir,
#             max_samples=max_samples,
#             **kwargs
#         )
#         self.pred_fn = {}

#     def on_train_begin(self, logs={}):
#         self.init_weights()
#         self.set_pred_fn()

#     def update(self):
#         self.sample_dataset()
#         ntk = self.get_ntk()
#         self.update_loss_weights(ntk)
#         self.check_weights()

#     # Handling predictions
#     ###########################################################################
#     def set_pred_fn(self):
#         for data_id in self.model.data_ids:
#             if data_id == 'res':
#                 _pred_fn = self.model.residual
#             else:
#                 _pred_fn = self.model.call
#             self.pred_fn[data_id] = _pred_fn

#     # Handling loss gradients
#     ###########################################################################
#     def flatten(self, x):
#         # Flatten array or list of arrays into 1D vector
#         if isinstance(x, (list,tuple)):
#             return list(itertools.chain(
#                 *[tf.split(tf.reshape(i,[-1]), tf.size(i).numpy()) for i in x]
#             ))
#         else:
#             return tf.split(tf.reshape(x,[-1]), tf.size(x).numpy())

#     def get_ntk(self):
#         pred = {}
#         with backprop.GradientTape(persistent=True) as tape:
#             for data_id in self.model.data_ids:
#                 x = self.dataset[data_id][0]
#                 pred[data_id] = self.flatten(
#                     self.pred_fn[data_id](
#                         self.dataset[data_id][0], training=False
#                     )
#                 )
#         return self.eval_diag_ntk(pred, tape)

#     def eval_diag_ntk(self, pred, tape):
#         utils.print_submain("From `NTKWeightsAdapter` callback:")
#         ntk = {}
#         for data_id, pred_i in pred.items():
#             utils.print_submain("  - Evaluating NTK matrix for `{}` data "
#                 "points".format(data_id))
#             ta = tf.TensorArray(
#                 dtype=K.floatx(),
#                 size=0,
#                 dynamic_size=True,
#                 clear_after_read=False
#             )
#             for i, elem in enumerate(tqdm(pred_i, file=sys.stdout)):
#                 grad = tape.gradient(
#                     elem,
#                     self.model.trainable_variables,
#                     unconnected_gradients='zero'
#                 )
#                 tr = tf.add_n([tf.reduce_sum(tf.multiply(g, g)) for g in grad])
#                 ta = ta.write(i, tr)
#             ntk[data_id] = ta.stack()
#         return ntk

#     # Handling loss weights
#     ###########################################################################
#     def update_loss_weights(self, ntk):
#         trace = {
#             data_id: tf.reduce_sum(ntk_i).numpy() \
#                 for data_id, ntk_i in ntk.items()
#         }
#         norm_grad = tf.reduce_sum(list(trace.values())).numpy()
#         for data_id in self.model.data_ids:
#             self.model.ode_loss_weights[data_id] = norm_grad / trace[data_id]



# @keras_export('keras.callbacks.SoftAttentionWeightsAdapter')
# class SoftAttentionWeightsAdapter(BaseWeightsAdapter):
#     """Base class for adaptive loss weights techniques."""

#     def __init__(
#         self,
#         data_container=None,
#         trainable=True,
#         rank=1,
#         limits=[1.,1.],
#         mask=None,
#         q=None,
#         freq=1,
#         log_dir=None,
#         **kwargs
#     ):
#         super(SoftAttentionWeightsAdapter, self).__init__(
#             data_container=data_container,
#             freq=freq,
#             log_dir=log_dir,
#             **kwargs
#         )
#         # Characterize weights tensors
#         self.trainable = trainable
#         self.rank = rank

#         # Define weights initializers
#         self.limits = limits
#         if isinstance(limits, list):
#             self.limits = {
#                 obj: limits for obj in self.model.objectives['train']
#             }
#         if not isinstance(self.limits, dict):
#             utils.raise_value_err(
#                 "`limits` in `{}` has to be a `dict` of [min, max] value "
#                 "list for each objecitve function.".format(self.name)
#             )
#         self.initializer = {
#             obj: tf_init.RandomUniform(*l) for obj, l in self.limits.items()
#         }

#         # Define the self-adaptation mask function
#         if q is not None:
#             if any([q_i <= 0. for q_i in q]):
#                 utils.raise_value_err(
#                     "`q` coefficents for the mask function in `{}` have "
#                     "to be positive.".format(self.name)
#                 )
#         else:
#             q = [1., 1.]
#         if mask == 'poly':
#             self.mask_fn = lambda x: q[0]*tf.pow(tf.math.abs(x), q[1])
#         elif mask == 'sigmoid':
#             self.mask_fn = lambda x: q[0]*tf.math.sigmoid(q[1]*x)
#         elif mask == 'exp':
#             self.mask_fn = lambda x: q[0]*tf.math.exp(q[1]*x)
#         else:
#             self.mask_fn = tf.square

#         self._update = True
#         self.init_weights()

#     # Callback methods --------------------------------------------------------
#     def on_train_begin(self, logs={}):
#         super(SoftAttentionWeightsAdapter, self).on_train_begin(logs=logs)
#         # Get indipendent model optimizer for each objective function
#         self.optimizer = {
#             obj: copy.deepcopy(self.model.optimizer) \
#                 for obj in self.model.objectives['train']
#         }

#     def on_epoch_begin(self, epoch, logs={}):
#         if epoch % self.freq == 0:
#             self._update = True
#         else:
#             self._update = False

#     # Handling weights methods ------------------------------------------------
#     def init_weights(self):
#         # Build sample weights
#         self.data_container.init_sample_weight(
#             self.initializer, self.trainable, self.rank
#         )
#         # Shuffle/Batch data
#         self.data_container.keep_indeces = True
#         self.data_container.on_epoch_end()

#     # Gradient descent methods ------------------------------------------------
#     def on_grad_step_begin(self, sample_weight=None):
#         if self._update and sample_weight is not None:
#             return tf.nest.map_structure(
#                 lambda sw: tf.Variable(sw, trainable=self.trainable),
#                 sample_weight
#             )

#     def on_grad_step_eval(self, sample_weight=None):
#         if sample_weight is not None:
#             return tf.nest.map_structure(self.mask_fn, sample_weight)

#     def on_grad_step_end(
#         self,
#         loss=0.,
#         sample_weight=None,
#         sample_index=None,
#         tape=None
#     ):
#         if self._update and self.trainable and sample_weight is not None:
#             for obj, sw in sample_weight.items():
#                 # Update batch sample weights
#                 sw = tf.nest.flatten(sw)
#                 grads = tf.nest.map_structure(
#                     lambda x: -tape.gradient(loss, x), sw
#                 )
#                 self.optimizer[obj].apply_gradients(zip(grads, sw))
#                 # Assign the updated sample weights to data container
#                 tf.nest.map_structure(
#                     lambda sw_old, sw_new: sw_old.scatter_nd_update(
#                         sample_index[obj], sw_new
#                     ),
#                     self.data_container[obj][-1],
#                     sw
#                 )
