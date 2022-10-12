import tensorflow as tf

from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.util import traceback_utils
from tensorflow.python.keras import backend
from keras.engine import base_layer
from pycomet import operations as ops
from pycomet import config as cfg
from pycomet import networks
from pycomet import utils
from varname import nameof


@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='Net')
class Net(tf.keras.Model):
    """Network base class."""

    # Initialization
    ###########################################################################
    def __init__( 
        self,
        dim=None,
        architecture=None,
        lays_name=None,
        lays_propty=None,
        *args,
        **kwargs
    ):
        # Initialize network
        self.architecture = architecture
        network_cls = utils.get_class(networks, self.architecture)
        super(Net, self).__init__(name=network_cls.__name__.lower())
        self.network = network_cls(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            *args,
            **kwargs
        )
        
        # Update attributes
        for k in (
            'dim', 'inputs_dim', 'outputs_dim', 'lays_name', 'lays_propty'
        ):
            setattr(self, k, getattr(self.network, k))

        # Extra attributes for handling multi-objective loss
        self.multi_obj_loss = False
        self.objectives = None
        self.obj_pred_fn = {}
        self.weighted_loss = None

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'dim': self.dim,
            'architecture': self.architecture,
            'lays_name': self.lays_name,
            'lays_propty': self.lays_propty
        }
        base_config = super(Net, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # Calling
    ###########################################################################
    def call(self, inputs, training=None, mask=None):
        return self.network.call(inputs, training=training, mask=mask)
        
    # Compiling
    ###########################################################################
    @traceback_utils.filter_traceback
    def compile(
        self,
        optimizer='adam',
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        jit_compile=None,
        extra_trainable_variables=None,
        **kwargs
    ):
        if extra_trainable_variables is not None:
            self.extra_trainable_variables = tf.nest.flatten(
                extra_trainable_variables
            )
            if any([
                not isinstance(v, tf.Variable) \
                    for v in self.extra_trainable_variables
            ]):
                utils.raise_value_err(
                    "All the 'extra_trainable_variables' "
                    "have to be a 'tf.Variable'."
                )
        if self.multi_obj_loss:
            # Check loss functions and weights
            for arg in (loss, loss_weights):
                self.check_arg(arg)
            base_layer.keras_api_gauge.get_cell('compile').set(True)
            with self.distribute_strategy.scope():
                # Other preliminary operations
                from_serialized = kwargs.pop('from_serialized', False)
                self._validate_compile(optimizer, metrics, **kwargs)
                self._run_eagerly = run_eagerly
                # Get optimizer
                self.optimizer = self._get_optimizer(optimizer)
                # Build loss containers
                self.compiled_loss = {}
                for obj in self.objectives['train']:
                    self.compiled_loss[obj] = compile_utils.LossesContainer(
                        loss[obj] if loss else None,
                        loss_weights[obj] if loss_weights else None
                    )
                    self.compiled_loss[obj]._loss_metric = metrics_mod.Mean(
                        name=obj + '_loss', dtype=backend.floatx()
                    )
                # Build metrics container
                if metrics is not None:
                    utils.warning(
                        "Metrics evaluation is not available "
                        "with multi-objective loss function."
                    )
                self.compiled_metrics = compile_utils.MetricsContainer(
                    metrics=None,
                    weighted_metrics=None,
                    output_names=self.output_names,
                    from_serialized=from_serialized,
                )
                # Steps per execution
                self._configure_steps_per_execution(steps_per_execution or 1)
                # Initializes attrs that are reset each time `compile` is called.
                self._reset_compile_cache()
                self._is_compiled = True
                self.loss = loss or {}
                if (self._run_eagerly or self.dynamic) and jit_compile:
                    utils.raise_value_err(
                        'You cannot enable `run_eagerly` and `jit_compile` '
                        'at the same time.'
                    )
                else:
                    self._jit_compile = jit_compile
        else:
            return super(Net, self).compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                loss_weights=loss_weights,
                weighted_metrics=weighted_metrics,
                run_eagerly=run_eagerly,
                steps_per_execution=steps_per_execution,
                jit_compile=jit_compile,
                **kwargs
            )
        
    def check_arg(self, arg):
        if arg is not None and not isinstance(arg, dict):
            utils.raise_value_err(
                f"`{nameof(arg)}` passed into `compile` has to be a `dict` "
                "with objective functions identifiers as keys."
            )

    @property
    def metrics(self):
        if self.multi_obj_loss:
            metrics = []
            if self._is_compiled and self.compiled_loss is not None:
                for container in self.compiled_loss.values():
                    metrics += [container._loss_metric]
            for l in self._flatten_layers():
                metrics.extend(l._metrics)
            return metrics
        else:
            return super(Net, self).metrics

    # Steps
    ###########################################################################
    # Training Step -----------------------------------------------------------
    def train_step(self, data):
        if self.multi_obj_loss:
            return self.multi_obj_train_step(data)
        else:
            return super(Net, self).train_step(data)

    def multi_obj_train_step(self, data):
        # Unpack data
        x, y, sample_weight, sample_index = ops.unpack_list(data, n_elem=4)
        # Manipulate sample weights
        if self.weighted_loss:
            sample_weight = self.weighted_loss.on_grad_step_begin(
                sample_weight
            )
        # Gradient descent step
        with tf.GradientTape(persistent=True) as tape:
            # Manipulate sample weights
            if self.weighted_loss:
                sample_weight = self.weighted_loss.on_grad_step_eval(
                    sample_weight
                )
            # Evaluate losses
            losses = self.compute_losses(x, y, sample_weight)
            # Regularization loss
            reg_loss = self.compute_reg_loss()
            # Evaluate total loss
            loss = tf.add_n(
                losses_utils.cast_losses_to_common_dtype(losses.values())
            )
            if reg_loss is not None:
                loss = loss + reg_loss
        # Update model parameters
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        # Update sample weights
        if self.weighted_loss:
            self.weighted_loss.on_grad_step_end(
                loss,
                sample_weight=sample_weight,
                sample_index=sample_index,
                tape=tape
            )
        # Collect metrics to return
        metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                metrics.update(result)
            else:
                metrics[metric.name] = result
        if reg_loss is not None:
            metrics.update({'reg_loss': reg_loss})
        metrics.update({'loss': tf.math.add_n(list(metrics.values()))})
        return metrics

    # Testing Step ------------------------------------------------------------
    def test_step(self, data):
        if self.multi_obj_loss:
            return self.multi_obj_test_step(data)
        else:
            return super(Net, self).test_step(data)

    def multi_obj_test_step(self, data):
        # Unpack data
        x, y, sample_weight = ops.unpack_list(data, n_elem=3)
        # Evaluate each objective function loss
        losses = self.compute_losses(x, y, sample_weight, training=False)
        # Collect metrics to return
        metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                metrics.update(result)
            else:
                metrics[metric.name] = result
        metrics.update({'loss': tf.math.add_n(list(metrics.values()))})
        return metrics

    # Predicting Step ---------------------------------------------------------
    def multi_obj_pred(self, x=None, objective=None, training=True):
        if x is None:
            x = self.inputs
        # Evaluate predictions for the specific objective function
        if callable(self.obj_pred_fn[objective]):
            return self.obj_pred_fn[objective](x, training=training)
        else:
            return self(x, training=training)

    # Compute losses ----------------------------------------------------------
    def compute_losses(self, x, y, sample_weight=None, training=True):
        phase = 'train' if training else 'valid'
        # Evaluate each objective function loss
        losses = {}
        for obj in self.objectives[phase]:
            # Get predictions
            y_pred = self.multi_obj_pred(
                x[obj], objective=obj, training=training
            )
            # Evaluate loss
            sw = sample_weight[obj] if sample_weight is not None else None
            losses[obj] = self.compiled_loss[obj](
                y[obj], y_pred, sample_weight=sw
            )
        return losses

    def compute_reg_loss(self):
        if self.losses:
            return tf.math.add_n(
                losses_utils.cast_losses_to_common_dtype(self.losses)
            )
