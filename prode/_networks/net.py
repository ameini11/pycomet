import tensorflow as tf

from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras import metrics as metrics_mod
from keras.engine import data_adapter
from keras import backend as K

from prode.utilities.model import layers
from prode.utilities import utils


@tf.keras.utils.register_keras_serializable(package='PrODE', name='Net')
class Net(tf.keras.Model):
    """Network base class."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        block='FFBlock',
        name='net',
        *args,
        **kwargs
    ):
        super(Net, self).__init__(name=name, *args, **kwargs)

        if not isinstance(dim, dict):
            utils.raise_value_err(
                "`dim` passed into `{}` network has "
                "to be a `dict` of length 2:\n"
                "  - 'inp': `int` (or list of `int`);\n"
                "  - 'out': `int` (or list of `int`).\n"
                "It represents the dimensions of the "
                "feature space.".format(self.name)
            )
        self.inp_dim = dim["inp"]
        self.out_dim = dim["out"]

        self.block = block if callable(block) \
            else utils.get_class(layers, block)

        self.inp_trans = None
        self.out_trans = None

        # Extra features for handling physics-informed multi-objective loss
        self.multi_obj_loss = False
        self.objectives = None
        self.residual = None
        self.losses_ids = ['weighted', 'unweighted']

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'inp_dim': self.inp_dim,
            'out_dim': self.out_dim,
            'block': self.block,
            'inp_trans': self.inp_trans,
            'out_trans': self.out_trans,
            'multi_obj_loss': self.multi_obj_loss,
            'objectives': self.objectives,
            'residual': self.residual
        }
        base_config = super(Net, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=False):
        return x

    # Building
    ###########################################################################
    def build(self, input_shape):
        super(Net, self).build(input_shape)
        # I/O characterization
        self.inputs = tf.nest.map_structure(
            lambda i: K.placeholder(shape=[None,i], dtype=K.floatx()),
            input_shape
        )
        self.outputs = self.call(self.inputs, training=False)
        # I/O features space dimensions
        self.input_dim = [i.shape[1] for i in self.inputs]
        self.output_dim = [o.shape[1] for o in self.outputs]
        # Network graph
        self.graph = tf.keras.Model(self.inputs, self.outputs, name=self.name)

    # Compiling
    ###########################################################################
    def compile(
        self,
        optimizer='rmsprop',
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs
    ):
        if self.multi_obj_loss:
            # Check loss functions and weights
            self.check_arg(loss, 'loss functions')
            self.check_arg(loss_weights, 'loss weights')
            # Other preliminary operations
            base_layer.keras_api_gauge.get_cell('compile').set(True)
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
                    name=obj + '_loss'
                )
            # Build metrics container
            if metrics is not None:
                utils.warning(
                    "With multi-objective loss function, "
                    "metrics evaluation is not available."
                )
            self.compiled_metrics = None
            # Configure steps
            self._configure_steps_per_execution(steps_per_execution or 1)
            # Initialize attrs that are reset each time `compile` is called.
            self._reset_compile_cache()
            self._is_compiled = True
            self.loss = loss or {}
        else:
            return super(Net, self).compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                loss_weights=loss_weights,
                weighted_metrics=weighted_metrics,
                run_eagerly=run_eagerly,
                steps_per_execution=steps_per_execution,
                **kwargs
            )

    def check_arg(self, arg, definition):
        if arg and not isinstance(arg, dict):
            arg = nameof(arg)
            utils.raise_value_err(
                f"`{arg}` passed into `compile` has to be a `dict`:\n"
                f"  - 'ics': initial condition pts {definition};\n"
                f"  - 'pts': additional anchor pts {definition};\n"
                f"  - 'res': collocation pts residual {definition}."
            )

    @property
    def metrics(self):
        metrics = []
        if self._is_compiled:
            if self.compiled_loss is not None:
                if isinstance(self.compiled_loss, dict):
                    for container in self.compiled_loss.values():
                        metrics += container.metrics
                else:
                    metrics += self.compiled_loss.metrics
            if self.compiled_metrics is not None:
                metrics += self.compiled_metrics.metrics
        return metrics

    # Steps
    ###########################################################################
    # Training Step
    def train_step(self, data):
        if self.multi_obj_loss:
            return self.multi_obj_train_step(data)
        else:
            return super(Net, self).train_step(data)

    # def multi_obj_train_step(self, data):
    #     # Unpack data
    #     x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
    #     # Manipulate sample weights
    #     if self.weighted_loss:
    #         sample_weight = self.weighted_loss.on_grad_step_begin(
    #             sample_weight
    #         )
    #     # Initialize loss variables
    #     losses = {i: {} for i in self.losses_ids}
    #     loss = {i: 0. for i in self.losses_ids}
    #     # Gradient descent step
    #     with backprop.GradientTape(persistent=True) as tape:
    #         # Manipulate sample weights
    #         if self.weighted_loss:
    #             sample_weight = self.weighted_loss.on_grad_step_eval(
    #                 sample_weight
    #             )
    #         # Evaluate each objective function loss
    #         for obj in self.objectives['train']:
    #             # Get predictions
    #             y_pred = self.predict_step(
    #                 x[obj], objective=obj, training=True
    #             )
    #             # Evaluate losses
    #             losses['unweighted'][obj] = self.compiled_loss[obj](
    #                 y[obj], y_pred, sample_weight=None
    #             )
    #             if self.weighted_loss:
    #                 losses['weighted'][obj] = self.compiled_loss[obj](
    #                     y[obj], y_pred, sample_weight=sample_weight
    #                 )
    #         # Regularization loss
    #         if self.losses:
    #             reg_loss = {
    #                 'reg': tf.add_n(
    #                     losses_utils.cast_losses_to_common_dtype(self.losses)
    #                 )
    #             }
    #             losses['weighted'].update(reg_loss)
    #         # Evaluate total loss
    #         for i in self.losses_ids:
    #             loss[i] = tf.add_n(losses[i].values())
    #     # Update model parameters
    #     self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    #     # Update loss weights
    #     if self.weighted_loss:
    #         self.weighted_loss.on_grad_step_end(
    #             loss, sample_weight=sample_weight, tape=tape
    #         )
    #     # Collect metrics to return
    #     metrics = {'loss': loss}
    #     metrics.update({obj+'_loss': l for obj, l in losses.items()})
    #     return metrics

    def multi_obj_train_step(self, data):
        # Unpack data
        x, y, sample_weight, sample_index = utils.unpack_x_y_sample_weight_index(data)
        # Manipulate sample weights
        if self.weighted_loss:
            sample_weight = self.weighted_loss.on_grad_step_begin(
                sample_weight
            )
        # Gradient descent step
        with backprop.GradientTape(persistent=True) as tape:
            # Manipulate sample weights
            if self.weighted_loss:
                sample_weight = self.weighted_loss.on_grad_step_eval(
                    sample_weight
                )
            # Evaluate losses
            losses = self.eval_losses(x, y, sample_weight)
            # Regularization loss
            reg_loss = self.eval_reg_loss()
            # Evaluate total loss
            loss = {i: tf.add_n(l.values()) for i, l in losses.items()}
            # Define the loss to be optimized
            loss_opt = loss['weighted'] if self.weighted_loss \
                else loss['unweighted']
            if reg_loss:
                loss_opt += reg_loss
        # Update model parameters
        self.optimizer.minimize(
            loss_opt,
            self.trainable_variables,
            tape=tape
        )
        # Update sample weights
        if self.weighted_loss:
            self.weighted_loss.on_grad_step_end(
                loss_opt,
                sample_weight=sample_weight,
                sample_index=sample_index,
                tape=tape
            )
        # Collect metrics to return
        metrics = {'loss': loss['unweighted']}
        if self.weighted_loss:
            metrics.update({'weighted_loss': loss['weighted']})
        if reg_loss:
            metrics.update({'reg_loss': reg_loss})
        metrics.update({
            obj+'_loss': l for obj, l in losses['unweighted'].items()
        })
        return metrics

    def eval_losses(self, x, y, sample_weight):
        losses = {'unweighted': {}}
        if self.weighted_loss:
            losses['weighted'] = {}
        # Evaluate each objective function loss
        for obj in self.objectives['train']:
            # Get predictions
            y_pred = self.predict_step(
                x[obj], objective=obj, training=True
            )
            # Evaluate loss
            losses['unweighted'][obj] = self.compiled_loss[obj](
                y[obj], y_pred, sample_weight=None
            )
            if self.weighted_loss:
                losses['weighted'][obj] = self.compiled_loss[obj](
                    y[obj], y_pred, sample_weight=sample_weight[obj]
                )
        return losses

    def eval_reg_loss(self):
        if self.losses:
            return tf.add_n(
                losses_utils.cast_losses_to_common_dtype(self.losses)
            )

    # Testing Step
    def test_step(self, data):
        if self.multi_obj_loss:
            return self.multi_obj_test_step(data)
        else:
            return super(Net, self).test_step(data)

    def multi_obj_test_step(self, data):
        # Unpack data
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Evaluate each objective function loss
        losses = {}
        for obj in self.objectives['valid']:
            y_pred = self.predict_step(x[obj], objective=obj)
            sw = sample_weight[obj] if sample_weight else None
            losses[obj] = self.compiled_loss[obj](y[obj], y_pred, sw)
        # Evaluate total loss
        loss = tf.add_n(losses.values())
        # Collect metrics to return
        metrics = {'loss': loss}
        metrics.update({obj+'_loss': l for obj, l in losses.items()})
        return metrics

    # Predicting Step
    def predict_step(self, data=None, objective=None, training=False):
        if self.multi_obj_loss:
            return self.multi_obj_predict_step(data, objective, training)
        else:
            return super(Net, self).predict_step(data)

    def multi_obj_predict_step(self, data, objective, training):
        # Unpack data
        x, *_ = data_adapter.unpack_x_y_sample_weight(data)
        if x is None:
            x = self.inputs
        # Evaluate predictions for the specific objective function
        if objective == 'res':
            return self.residual(x, training=training)
        else:
            return self(x, training=training)


    # Input/Output transformations
    ###########################################################################
    @property
    def inp_trans(self):
        return self._inp_trans

    @inp_trans.setter
    def inp_trans(self, function):
        self._inp_trans = function

    @property
    def out_trans(self):
        return self._out_trans

    @out_trans.setter
    def out_trans(self, function):
        self._out_trans = function

    # Losses properties
    ###########################################################################
    @property
    def objectives(self):
        return self._objectives

    @objectives.setter
    def objectives(self, identifiers):
        self._objectives = identifiers

    @property
    def multi_obj_loss(self):
        return self._multi_obj_loss

    @multi_obj_loss.setter
    def multi_obj_loss(self, value):
        self._multi_obj_loss = value

    # Residual loss
    @property
    def residual(self):
        return self._residual

    @residual.setter
    def residual(self, function):
        self._residual = function
