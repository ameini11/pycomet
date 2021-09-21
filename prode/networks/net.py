import tensorflow as tf

from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras import metrics as metrics_mod

from prode.utilities.model import layers
from prode.utilities import utils
from prode.networks import steps


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
                "`dim` passed into `{}` network has to be a `dict` of length 2:\n"
                "  - 'inp': `int` (or list of `int`);\n"
                "  - 'out': `int` (or list of `int`).\n"
                "It represents the dimensions of the feature space.".format(self.name)
            )
        self.inp_dim = dim["inp"]
        self.out_dim = dim["out"]

        self.block = block if callable(block) else utils.get_class(layers, block)

        self.inp_trans = None
        self.out_trans = None

        # NN extra features for handling ODE data types
        self.trainable_weights = None
        self.residual = None
        self.data_ids = None
        self.data_ids_valid = None


    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'inp_dim': self.inp_dim,
            'out_dim': self.out_dim,
            'block': self.block,
            'inp_trans': self.inp_trans,
            'out_trans': self.out_trans,
            'trainable_weights': self.trainable_weights,
            'residual': self.residual,
            'data_ids': self.data_ids,
            'data_ids_valid': self.data_ids_valid
        }
        base_config = super(Net, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x):
        return x

    # Compiling
    ###########################################################################
    def compile(
        self,
        data_cls,
        optimizer='rmsprop',
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs
    ):
        if data_cls.name == 'ODE':
            self.check_arg(loss, 'loss functions')
            self.check_arg(loss_weights, 'loss weights')

            base_layer.keras_api_gauge.get_cell('compile').set(True)

            from_serialized = kwargs.pop('from_serialized', False)

            self._validate_compile(optimizer, metrics, **kwargs)
            self._run_eagerly = run_eagerly

            # Defining optimizer
            self.optimizer = self._get_optimizer(optimizer)

            # Defining loss containers
            self.compiled_loss = {}
            for data_id in self.data_ids:
                if self.output_names:
                    _output_names = [ data_id + '_' + name for name in self.output_names ]
                else:
                    _output_names = data_id
                _loss = loss[data_id] if loss else None
                _loss_weights = loss_weights[data_id] if loss_weights else None
                self.compiled_loss[data_id] = compile_utils.LossesContainer(
                    _loss, _loss_weights, output_names=_output_names)
                self.compiled_loss[data_id]._loss_metric = metrics_mod.Mean(name=data_id + '_loss')

            # Defining metrics container
            if metrics is not None:
                utils.warning("With `data.ode`, metrics evaluation is not available.")
            self.compiled_metrics = None

            self._configure_steps_per_execution(steps_per_execution or 1)

            # Initializes attrs that are reset each time `compile` is called.
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
                **kwargs)

    def check_arg(self, arg, definition):
        if arg and not isinstance(arg, dict):
            msg_args = [ nameof(arg), *([definition]*3) ]
            utils.raise_value_err(
                "`{}` passed into `compile` has to be a `dict`:\n"
                "  - 'ics': initial condition pts {};\n"
                "  - 'pts': additional anchor pts {};\n"
                "  - 'res': collocation pts residual {}.".format(*msg_args)
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
        if self.residual:
            return steps.train_step(self, data)
        else:
            return super(Net, self).train_step(data)

    # Testing Step
    def test_step(self, data):
        if self.residual:
            return steps.test_step(self, data)
        else:
            return super(Net, self).test_step(data)


    # Apply I/O Transformation
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


    # def apply_input_transform(self, transform):
    #     """Compute the features by appling a transform to the network inputs, i.e.,
    #     features = transform(inputs). Then, outputs = network(features)."""
    #     self.inp_trans = transform

    # def apply_output_transform(self, transform):
    #     """Apply a transform to the network outputs, i.e.,
    #     outputs = transform(inputs, outputs)."""
    #     self.out_trans = transform


    # Data identities
    ###########################################################################
    @property
    def data_ids(self):
        return self._data_ids

    @data_ids.setter
    def data_ids(self, identifiers):
        self._data_ids = identifiers

    @property
    def data_ids_valid(self):
        return self._data_ids_valid

    @data_ids_valid.setter
    def data_ids_valid(self, identifiers):
        self._data_ids_valid = identifiers

    @property
    def trainable_weights(self):
        return self._trainable_weights

    @trainable_weights.setter
    def trainable_weights(self, weights):
        self._trainable_weights = weights


    # # Add loss properties
    # ###########################################################################
    @property
    def residual(self):
        return self._residual

    @residual.setter
    def residual(self, function):
        self._residual = function

    # def add_residual(self, function):
    #     """Add a `list`/`dict` of loss functions for composite loss."""
    #     self.residual = function

    # def add_loss_weights(self, trainable_weights):
    #     if trainable_weights is not None:
    #         self.trainable_weights = self.get_loss_properties(trainable_weights, 'trainable_weights')

    # def get_loss_properties(self, properties, prop_name):
    #     if isinstance(properties, dict):
    #         return properties
    #     elif isinstance(properties, (list,tuple)):
    #         if isinstance(self.data_ids, (list,tuple,dict)):
    #             if len(properties) == len(self.data_ids):
    #                 return {data_id: properties[i] for i, data_id in enumerate(self.data_ids)}
    #             else:
    #                 utils.raise_value_err(
    #                     "`{}` and `data.data_ids` passed into `{}.get_loss_properties` "
    #                     "method have to be of the same length.".format(prop_name, self.name))
    #         else:
    #             utils.raise_value_err(
    #                 "`data.data_ids` passed into `{}.get_loss_properties` method has "
    #                 "to be a `dict` or a `list`/`tuple`.".format(self.name))
    #     else:
    #         utils.raise_value_err(
    #             "`{}` passed into `{}.get_loss_properties` method has "
    #             "to be a `dict` or a `list`/`tuple`.".format(prop_name, self.name))
