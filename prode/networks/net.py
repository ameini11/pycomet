import tensorflow as tf

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
        super(Net, self).__init__(*args, **kwargs)

        if not isinstance(dim, dict):
            utils.raise_value_err(
                "`dim` passed into `{}` network has to be a `dict` of length 2:\n"
                "  - 'inp': `int` (or list of `int`);\n"
                "  - 'out': `int` (or list of `int`).\n"
                "It represents the dimensions of the feature space.".format(self.name))
        self.inp_dim = dim["inp"]
        self.out_dim = dim["out"]

        self.block = utils.get_class(layers, block)

        self.loss_weights = None

        self.inp_trans = None
        self.out_trans = None

        self.loss_fns = None
        self.data_ids = None
        self.data_ids_val = None


    # Configuration update
    ###########################################################################
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'inp_dim':          self.inp_dim,
            'out_dim':          self.out_dim,
            'block':            self.block,
            'loss_weights':     self.loss_weights,
            'inp_trans':        self.inp_trans,
            'out_trans':        self.out_trans,
            'loss_fns':         self.loss_fns,
            'data_ids':         self.data_ids,
            'data_ids_val':     self.data_ids_val
        })
        return config

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
        data_type,
        optimizer='rmsprop',
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs
    ):
        if data_type == 'ODE':
            self.check_arg(loss, 'loss functions')
            self.check_arg(loss_weights, 'loss weights')

            base_layer.keras_api_gauge.get_cell('compile').set(True)

            from_serialized = kwargs.pop('from_serialized', False)

            self._validate_compile(optimizer, metrics, **kwargs)
            self._run_eagerly = run_eagerly

            self.optimizer = self._get_optimizer(optimizer)
            self.compiled_loss = {}
            for data_id in self.data_ids:
                _output_names = [ data_id + '_' + name for name in self.output_names ]
                _loss = loss[data_id] if loss else None
                _loss_weights = loss_weights[data_id] if loss_weights else None
                if data_id == 'res':
                    self.compiled_loss[data_id] = ResidualLossContainer(
                        _loss, _loss_weights, output_names=_output_names)
                else:
                    self.compiled_loss[data_id] = compile_utils.LossesContainer(
                        _loss, _loss_weights, output_names=_output_names)
                self.compiled_loss[data_id]._loss_metric = metrics_mod.Mean(name=data_id + '_loss')

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
            format_args = [ nameof(arg), *([definition]*3) ]
            utils.raise_value_err(
                "`{}` passed into `compile` has to be a `dict`:\n"
                "  - 'ics': initial condition pts {};\n"
                "  - 'pts': additional anchor pts {};\n"
                "  - 'res': collocation pts residual {}.".format(*format_args))

    @property
    def metrics(self):
        metrics = []
        if self._is_compiled:
            if self.compiled_loss is not None:
                for container in self.compiled_loss.values():
                    metrics += container.metrics
        return metrics



    # Steps
    ###########################################################################
    # Training Step
    def train_step(self, data):
        if self.loss_fns:
            return steps.train_step(self, data)
        else:
            return super(Net, self).train_step(data)

    # Testing Step
    def test_step(self, data):
        if self.loss_fns:
            return steps.test_step(self, data)
        else:
            return super(Net, self).test_step(data)


    # Apply I/O Transformation
    ###########################################################################
    def apply_input_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features)."""
        self.inp_trans = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs)."""
        self.out_trans = transform


    # Data identities
    ###########################################################################
    def add_data_ids(self, data_cls):
        self.data_ids = data_cls.data_ids
        self.data_ids_val = data_cls.data_ids_val


    # Add loss properties
    ###########################################################################
    def add_loss_fun(self, loss_fns, data_cls):
        """Add a `list`/`dict` of loss functions for composite loss."""
        self.loss_fns = self.get_loss_properties(loss_fns, 'loss_fns')
        if 'res' in self.loss_fns:
            self.loss_fns['res'] = data_cls.residual_loss(self.loss_fns['res'])

    def add_loss_weights(self, loss_weights):
        if loss_weights is not None:
            self.loss_weights = self.get_loss_properties(loss_weights, 'loss_weights')

    def get_loss_properties(self, properties, prop_name):
        if isinstance(properties, dict):
            return properties
        elif isinstance(properties, (list,tuple)):
            if isinstance(self.data_ids, (list,tuple,dict)):
                if len(properties) == len(self.data_ids):
                    return {data_id: properties[i] for i, data_id in enumerate(self.data_ids)}
                else:
                    utils.raise_value_err(
                        "`{}` and `data.data_ids` passed into `{}.get_loss_properties` "
                        "method have to be of the same length.".format(prop_name, self.name))
            else:
                utils.raise_value_err(
                    "`data.data_ids` passed into `{}.get_loss_properties` method has "
                    "to be a `dict` or a `list`/`tuple`.".format(self.name))
        else:
            utils.raise_value_err(
                "`{}` passed into `{}.get_loss_properties` method has "
                "to be a `dict` or a `list`/`tuple`.".format(prop_name, self.name))
