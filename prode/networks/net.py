import tensorflow as tf

from prode.utilities.model_utils import layers
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
        loss_weights=None,
        name='net'):
        super(Net, self).__init__()

        if not isinstance(dim, dict):
            utils.raise_value_err(
                "`dim` passed into `{}` network has to be a `dict` of length 2:\n"
                "  - 'inp': `int` (or list of `int`);\n"
                "  - 'out': `int` (or list of `int`).\n"
                "It represents the dimensions of the feature space.".format(self.name))
        self.inp_dim = dim["inp"]
        self.out_dim = dim["out"]

        self.block   = utils.get_class(layers, block)

        self.loss_weights = loss_weights

        self.inp_trans    = None
        self.out_trans    = None

        self.loss_fns     = None
        self.data_ids     = None
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
        self.data_ids     = data_cls.data_ids
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
        else:
            self.loss_weights = None

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
