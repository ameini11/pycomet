import tensorflow as tf

from prode.net import weighted_loss
from prode.net import _steps
from prode     import utils
from prode     import layers
from prode     import losses as losses_module


@tf.keras.utils.register_keras_serializable(package='PrODE', name='Net')
class Net(tf.keras.Model):
    """Network base class."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        block='FFBlock',
        adaptive_weights='LossWeightsAdapter',
        name='net'):
        super(Net, self).__init__()

        if not isinstance(dim, dict):
            utils.raise_value_err(
                "`dim` passed into `{}` network has to be a `dict` of length 2:\n"
                "  - 'inp': `int` (or list of `int`);\n"
                "  - 'out': `int` (or list of `int`).".format(self.name))
        self.inp_dim = dim["inp"]
        self.out_dim = dim["out"]

        self.block = utils.get_class(layers, block)

        self.adaptive_weights = utils.get_class(weighted_loss, adaptive_weights)

        self.inp_trans = None
        self.out_trans = None

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
            'adaptive_weights': self.adaptive_weights,
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


    # Functions
    ###########################################################################
    def apply_input_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features)."""
        self.inp_trans = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs)."""
        self.out_trans = transform

    def add_losses(self, loss_fns, data_ids=None):
        """Add a `list`/`dict` of loss functions for composite loss."""
        if isinstance(loss_fns, dict):
            self.loss_fns = loss_fns
        elif isinstance(loss_fns, (list,tuple)):
            if isinstance(data_ids, (list,tuple)):
                if len(loss_fns) == len(data_ids):
                    self.loss_fns = {data_ids[i]: loss_fns[i] for i in range(len(loss_fns))}
                else:
                    utils.raise_value_err(
                        "`loss_fns` and `data_ids` passed into `{}.add_losses` "
                        "method have to be of the same length.".format(self.name))
            elif isinstance(data_ids, dict):
                self.loss_fns = {data_id: loss_fns[i] for i, data_id in enumerate(data_ids)}
            else:
                utils.raise_value_err(
                    "`data_ids` passed into `{}.add_losses` method has "
                    "to be a `dict` or a `list`/`tuple`.".format(self.name))
        else:
            utils.raise_value_err(
                "`loss_fns` passed into `{}.add_losses` method has "
                "to be a `dict` or a `list`/`tuple`.".format(self.name))


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