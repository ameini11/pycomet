import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='prode', name='Net')
class Net(tf.keras.Model):
    """Network base class."""

    # Initialization
    ###########################################################################
    def __init__(self, name='net'):
        super(Net, self).__init__()
        self.training     = True     # Boolean: cen be used for training specific layers parameters

        self.inp_trans    = None
        self.out_trans    = None

        self.res_loss_fn  = None


    # Configuration update
    ###########################################################################
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'training':     self.training,
            'inp_trans':    self.inp_trans,
            'out_trans':    self.out_trans,
            'res_loss_fn':  self.res_loss_fn
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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

    def add_residual_loss(self, loss_fn):
        """Add the residual loss function to loss list."""
        self.res_loss_fn = loss_fn
