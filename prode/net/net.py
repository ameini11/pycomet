import tensorflow as tf


class Net(tf.keras.Model):
    """Network base class."""

    # Initialization
    ###########################################################################
    def __init__(self, name='net'):
        super(Net, self).__init__()
        self.training     = True     # Boolean value for training other layers parameters

        self.inp_trans    = None
        self.out_trans    = None

        self.res_loss_fn  = None
        self.res_loss     = 0.
        self.res_val_loss = 0.


    # Configuration update
    ###########################################################################
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'training':     self.training,
            'inp_trans':    self.inp_trans,
            'out_trans':    self.out_trans,
            'res_loss_fn':  self.res_loss_fn,
            'res_loss':     self.res_loss,
            'res_val_loss': self.res_val_loss
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    # Functions
    ###########################################################################
    def apply_feature_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features)."""
        self.inp_trans = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs)."""
        self.out_trans = transform

    def add_residual_evaluation(self, loss_fn):
        """Add the residual loss function to loss list."""
        self.res_loss_fn = loss_fn
