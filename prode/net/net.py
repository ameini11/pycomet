from tensorflow import tf


class Net(tf.keras.Model):
    """Network base class."""

    # Initialization
    ###########################################################################
    def __init__(self):
        super(Net, self).__init__()
        self.training   = True     # Boolean value for training other layers parameters
        self.dropout    = True     # Boolean value for training dropout layers parameters

        self.inp_trans  = None
        self.out_trans  = None

        self.res_loss   = None

    # Functions
    ###########################################################################
    def apply_feature_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features).
        """
        self.inp_trans = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """
        self.out_trans = transform

    def add_residual_evaluation(self, loss_fn):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """
        self.res_loss = loss_fn
