import tensorflow as tf
import tensorflow.keras.layers as lay

from prode.utilities.model import activations
from prode.utilities.model import initializers
from prode.utilities.model import regularizers
from prode.networks.net import Net
from prode.networks.opnn import OpNN
from prode.utilities import utils


@tf.keras.utils.register_keras_serializable(package='PrODE', name='MMOpNN')
class MMOpNN(Net):
    """Multi-Physics Multi-Scale Deep Operator neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        names,
        dim,
        layer_size,
        activation,
        kernel_initializer,
        inp_opnn_trans=None,
        out_opnn_trans=None,
        regularization=None,
        use_end_bias=True,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        block='FFBlock',
        name='mmopnn'
    ):
        super(MMOpNN, self).__init__(
            dim=dim,
            block=block,
            name=name
        )

        ### Hyperparameters Initialization ------------------------------------
        if len(names) <= 1:
            utils.raise_value_err(
                "The number of `names` passed into `{}` has to be grater "
                "than 1 (multiple DeepONets).".format(self.name))
        else:
            self.names = names

        self.dim = [ {
            "inp": self.inp_dim,
            "out": self.out_dim[i]
        } for i in range(len(self.names)) ]

        self.layer_size = layer_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.inp_opnn_trans = inp_opnn_trans
        self.out_opnn_trans = out_opnn_trans
        self.regularization = regularization
        self.use_end_bias = use_end_bias
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm

        ### OpNNs Initialization ----------------------------------------------
        self.opnns = [ OpNN(
            self.dim[i],
            self.layer_size[name],
            self.activation[name],
            self.kernel_initializer,
            regularization=self.regularization,
            use_end_bias=self.use_end_bias,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            weight_norm=self.weight_norm,
            block=self.block,
            name='opnn_'+name
        ) for i, name in enumerate(self.names) ]

        # Apply I/O Transformation
        if self.inp_opnn_trans:
            for i, name in enumerate(self.names):
                self.opnns[i].apply_input_transform(self.inp_opnn_trans[name])
        if self.out_opnn_trans:
            for i, name in enumerate(self.names):
                self.opnns[i].apply_output_transform(self.out_opnn_trans[name])


    # Configuration update
    ###########################################################################
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'names':              self.names,
            'dim':                self.dim,
            'layer_size':         self.layer_size,
            'activation':         self.activation,
            'kernel_initializer': self.kernel_initializer,
            'inp_opnn_trans':     self.inp_opnn_trans,
            'out_opnn_trans':     self.out_opnn_trans,
            'regularization':     self.regularization,
            'use_end_bias':       self.use_end_bias,
            'dropout_rate':       self.dropout_rate,
            'batch_norm':         self.batch_norm,
            'weight_norm':        self.weight_norm,
            'opnns':              self.opnns
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    # Calling
    ###########################################################################
    def call(self, x):

        # Inputs
        if isinstance(x, (list, tuple)):
            x_u, x_y = x[0], x[1]
        else:
            utils.raise_value_err(
                "The inputs of the `{}` net have to be a `list` "
                "or a `tuple`: [u, y]".format(self.name))

        # Input transformation
        if self.inp_trans is not None:
            if isinstance(self.inp_trans, dict):
                x_u = self.inp_trans["branch"](x_u)
                x_y = self.inp_trans["trunk"](x_y)
            else:
                x_u = self.inp_trans(x_u)
                x_y = self.inp_trans(x_y)

        # OpNNs
        G_uy = [ g_uy([x_u, x_y]) for g_uy in self.opnns ]


        # Output transformation
        if self.out_trans is not None:
            G_uy = self.out_trans(G_uy)
        
        return G_uy


    # Graph
    ###########################################################################
    def build_graph(self):
        x_u = tf.keras.Input(shape=(self.inp_dim[0],), name="branch_input")
        x_y = tf.keras.Input(shape=(self.inp_dim[1],), name="trunk_input")
        x   = [x_u, x_y]
        return tf.keras.Model(inputs=x, outputs=self.call(x), name=self.name)
