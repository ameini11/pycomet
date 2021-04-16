import tensorflow as tf
import tensorflow.keras.layers as lay

from . import activations
from . import initializers
from . import regularizers
from . import layers

from .net import FuncNet


class FuncOpNN(object):
    """Deep Operator neural networks with Keras Functional API"""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        u_dim=1,
        y_dim=1,
        G_dim=1,
        layer_size_branch,
        layer_size_trunk,
        activation,
        kernel_initializer,
        regularization=None,
        use_end_scale=True,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        name='func_opnn'):
        super(FuncOpNN, self).__init__(name=name)
        
        ### Hyperparameters Initialization ------------------------------------
        self.u_dim = u_dim
        self.y_dim = y_dim
        self.G_dim = G_dim

        self.layer_size_branch = layer_size_branch
        self.layer_size_trunk  = layer_size_trunk

        if isinstance(activation, dict):
            self.act_branch = activations.get(activation["branch"])
            self.act_trunk  = activations.get(activation["trunk"])
        else:
            self.act_branch = self.act_trunk = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer        = regularizers.get(regularization)
        self.dropout_rate       = dropout_rate
        if batch_norm not in [None, "before", "after"]:
            raise ValueError(
                "batch_norm == {}. Available choices: {{None|'before'|'after'}}.".format(
                    batch_norm
                )
            )

        self.batch_norm    = batch_norm
        self.weight_norm   = weight_norm
        self.use_end_scale = use_end_scale


        ### Blocks Initialization ---------------------------------------------
        self.branches   = [ layers.FNNBlock(
                                self.layer_size_branch,
                                self.act_branch,
                                self.kernel_initializer,
                                regularization=self.regularizer,
                                dropout_rate=self.dropout_rate,
                                batch_norm=self.batch_norm,
                                weight_norm=self.weight_norm,
                                name='Branch'+str(i)) for i in range(self.G_dim) ]

        self.trunk      =   layers.FNNBlock(
                                self.layer_size_trunk,
                                self.act_trunk,
                                self.kernel_initializer,
                                regularization=self.regularizer,
                                dropout_rate=self.dropout_rate,
                                batch_norm=self.batch_norm,
                                weight_norm=self.weight_norm,
                                name='Trunk')

        if self.use_end_scale:
            self.end_scale = lay.Dense(
                                    units=self.G_dim,
                                    activation='linear',
                                    use_bias=True,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer="zeros",
                                    kernel_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    bias_constraint=None,
                                    name='FinalScale'
                                )


        ### Model Initialization (Functional API) -----------------------------

        # Input layer
        self.x_u = tf.keras.Input(shape=(self.u_dim,))
        self.x_y = tf.keras.Input(shape=(self.y_dim,))

        if self.inp_trans is not None:
            if isinstance(self.inp_trans, dict):
                self.x_u = self.inp_trans["branch"](self.x_u)
                self.x_y = self.inp_trans["trunk"](self.x_y)
            else:
                self.x_u = self.inp_trans(self.x_u)
                self.x_y = self.inp_trans(self.x_y)

        # Branch/-es
        self.G_u = [ self.branches[i](self.x_u) for i in range(self.G_dim) ]

        # Trunk
        self.G_y = self.trunk(self.x_y) for i in range(self.G_dim)

        # Output layer
        self.G_uy = [ lay.Dot(axes=1)([G_u[i], G_y]) for i in range(self.G_dim) ]
        if self.use_end_scale:
            self.G_uy = self.end_scale(tf.concat(G_uy, 1))

        if self.out_trans is not None:
            self.G_uy = self.out_trans(self.G_uy)

        self.y = self.G_uy

        # Kera Model
        self.network = tf.keras.Model(inputs=[self.x_u, self.x_y], outputs=[self.y])

        if self.res_loss is not None:
            self.network.add_loss(self.res_loss(self, self.x_u, self.x_y), name='res_loss')
