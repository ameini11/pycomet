import tensorflow as tf
import tensorflow.keras.layers as lay

from .. import layers
from . import activations
from . import initializers
from . import regularizers

from .func_net import FuncNet


class FuncOpNN(FuncNet):
    """Deep Operator neural networks with Keras Functional API."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        u_dim,
        y_dim,
        G_dim,
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
        super(FuncOpNN, self).__init__()
        
        ### Hyperparameters Initialization ------------------------------------
        self.name  = name

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
        self.regularizer        = regularizers.get(**regularization)
        self.dropout_rate       = dropout_rate

        if isinstance(activation, dict):
            self.batch_norm_branch = batch_norm["branch"]
            self.batch_norm_trunk  = batch_norm["trunk"]
        else:
            self.batch_norm_branch = self.batch_norm_trunk = batch_norm

        self.batch_norm    = batch_norm
        self.weight_norm   = weight_norm
        self.use_end_scale = use_end_scale

        ### Blocks Initialization ---------------------------------------------
        self.branches   = [ layers.FFBlock(
                                self.layer_size_branch,
                                self.act_branch,
                                self.kernel_initializer,
                                regularization=self.regularizer,
                                dropout_rate=self.dropout_rate,
                                batch_norm=self.batch_norm_branch,
                                weight_norm=self.weight_norm,
                                name='Branch_'+str(i)) for i in range(self.G_dim) ]

        self.trunk      =   layers.FFBlock(
                                self.layer_size_trunk,
                                self.act_trunk,
                                self.kernel_initializer,
                                regularization=self.regularizer,
                                dropout_rate=self.dropout_rate,
                                batch_norm=self.batch_norm_trunk,
                                weight_norm=self.weight_norm,
                                name='Trunk')

        self.end_scale  =   lay.Dense(
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
                                name='FinalScale') if self.use_end_scale else None


    ### Model Initialization (Functional API) -----------------------------
    def get_uncompiled_model(self):

        # Input layer
        x_u = tf.keras.Input(shape=(self.u_dim,), name="branch_input")
        x_y = tf.keras.Input(shape=(self.y_dim,), name="trunk_input")

        # Input Transformation
        if self.inp_trans is not None:
            if isinstance(self.inp_trans, dict):
                x_u = self.inp_trans["branch"](x_u)
                x_y = self.inp_trans["trunk"](x_y)
            else:
                x_u = self.inp_trans(x_u)
                x_y = self.inp_trans(x_y)

        # Branch/-es
        G_u = [ self.branches[i](x_u) for i in range(self.G_dim) ]

        # Trunk
        G_y = self.trunk(x_y)

        # Output layer
        G_uy = [ lay.Dot(axes=1)([G_u[i], G_y]) for i in range(self.G_dim) ]
        G_uy = lay.Concatenate()(G_uy)
        if self.use_end_scale:
            G_uy = self.end_scale(G_uy)

        if self.out_trans is not None:
            G_uy = self.out_trans(G_uy)

        # Keras Model
        model = tf.keras.Model(inputs=[x_u, x_y], outputs=G_uy, name=self.name)

        return model
