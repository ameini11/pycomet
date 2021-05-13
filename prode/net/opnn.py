import tensorflow as tf
import tensorflow.keras.layers as lay

from .. import layers
from .. import activations
from .. import initializers
from .. import regularizers
from . import steps

from .net import Net


@tf.keras.utils.register_keras_serializable(package='prode', name='OpNN')
class OpNN(Net):
    """Deep Operator neural network."""

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
        name='opnn'):
        super(OpNN, self).__init__(name=name)
        
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
        self.regularizer        = regularizers.get(**regularization)
        self.dropout_rate       = dropout_rate

        if isinstance(batch_norm, dict):
            self.batch_norm_branch = batch_norm["branch"]
            self.batch_norm_trunk  = batch_norm["trunk"]
        else:
            self.batch_norm_branch = self.batch_norm_trunk = batch_norm

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
                                name='Trunk_0')

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
                                name='FinalScale_0') if self.use_end_scale else None


    # Configuration update
    ###########################################################################
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'u_dim':              self.u_dim,
            'y_dim':              self.y_dim,
            'G_dim':              self.G_dim,
            'layer_size_branch':  self.layer_size_branch,
            'layer_size_trunk':   self.layer_size_trunk,
            'act_branch':         self.act_branch,
            'act_trunk':          self.act_trunk,
            'kernel_initializer': self.kernel_initializer,
            'regularizer':        self.regularizer,
            'dropout_rate':       self.dropout_rate,
            'batch_norm_branch':  self.batch_norm_branch,
            'batch_norm_trunk':   self.batch_norm_trunk,
            'weight_norm':        self.weight_norm,
            'use_end_scale':      self.use_end_scale,
            'branches':           self.branches,
            'trunk':              self.trunk,
            'end_scale':          self.end_scale
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
            raise ValueError('[PrODE]: The inputs for the Deep Operator net '
                             'have to be a list or tuple.')

        # Input transformation
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

        # Output transformation
        if self.out_trans is not None:
            G_uy = self.out_trans(G_uy)
        
        return G_uy


    # Steps
    ###########################################################################
    # Training Step
    def train_step(self, data):
        if self.res_loss_fn is not None:
            return steps.train_step(self, data)
        else:
            return super(OpNN, self).train_step(data)

    # Testing Step
    def test_step(self, data):
        if self.res_loss_fn is not None:
            return steps.test_step(self, data)
        else:
            return super(OpNN, self).test_step(data)


    # Graph
    ###########################################################################
    def build_graph(self):
        x_u = tf.keras.Input(shape=(self.u_dim,), name="branch_input")
        x_y = tf.keras.Input(shape=(self.y_dim,), name="trunk_input")
        x   = [x_u, x_y]
        return tf.keras.Model(inputs=x, outputs=self.call(x), name=self.name)
        