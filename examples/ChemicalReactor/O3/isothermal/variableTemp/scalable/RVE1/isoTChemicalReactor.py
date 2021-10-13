import tensorflow as tf
import tensorflow.keras.layers as tf_layers

from prode.utilities.model import activations
from prode.utilities.model import initializers
from prode.utilities.model import regularizers
from prode.utilities.model import layers
from prode.utilities import utils
from prode.utilities import const
from prode.networks.opnn import OpNN
from prode.networks.net import Net


@tf.keras.utils.register_keras_serializable(package='PrODE', name='IsoTChemicalReactor')
class IsoTChemicalReactor(Net):
    """Multi-Physics Multi-Scale Deep Operator neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        groups_graph,
        names,
        opnn_dim,
        layer_size,
        activation,
        kernel_initializer,
        end_scale=None,
        opnn_inp_trans=None,
        opnn_out_trans=None,
        opnn_end_scale='bias',
        stacked=True,
        regularization=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        block='FFBlock',
        name='mmopnn',
        *args,
        **kwargs
    ):
        super(IsoTChemicalReactor, self).__init__(
            dim=dim,
            block=block,
            name=name,
            *args,
            **kwargs
        )



        ### Final Scaling -----------------------------------------------------
        self.groups_graph = groups_graph
        self.TS = utils.depth(self.graph)


        if len(names) != self.TS:
            utils.raise_value_err(
                "The number of `names` passed into `{}` has to be "
                "equal to the number of timescales.".format(self.name))
        else:
            self.names = names

        self.groups_idx = self.indexing(self.groups_graph)

        self.ll_conn = self.lev_to_lev_connectivity(self.groups_graph)


        ### Hyperparameters Initialization ------------------------------------
        self.layer_size = layer_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm

        ### OpNNs Initialization ----------------------------------------------
        if len(names) <= 1:
            utils.raise_value_err(
                "The number of `names` passed into `{}` has to be "
                "grater than 1 (multiple DeepONets).".format(self.name))
        else:
            self.names = names
        self.opnn_dim = opnn_dim
        self.opnn_end_scale = opnn_end_scale
        self.stacked = stacked

        self.opnns = {}
        for ts, ts_name in enumerate(self.names):
            self.opnns[ts_name] = [
                OpNN(


                    self.tsnn_dim[ts_name],
                    self.layer_size[ts_name],
                    self.activation[ts_name],
                    self.kernel_initializer,
                    regularization=self.regularization,
                    end_scale=self.opnn_end_scale,
                    stacked=self.stacked,
                    dropout_rate=self.dropout_rate,
                    batch_norm=self.batch_norm,
                    weight_norm=self.weight_norm,
                    block=self.block,
                    name='opnn_'+ts_name+str(ts_i)

                    self,
                    dim,
                    groups_idx,
                    temp_limits,
                    database,
                    grouping,
                    layer_size,
                    activation,
                    kernel_initializer,
                    regularization=None,
                    dropout_rate=0,
                    batch_norm=None,
                    weight_norm=False,
                    opnn_end_scale='bias',
                    stacked=True,
                    block='FFBlock',
                    name='mmopnn',
                    *args,
                    **kwargs

                    
                ) for ts_i in range(len(self.group_idx[ts]))
            ]

        ### Final Scaling -----------------------------------------------------
        self.TS = self.time_scales(graph)

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'names': self.names,
            'opnn_dim': self.opnn_dim,
            'layer_size': self.layer_size,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'opnn_inp_trans': self.opnn_inp_trans,
            'opnn_out_trans': self.opnn_out_trans,
            'opnn_end_scale': self.opnn_end_scale,
            'regularization': self.regularization,
            'end_scale': self.end_scale,
            'stacked': self.stacked,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm
        }
        base_config = super(IsoTChemicalReactor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=None, mask=None):

        # Inputs manipulation -------------------------------------------------
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

        # Kernel - TimeScales -------------------------------------------------
        y = [ ts([x_u, x_y]) for ts in self.timescales ]

        # Outputs manipulation ------------------------------------------------
        if len(y) > 1:

            # Concatenate outputs
            y = [ tf_layers.Concatenate()(y_i) \
                for y_i in y if isinstance(y_i, (list,tuple)) else y_i ]

            # Multiply time scales
            for l, indeces in self.ll_conn.items():
                y_slow = tf.split(
                    y[l], num_or_size_splits=y[l].shape[1], axis=1
                )
                y_fast = [ tf.gather(y[l+1], idx, axis=1) for idx in indeces ]
                if len(y_slow) != len(y_fast):
                    y_slow = y_slow[-len(y_fast):]
                y_mul = [ tf_layers.Multiply()([ys, yf]) \
                    for ys, yf in zip(y_slow, y_fast) ]
                y[l+1] = tf_layers.Concatenate()(y_mul) \
                    if len(y_mul) > 1 else y_mul[-1]
        else:
            return y[-1]

    # Graph
    ###########################################################################
    def get_graph(self):
        x = [ tf.keras.Input(shape=(self.inp_dim[0],), name="branch_input"),
              tf.keras.Input(shape=(self.inp_dim[1],), name="trunk_input") ]
        y = self.call(x)
        return tf.keras.Model(inputs=x, outputs=y, name=self.name)

    # Groups connectivity/indexing
    ###########################################################################
    def lev_to_lev_connectivity(self, graph, level=0, conn={}):
        d = {}
        conn[level] = []
        for g in graph.values():
            if hasattr(g, 'keys'):
                l = list(g.keys())
                d = {**d, **g}
            else:
                l = g
            conn[level].append(l)
        if d != {}:
            return self.lev_to_lev_connectivity(d, level=level+1, conn=conn)
        else:
            return conn

    def indexing(self, graph, ts={}):
        conn = start_to_end_connectivity(graph)
        if self.TS_levels == 0:
            ts = list(conn)
        else:
            for l in range(self.TS_levels):
                ts[l] = merge(conn, it_max=l)
        return ts

    @staticmethod
    def nested_connectivity(graph):
        if isinstance(graph, dict):
            conn = []
            for k, v in graph.items():
                conn.append(start_to_end_connectivity(v))
            return conn
        else:
            if isinstance(graph, (list, tuple, set)):
                return list(graph)
            else:
                utils.raise_value_err(
                    "The last level of the groups graph has to be a list."
                )

    @staticmethod
    def merge(ls, it=0, it_max=1):
        if any(isinstance(l, list) for l in ls):
            if it < it_max:
                ls = list(itertools.chain.from_iterable(ls))
                return merge(ls, it=it+1, it_max=it_max)
            else:
                return join(ls)

    @staticmethod
    def join(ls):
        try:
            ls = [list(itertools.chain.from_iterable(l)) for l in ls]
            return join(ls)
        except:
            return ls


# =========================================================================== #
# TimeScale Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PrODE', name='TimeScale')
class TimeScale(Net):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        groups_idx,
        temp_limits,
        database,
        grouping,
        layer_size,
        activation,
        kernel_initializer,
        regularization=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        opnn_end_scale='bias',
        stacked=True,
        block='FFBlock',
        name='mmopnn',
        *args,
        **kwargs
    ):
        super(TimeScale, self).__init__(
            dim=dim,
            block=block,
            name=name,
            *args,
            **kwargs
        )

        ### Dimensionality ----------------------------------------------------
        # Inputs
        self.groups_idx = groups_idx
        self.temp_limits = temp_limits
        self.temperature_norm = layers.LinearNorm(
            *self.temp_limits, trainable=False
        )

        # Outputs
        if not isinstance(self.out_dim, (list,tuple)):
            self.out_dim = [self.out_dim]

        # OpNNs
        self.opnn_dim = [{
            "inp": self.inp_dim,
            "out": self.out_dim[i]
        } for i in range(len(self.out_dim))]

        ### Hyperparameters Initialization ------------------------------------
        self.layer_size = layer_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm

        ### OpNNs Initialization ----------------------------------------------
        self.opnn_end_scale = opnn_end_scale
        self.stacked = stacked

        self.opnns = [
            OpNN(
                self.opnn_dim[i],
                self.layer_size,
                self.activation,
                self.kernel_initializer,
                regularization=self.regularization,
                end_scale=self.opnn_end_scale,
                stacked=self.stacked,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                weight_norm=self.weight_norm,
                block=self.block,
                name='opnn_' + str(i)
            ) for i in range(len(self.out_dim))
        ]

        ### Equilibrium Layer -------------------------------------------------
        self.database = database
        self.grouping = grouping
        self.equilibrium = Equilibrium(self.database, self.grouping)

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'groups_idx': self.groups_idx,
            'temp_limits': self.temp_limits,
            'layer_size': self.layer_size,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'regularization': self.regularization,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'opnn_end_scale': self.opnn_end_scale,
            'stacked': self.stacked,
            'database': self.database,
            'grouping': self.grouping
        }
        base_config = super(TimeScale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=None, mask=None):

        # Extract inputs
        ic, t = x

        # Inputs manipulation -------------------------------------------------
        # Extract physical inputs
        T, rho, Y = tf.split(
            ic, num_or_size_splits=[1,1,self.inp_dim[0]-2], axis=1
        )

        # Normalize temperature
        T_norm = self.temperature_norm(T)

        # Compose new inputs
        Y_grouped = []
        for indeces in self.groups_idx:
            Y_grouped.append(
                tf.math.reduce_sum(
                    tf.gather(Y, indeces, axis=1), axis=1, keepdims=True
                )
            )
        ic = tf_layers.Concatenate()([T_norm, rho, *Y_grouped])

        # Kernel - OpNNs ------------------------------------------------------
        y = [ g_uy([ic, t]) for g_uy in self.opnns.values() ]

        # Outputs manipulation ------------------------------------------------
        # Equilibrium layer
        if self.equilibrium is not None:
            y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
            y = self.equilibrium(y, T)
            y = tf.split(y, num_or_size_splits=self.out_dim, axis=1)

        # Softmax layer
        y = [ tf_layers.Softmax(axis=-1)(y_i) for y_i in y ]

        return y if len(y) > 1 else y[-1]


# # =========================================================================== #
# # TimeScale Layer
# # =========================================================================== #
# @tf.keras.utils.register_keras_serializable(package='PrODE', name='TimeScale')
# class TimeScale(tf_layers.Layer):

#     # Initialization
#     ###########################################################################
#     def __init__(
#         self,
#         dim,
#         groups_idx,
#         temp_limits,
#         database,
#         grouping,
#         layer_size,
#         activation,
#         kernel_initializer,
#         regularization=None,
#         dropout_rate=0,
#         batch_norm=None,
#         weight_norm=False,
#         opnn_end_scale='bias',
#         stacked=True,
#         block='FFBlock',
#         name='mmopnn',
#         *args,
#         **kwargs
#     ):
#         super(TimeScale, self).__init__(
#             name=name,
#             *args,
#             **kwargs
#         )

#         ### Dimensionality ----------------------------------------------------
#         # Inputs
#         self.inp_dim = dim["inp"]
#         self.groups_idx = groups_idx
#         self.temp_limits = temp_limits
#         self.temperature_norm = layers.LinearNorm(
#             *self.temp_limits, trainable=False
#         )

#         # Outputs
#         self.out_dim = dim["out"]
#         if not isinstance(self.out_dim, (list,tuple)):
#             self.out_dim = [self.out_dim]

#         # OpNNs
#         self.opnn_dim = [{
#             "inp": self.inp_dim,
#             "out": self.out_dim[i]
#         } for i in range(len(self.out_dim))]

#         ### Hyperparameters Initialization ------------------------------------
#         self.layer_size = layer_size
#         self.activation = activation
#         self.kernel_initializer = kernel_initializer
#         self.regularization = regularization
#         self.dropout_rate = dropout_rate
#         self.batch_norm = batch_norm
#         self.weight_norm = weight_norm

#         ### OpNNs Initialization ----------------------------------------------
#         self.opnn_end_scale = opnn_end_scale
#         self.stacked = stacked
#         self.block = block if callable(block) else utils.get_class(layers, block)

#         self.opnns = [
#             OpNN(
#                 self.opnn_dim[i],
#                 self.layer_size,
#                 self.activation,
#                 self.kernel_initializer,
#                 regularization=self.regularization,
#                 end_scale=self.opnn_end_scale,
#                 stacked=self.stacked,
#                 dropout_rate=self.dropout_rate,
#                 batch_norm=self.batch_norm,
#                 weight_norm=self.weight_norm,
#                 block=self.block,
#                 name='opnn_' + str(i)
#             ) for i in range(len(self.out_dim))
#         ]

#         ### Equilibrium Layer -------------------------------------------------
#         self.database = database
#         self.grouping = grouping
#         self.equilibrium = Equilibrium(self.database, self.grouping)

#     # Configuration update
#     ###########################################################################
#     def get_config(self):
#         config = {
#             'ic_dim': self.ic_dim,
#             'inp_dim': self.inp_dim,
#             'groups_idx': self.groups_idx,
#             'temp_limits': self.temp_limits,
#             'out_dim': self.out_dim,
#             'layer_size': self.layer_size,
#             'activation': self.activation,
#             'kernel_initializer': self.kernel_initializer,
#             'regularization': self.regularization,
#             'dropout_rate': self.dropout_rate,
#             'batch_norm': self.batch_norm,
#             'weight_norm': self.weight_norm,
#             'opnn_end_scale': self.opnn_end_scale,
#             'stacked': self.stacked,
#             'block': self.block,
#             'database': self.database,
#             'grouping': self.grouping
#         }
#         base_config = super(TimeScale, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

#     # Calling
#     ###########################################################################
#     def call(self, x, training=None, mask=None):

#         # Extract inputs
#         ic, t = x

#         # Inputs manipulation -------------------------------------------------
#         # Extract physical inputs
#         T, rho, Y = tf.split(
#             ic, num_or_size_splits=[1,1,self.inp_dim[0]-2], axis=1
#         )

#         # Normalize temperature
#         T_norm = self.temperature_norm(T)

#         # Compose new inputs
#         Y_grouped = []
#         for indeces in self.groups_idx:
#             Y_grouped.append(
#                 tf.math.reduce_sum(
#                     tf.gather(Y, indeces, axis=1), axis=1, keepdims=True
#                 )
#             )
#         ic = tf_layers.Concatenate()([T_norm, rho, *Y_grouped])

#         # Kernel - OpNNs ------------------------------------------------------
#         y = [ g_uy([ic, t]) for g_uy in self.opnns.values() ]

#         # Outputs manipulation ------------------------------------------------
#         # Equilibrium layer
#         if self.equilibrium is not None:
#             y = tf_layers.Concatenate()(y) if len(y) > 1 else y[-1]
#             y = self.equilibrium(y, T)
#             y = tf.split(y, num_or_size_splits=self.out_dim, axis=1)

#         # Softmax layer
#         y = [ tf_layers.Softmax(axis=-1)(y_i) for y_i in y ]

#         # Return outputs
#         return y if len(y) > 1 else y[-1]


# =========================================================================== #
# Equilibrium Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PrODE', name='Equilibrium')
class Equilibrium(tf_layers.Subtract):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        database,
        grouping,
        name='equilibrium',
        *args,
        **kwargs
    ):
        super(Equilibrium, self).__init__(
            name=name,
            *args,
            **kwargs)

        # Retrieving Molecular data -------------------------------------------
        self.grouping = grouping
        self.levels_file = database + '/levels/O3_UMN/O2/FromUMN_Sorted.inp'
        self.binMap_file = database + '/grouping/O3_UMN/O2/LevelsMap_' \
            + self.grouping + '.csv'
        self.g, self.E, self.num_lev = self.read_levels()
        self.lev_to_bin, self.num_bins = self.read_groups()

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'grouping': self.grouping,
            'levels_file': self.levels_file,
            'binMap_file': self.binMap_file,
            'g': self.g,
            'E': self.E,
            'num_lev': self.num_lev,
            'lev_to_bin': self.lev_to_bin,
            'num_bins': self.num_bins
        }
        base_config = super(Equilibrium, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x, T = inputs
        else:
            utils.raise_value_err(
                "The inputs of the `{}` layer have to be a `list` "
                "or a `tuple`: [x, T]".format(self.name))
        e = self.energy_bins(T) * tf.math.reciprocal(T) / const.UKB
        return super(Equilibrium, self).call([x, e])

    def energy_bins(self, T):
        '''Compute bins partition function.'''
        T_rec = tf.math.reciprocal(tf.transpose(T))
        q = tf.math.multiply(
            self.g, tf.math.exp(
                - tf.tensordot(self.E, T_rec, 1) / const.UKB
            )
        )
        E_bins = []
        for bin_i in range(self.num_bins):
            mask = self.lev_to_bin == bin_i
            Q_bin_i = tf.math.reduce_sum(q[mask], axis=0, keepdims=True)
            E_bin_i = self.E[mask] * q[mask] * tf.math.reciprocal(Q_bin_i)
            E_bins.append(
                tf.math.reduce_sum(E_bin_i, axis=0, keepdims=True)
            )
        return tf.transpose(tf.concat(E_bins, 0))

    # Util Functions
    ###########################################################################
    def read_levels(self):
        '''Retrieve all the ro-vib levels.'''
        data = pd.read_csv(
            self.levels_file, header=None, skiprows=15, delim_whitespace=True
        )
        data = data.apply(pd.to_numeric, errors='coerce')
        vqn = np.array(data[0])                     # Vibrational Q.N.
        jqn = np.array(data[1])                     # Rotational  Q.N.
        g = 1.5*(2.0*jqn+1.0)                       # Degeneracy
        E = np.array(data[2]) * const.EH_to_J       # Energy in J
        E = E - E[0]
        num = len(vqn)
        g, E = list(map(lambda x: tf.expand_dims(tf.convert_to_tensor(x, \
            dtype=backend.floatx()), 1), [g, E]))
        return g, E, num

    def read_groups(self):
        '''Retrieve mapping of each rot-vib level to the respective bin.'''
        data = pd.read_csv(self.binMap_file, header=None, skiprows=1)
        data = data.apply(pd.to_numeric, errors='coerce')
        idx = np.array(data[0])                                                                           
        lev_to_bin = np.array(data[1]) - 1    # Level-to-Bin
        num = max(lev_to_bin) + 1
        return lev_to_bin, num
