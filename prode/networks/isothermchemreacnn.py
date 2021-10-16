import itertools
import tensorflow as tf
import tensorflow.keras.layers as tf_layers

from prode.utilities.model import activations
from prode.utilities.model import initializers
from prode.utilities.model import regularizers
from prode.utilities.model import layers
from prode.utilities import utils
from prode.utilities import const
from prode.networks.tsnn import TimeScaleNN
from prode.networks.net import Net


@tf.keras.utils.register_keras_serializable(package='PrODE', name='IsoThermChemReacNN')
class IsoThermChemReacNN(Net):
    """Multi-Physics Multi-Scale Deep Operator neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        tsnn_dim,
        groups_graph,
        names,
        temp_limits,
        grouping,
        levels,
        lev_bin_map,
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
        name='isothermchemreacnn',
        *args,
        **kwargs
    ):
        super(IsoThermChemReacNN, self).__init__(
            dim=dim,
            block=block,
            name=name,
            *args,
            **kwargs
        )

        ### Physical Quantities -----------------------------------------------
        self.groups_graph = groups_graph
        self.TS = utils.depth(self.groups_graph)
        if len(names) != self.TS+1:
            utils.raise_value_err(
                "The number of `names` passed into `{}` "
                "has to be equal to the number of timescales "
                "in `groups_graph.".format(self.name))
        else:
            self.names = names

        # Timescales connectivity
        self.ts_fast_conn = self.ts_to_fastest_connectivity(self.groups_graph)  # {0: [[0, 1, 2, 3, 4, 5, 6, 7, 8]], 1: [[0, 1, 2], [3, 4, 5], [6, 7, 8]], 2: None}
        self.ts_ts_conn = self.ts_to_ts_connectivity(self.groups_graph)         # {0: [[0, 1, 2]], 1: [[0, 1, 2], [3, 4, 5], [6, 7, 8]], 2: None}

        # Temperature
        self.temp_limits = temp_limits

        # Levels & grouping
        self.grouping = grouping
        self.levels = levels
        self.lev_bin_map = lev_bin_map

        ### Hyperparameters Initialization ------------------------------------
        self.layer_size = utils.value_to_dict(layer_size, self.names)
        self.activation = utils.value_to_dict(activation, self.names)
        self.kernel_initializer = utils.value_to_dict(
            kernel_initializer, self.names
        )
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm

        ### TimeScaleNNs Initialization ---------------------------------------
        self.tsnn_dim = tsnn_dim
        self.opnn_end_scale = utils.value_to_dict(opnn_end_scale, self.names)
        self.stacked = stacked
        self.timescales = [
            TimeScaleNN(
                self.tsnn_dim[name],
                self.ts_fast_conn[i],
                self.temp_limits,
                self.grouping[i],
                self.levels,
                self.lev_bin_map[i],
                self.layer_size[name],
                self.activation[name],
                self.kernel_initializer[name],
                regularization=self.regularization,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                weight_norm=self.weight_norm,
                opnn_end_scale=self.opnn_end_scale[name],
                stacked=self.stacked,
                block=self.block,
                name=name
            ) for i, name in enumerate(self.names)
        ]

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'groups_graph': self.groups_graph,
            'names': self.names,
            'temp_limits': self.temp_limits,
            'grouping': self.grouping,
            'levels': self.levels,
            'lev_bin_map': self.lev_bin_map,
            'tsnn_dim': self.tsnn_dim,
            'layer_size': self.layer_size,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'regularization': self.regularization,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'opnn_end_scale': self.opnn_end_scale,
            'stacked': self.stacked
        }
        base_config = super(IsoThermChemReacNN, self).get_config()
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
        y = [ ts([x_u, x_y], training, mask) for ts in self.timescales ]

        # Outputs manipulation ------------------------------------------------
        if len(y) > 1:

            # Concatenate outputs
            y = [ tf_layers.Concatenate()(y_i) \
                if isinstance(y_i, (list,tuple)) else y_i for y_i in y]

            # Multiply time scales
            for ts, indeces in self.ts_ts_conn.items():
                if indeces:
                    y_slow = tf.split(
                        y[ts], num_or_size_splits=y[ts].shape[1], axis=1
                    )
                    y_fast = [ tf.gather(y[ts+1], idx, axis=1) for idx in indeces ]
                    if len(y_slow) != len(y_fast):
                        y_slow = y_slow[-len(y_fast):]
                    y_mul = [ tf_layers.Multiply()([ys, yf]) \
                        for ys, yf in zip(y_slow, y_fast) ]
                    y[ts+1] = tf_layers.Concatenate()(y_mul) \
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
    def ts_to_ts_connectivity(self, graph, level=0, conn={}):
        d = {}
        conn[level] = []
        if isinstance(graph, dict):
            for g in graph.values():
                if hasattr(g, 'keys'):
                    l = list(g.keys())
                    d = {**d, **g}
                else:
                    l = d = g
                conn[level].append(l)
        else:
            conn[level] = None
        if d != {}:
            return self.ts_to_ts_connectivity(d, level=level+1, conn=conn)
        else:
            return conn

    # def ts_to_ts_connectivity(self, graph, level=0, conn={}):
    #     d = {}
    #     conn[level] = []
    #     for g in graph.values():
    #         if hasattr(g, 'keys'):
    #             l = list(g.keys())
    #             d = {**d, **g}
    #         else:
    #             l = g
    #         conn[level].append(l)
    #     if d != {}:
    #         return self.ts_to_ts_connectivity(d, level=level+1, conn=conn)
    #     else:
    #         return conn

    def ts_to_fastest_connectivity(self, graph, ts={}):
        conn = self.graph_to_nested_list(graph)
        for l in range(-1, self.TS):
            ts[l+1] = self.merge(conn, it_max=l+1)
        return ts

    def graph_to_nested_list(self, graph):
        if isinstance(graph, dict):
            ls = []
            for k, v in graph.items():
                ls.append(self.graph_to_nested_list(v))
            return ls
        else:
            if isinstance(graph, (list, tuple, set)):
                return list(graph)
            else:
                utils.raise_value_err(
                    "The last level of the groups graph has to be a list."
                )

    def merge(self, ls, it=0, it_max=1):
        if any(isinstance(l, list) for l in ls):
            if it < it_max:
                ls = list(itertools.chain.from_iterable(ls))
                return self.merge(ls, it=it+1, it_max=it_max)
            else:
                return self.join(ls)

    def join(self, ls):
        try:
            ls = [list(itertools.chain.from_iterable(l)) for l in ls]
            return self.join(ls)
        except:
            return ls
