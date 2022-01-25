import itertools
import tensorflow as tf
import tensorflow.python.keras.layers as tf_layers

from prode.utilities import utils
from prode.networks.tsnn import TimeScaleNN
from prode.networks.net import Net


@tf.keras.utils.register_keras_serializable(package='PrODE', name='IsoThermChemReacNN')
class IsoThermChemReacNN(Net):
    """Multi-Physics Multi-Scale Deep Operator neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        ts_thermochem,
        groups_graph,
        temp_limits,
        dim,
        tsnn_dim,
        names,
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
        cross_encoder=False,
        cross_encoder_layer_size=None,
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

        # Timescales thermochemistry
        self.ts_thermochem = ts_thermochem

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
        self.cross_encoder = cross_encoder
        self.cross_encoder_layer_size = cross_encoder_layer_size

        ### TimeScaleNNs Initialization ---------------------------------------
        self.sub_nets = ("branch", "trunk")
        self.tsnn_dim = tsnn_dim
        self.opnn_end_scale = utils.value_to_dict(opnn_end_scale, self.names)
        self.stacked = stacked
        self.timescales = [
            TimeScaleNN(
                **self.ts_thermochem[name],
                temp_limits=self.temp_limits,
                dim=self.tsnn_dim[name],
                ts_fast_conn=self.ts_fast_conn[i],
                layer_size=self.layer_size[name],
                activation=self.activation[name],
                kernel_initializer=self.kernel_initializer[name],
                regularization=self.regularization,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                weight_norm=self.weight_norm,
                opnn_end_scale=self.opnn_end_scale[name],
                stacked=self.stacked,
                block=self.block,
                cross_encoder=self.cross_encoder,
                cross_encoder_layer_size=self.cross_encoder_layer_size,
                name=name
            ) for i, name in enumerate(self.names)
        ]

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'ts_thermochem': self.ts_thermochem,
            'groups_graph': self.groups_graph,
            'temp_limits': self.temp_limits,
            'sub_nets': self.sub_nets,
            'tsnn_dim': self.tsnn_dim,
            'names': self.names,
            'layer_size': self.layer_size,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'regularization': self.regularization,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'opnn_end_scale': self.opnn_end_scale,
            'stacked': self.stacked,
            'cross_encoder': self.cross_encoder,
            'cross_encoder_layer_size': self.cross_encoder_layer_size
        }
        base_config = super(IsoThermChemReacNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=False, mask=None, noneq_factor=False):

        # Inputs manipulation -------------------------------------------------
        if isinstance(x, (list, tuple)):
            x = {k: x[i] for i, k in enumerate(self.sub_nets)}
        else:
            utils.raise_value_err(
                "The inputs of the `{}` net have to be a `list` "
                "or a `tuple`: [u, y]".format(self.name))

        # Input transformation
        if self.inp_trans is not None:
            for k in self.sub_nets:
                if isinstance(self.inp_trans, dict):
                    x[k] = self.inp_trans[k](x[k])
                else:
                    x[k] = self.inp_trans(x[k])

        x = tf.nest.flatten(x)

        # Kernel - Timescales NN ----------------------------------------------
        y = [ ts(x, training, mask, noneq_factor) for ts in self.timescales ]
        if training is False and noneq_factor is True:
            return y if len(y) > 1 else y[-1]

        # Outputs manipulation ------------------------------------------------
        if len(y) > 1:
            # Concatenate Timescales NN outputs
            y = [ tf_layers.Concatenate()(y_i) \
                if isinstance(y_i, (list,tuple)) else y_i for y_i in y ]
            # Loop over timescales
            for ts, indeces in self.ts_ts_conn.items():
                if indeces:
                    # Split the slower timescale
                    y_slow = tf.split(
                        y[ts], num_or_size_splits=y[ts].shape[1], axis=1
                    )
                    # Gather the faster timescale by indeces
                    y_fast = [ tf.gather(y[ts+1], idx, axis=1) \
                        for idx in indeces ]
                    if len(y_slow) != len(y_fast):
                        y_slow = y_slow[:len(y_fast)]
                    # Multiply slower and faster timescales
                    y_mul = [ tf_layers.Multiply()([ys, yf]) \
                        for ys, yf in zip(y_slow, y_fast) ]
                    y[ts+1] = tf_layers.Concatenate()(y_mul) \
                        if len(y_mul) > 1 else y_mul[-1]
            return y
        else:
            return y[-1]

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
