import itertools

from keras import layers as tf_layers
from pycomet import utils
from pycomet import networks
from pycomet.networks.basenet import BaseNet
from pycomet.networks.chemreacnn.cg.adaptive import TimeScaleOpNN
from pycomet.networks.chemreacnn.cg.adaptive import CurrentToNextTS
from pycomet.networks.opnn import _init_input_lay, _manipulate_input


class CGOpNN(BaseNet):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        groups_graph,
        lays_name=None,
        lays_propty=None,
        force_siso=False,
        *args,
        **kwargs
    ):
        # Layers names
        super(CGOpNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            force_siso=force_siso,
            *args,
            **kwargs
        )
        # Input dimensionality ------------------------------------------------
        if not isinstance(self.input_dim, (list, tuple)):
            utils.raise_value_err(
                f"`dim['inp']` passed into `{self.name}` has to be a "
                "`list` or a `tuple`of length 2: [u_dim, y_dim]"
            )
        # Timescales connectivity ---------------------------------------------
        self.fastest_to_current_ts = self.connect_fastest_to_current_ts(groups_graph)   # {0: [[0, 1, 2, 3, 4, 5, 6, 7, 8]], 1: [[0, 1, 2], [3, 4, 5], [6, 7, 8]], 2: None}
        self.current_to_next_ts = self.connect_current_to_next_ts(groups_graph)         # {0: [[0, 1, 2]], 1: [[0, 1, 2], [3, 4, 5], [6, 7, 8]], 2: None}
        # Layers --------------------------------------------------------------
        # Default layers
        self.default_lays_propty.update({
            'input': {
                'branch': {
                    'index': 0,
                    'xform': None
                },
                'trunk': {
                    'index': 1,
                    'xform': None
                }
            },
            'output': {
                'xform': None
            }
        })
        # Layers names
        if lays_name == None:
            self.lays_name = self.lays_name + ("temp", "mass")
        else:
            self.lays_name = lays_name
        # Layers properties
        self.lays_propty = self.check_layers(
            layers=self.lays_propty,
            not_opt=("temp", "mass"),
            default=self.default_lays_propty
        )
        # Initializing
        self.active_lays = self.init_layers()
        self.init_preinput_lay()

    # Layers initialization
    ###########################################################################
    def init_layers(self):
        lays = {}
        for identifier, propty in self.lays_propty.items():
            if identifier == "input":
                lays[identifier] = _init_input_lay(self, identifier, propty)
            if identifier == "temp":
                lays[identifier] = self.init_temp_lay(propty)
            elif identifier == "mass":
                lays[identifier] = self.init_mass_lay(propty)
        lays["ts_connect"] = CurrentToNextTS(self.current_to_next_ts)
        return lays

    def init_temp_lay(self, propty={}):
        opnn = 'OpNN' if 'lay_id' not in propty else propty['lay_id']
        opnn = opnn if callable(opnn) else utils.get_class(networks, opnn)
        return opnn(
            dim=propty['dim'],
            lays_propty=propty['lays_propty']
        )

    def init_mass_lay(self, propty):
        if not isinstance(propty, dict):
            utils.raise_value_err(
                "The properties for 'mass' layer has to be provided "
                "as a dictionary of (ts_opnn_name, ts_opnn_args) pairs."
            )
        lays = {}
        for i, (name, ipropty) in enumerate(propty.items()):
            lays[name] = TimeScaleOpNN(
                dim=ipropty['dim'],
                lays_propty=ipropty['lays_propty'],
                fastest_to_current_ts=self.fastest_to_current_ts[i]
            )
        return lays

    # Calling
    ###########################################################################
    def call_net(
        self,
        x,
        training=None,
        neq_mask=None,
        eq_lay=None
    ):
        y = x
        # Inputs manipulation
        y = _manipulate_input(self, y)
        # Kernel - Temperature (OpNN)
        T = None
        if 'temp' in self.active_lays:
            T = self.active_lays['temp'](y, training=training)
        # Kernel - Mass (TimeScaleOpNNs)
        kwargs = dict(x=[y, T], training=training, eq_lay=eq_lay)
        if training or (neq_mask is None):
            y = [ts(**kwargs) for ts in self.active_lays['mass'].values()]
        else:
            y = []
            for i, ts in enumerate(self.active_lays['mass'].values()):
                if i != 0:
                    kwargs["neq_mask"] = neq_mask[i-1]
                y.append(ts(**kwargs))
        # Outputs manipulation
        y = self._manipulate_output(y)
        if T is not None:
            return tf.nest.flatten([y, T])
        else:
            return y

    def _manipulate_output(self, y):
        if len(y) > 1:
            # Connect groups
            y = self.active_lays["ts_connect"](y)
            # Compose outputs: atom mass fraction + last timescale
            return [tf.gather(y[0], [0], axis=1), y[-1]]
        else:
            return nn_elem.Split(y[-1].shape[1], axis=1)(y[-1])

    # Groups connectivity/indexing
    ###########################################################################
    def connect_current_to_next_ts(self, graph, level=0, connectivity={}):
        d = {}
        connectivity[level] = []
        if isinstance(graph, dict):
            for g in graph.values():
                if hasattr(g, 'keys'):
                    l = list(g.keys())
                    d = {**d, **g}
                else:
                    l = d = g
                connectivity[level].append(l)
        else:
            connectivity[level] = None
        if d != {}:
            return self.connect_current_to_next_ts(
                d, level=level+1, connectivity=connectivity
            )
        else:
            return connectivity

    def connect_fastest_to_current_ts(self, graph, ts={}):
        nb_ts = utils.depth(graph)
        connectivity = self.graph_to_nested_list(graph)
        for l in range(-1, nb_ts):
            ts[l+1] = self.merge(connectivity, it_max=l+1)
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
