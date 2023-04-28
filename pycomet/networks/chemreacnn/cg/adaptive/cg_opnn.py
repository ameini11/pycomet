import tensorflow as tf

from pycomet import utils
from pycomet import networks
from pycomet.networks import elements as nn_elem
from pycomet.networks.basenet import BaseNet
from pycomet.networks.opnn import _init_input_lay, _manipulate_input
from pycomet.networks.chemreacnn.cg.adaptive.ts_opnn import TimeScaleOpNN
from pycomet.networks.chemreacnn.cg.adaptive.layers import Boltzmann
from pycomet.networks.chemreacnn.cg.adaptive.layers import CurrentToNextTS
from pycomet.networks.chemreacnn.cg.adaptive.tools.groups_connect import GroupsConnect


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
        inference=None,
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        pred_temp=True,
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        *args,
        **kwargs
    ):
        # Layers names
        super(CGOpNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            force_siso=force_siso,
            inference=inference,
            *args,
            **kwargs
        )
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.pred_temp = pred_temp
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # Input dimensionality ------------------------------------------------
        if not isinstance(self.input_dim, (list, tuple)):
            utils.raise_value_err(
                f"`dim['inp']` passed into `{self.name}` has to be a "
                "`list` or a `tuple`of length 2: [u_dim, y_dim]"
            )
        # Timescales connectivity ---------------------------------------------
        self.grp_conn = GroupsConnect(groups_graph)
        # Layers --------------------------------------------------------------
        # Default layers
        self.default_lays_propty.update({
            "input": {
                "branch": {
                    "index": 0,
                    "xform": None
                },
                "trunk": {
                    "index": 1,
                    "xform": None
                }
            },
            "output": {
                "xform": None
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
            elif identifier == "temp":
                lays[identifier] = self.init_temp_lay(propty)
            elif identifier == "mass":
                lays[identifier] = self.init_mass_lay(propty)
            elif identifier == 'output':
                lays[identifier] = self.init_inout_lay(propty)
        lays["ts_connect"] = CurrentToNextTS(self.grp_conn.current_to_next_ts)
        return lays

    def init_temp_lay(self, propty={}):
        opnn = "OpNN" if "lay_id" not in propty else propty["lay_id"]
        opnn = opnn if callable(opnn) else utils.get_class(networks, opnn)
        return opnn(
            dim=propty["dim"],
            lays_propty=propty["lays_propty"]
        )

    def init_mass_lay(self, propty):
        if not isinstance(propty, dict):
            utils.raise_value_err(
                "The properties for 'mass' layer has to be provided "
                "as a dictionary of (ts_opnn_name, ts_opnn_args) pairs."
            )
        lays = {}
        for name, ipropty in propty.items():
            lays[name] = TimeScaleOpNN(
                dim=ipropty["dim"],
                lays_propty=ipropty["lays_propty"],
                fastest_to_current_ts=self.grp_conn.fastest_to_current_ts[name]
            )
        return lays

    # Calling
    ###########################################################################
    def call_net(
        self,
        x,
        training=None
    ):
        y = x
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if not self.pred_temp:
            y, T = y[:-1], y[-1]
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # Inputs manipulation
        y = _manipulate_input(self, y)
        x_ts = [y["branch"], y["trunk"]]
        # Kernel - Temperature (OpNN)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.pred_temp:
            T = None
            if "temp" in self.active_lays:
                T = self.active_lays["temp"](y, training=training)
        # T = None
        # if "temp" in self.active_lays:
        #     T = self.active_lays["temp"](y, training=training)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # Kernel - Mass (TimeScaleOpNNs)
        Y = None
        if "mass" in self.active_lays:
            Y = []
            x_ts = tf.nest.flatten([x_ts, T])
            for ts in self.active_lays["mass"].values():
                Y.append(ts(x=x_ts, training=training))
        # Outputs manipulation
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if not self.pred_temp:
            y = tf.nest.flatten(Y)
            return y if len(y) > 1 else y[-1]
        else:
            return self._manipulate_output(Y, T)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    def call_net_infer(
        self,
        x,
        x_extra,
        *args,
        **kwargs
    ):
        # Extract inputs
        y, mask = x, x_extra
        # Inputs manipulation
        y = _manipulate_input(self, y)
        x_ts = [y["branch"], y["trunk"]]
        # Kernel - Temperature (OpNN)
        T = None
        if "temp" in self.active_lays:
            T = self.active_lays["temp"](y)
        # Kernel - Mass (TimeScaleOpNNs)
        Y = []
        x_ts = tf.nest.flatten([x_ts, T])
        for i, ts in enumerate(self.active_lays["mass"].values()):
            if i == 0:
                # First Timescale
                y_i = ts(x=x_ts)
            else:
                # Second to highest Timescales
                y_i = ts(x=x_ts, mask=mask[i-1])
            Y.append(y_i)
        # Outputs manipulation
        return self._manipulate_output(Y, T)

    # Sub-operations
    ###########################################################################
    def _manipulate_output(self, Y, T):
        y = []
        # Mass fractions
        if Y is not None:
            Y = tf.nest.flatten(Y)
            if len(Y) > 1:
                Y = self.active_lays["ts_connect"](Y)
            if self.inference is not None:
                if len(Y) > 1:
                    # Compose outputs: atom mass fraction + last timescale
                    Y = [tf.gather(Y[0], [0], axis=1), Y[-1]]
                else:
                    Y = nn_elem.Split(Y[-1].shape[1], axis=1)(Y[-1])
            y.append(Y)
        # Temperature
        if T is not None:
            y.append(T)
        # Flattening
        y = tf.nest.flatten(y)
        y = y if len(y) > 1 else y[-1]
        # Output transformation
        return self._manipulate_inout(inputs=dict(y=y), identifier='output')
