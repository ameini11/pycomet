import tensorflow as tf

from pycomet import utils
from pycomet import networks
from pycomet.networks.basenet import BaseNet
from pycomet.networks.opnn import _init_input_lay, _manipulate_input
from pycomet.networks.chemreacnn.cg.adaptive.ts_opnn import TimeScaleOpNN
from pycomet.networks.chemreacnn.cg.adaptive.layers import CurrentToNextTS
from pycomet.networks.chemreacnn.cg.adaptive.groups_connect import GroupsConnect


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
        for i, (name, ipropty) in enumerate(propty.items()):
            lays[name] = TimeScaleOpNN(
                dim=ipropty["dim"],
                lays_propty=ipropty["lays_propty"],
                fastest_to_current_ts=self.grp_conn.fastest_to_current_ts[i]
            )
        return lays

    # Calling
    ###########################################################################
    def call_net(
        self,
        x,
        training=None
    ):
        y, tol = x[:-1], x[-1]
        # Kernel - Neq-DeepONet
        neq_mask = self.active_lays["neq_opnn"](x)
        # Kernel - CG-DeepONet
        y = tf.nest.flatten([y, neq_mask])
        y = self.active_lays["cg_opnn"](y)
        return y_cg