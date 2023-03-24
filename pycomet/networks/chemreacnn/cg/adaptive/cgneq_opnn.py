import tensorflow as tf

from pycomet import utils
from pycomet.networks.basenet import BaseNet
from pycomet.networks.chemreacnn.cg.adaptive.cg_opnn import CGOpNN
from pycomet.networks.chemreacnn.cg.adaptive.neq_opnn import NeqOpNN


class CGNeqOpNN(BaseNet):

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
        self.groups_graph = groups_graph
        # Layers --------------------------------------------------------------
        # Layers names
        if lays_name == None:
            self.lays_name = self.lays_name + ("cgopnn", "neqopnn")
        else:
            self.lays_name = lays_name
        # Layers properties
        self.lays_propty = self.check_layers(
            layers=self.lays_propty,
            not_opt=("cgopnn", "neqopnn"),
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
            if identifier == "cgopnn":
                lays[identifier] = CGOpNN(
                    dim=propty["dim"],
                    groups_graph=self.groups_graph,
                    lays_propty=propty["lays_propty"]
                )
            elif identifier == "neqopnn":
                lays[identifier] = NeqOpNN(
                    dim=propty["dim"],
                    groups_graph=self.groups_graph,
                    lays_propty=propty["lays_propty"]
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
        mask = self.active_lays["neq_opnn"](x)
        # Kernel - CG-DeepONet
        y = tf.nest.flatten([y, mask])
        y = self.active_lays["cg_opnn"](y)
        return y
