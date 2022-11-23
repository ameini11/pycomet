from pycomet.networks.basenet import BaseNet
from pycomet import networks
from pycomet import utils


class MixedNet(BaseNet):
    """Multi-Physics Multi-Scale Deep Operator neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        lays_name=None,
        lays_propty=None,
        force_siso=False,
        *args,
        **kwargs
    ):
        super(MMOpNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            force_siso=force_siso,
            *args,
            **kwargs
        )
        # Layers --------------------------------------------------------------
        # Layers names
        if lays_name == None:
            self.lays_name = self.lays_name + ('networks',)
        else:
            self.lays_name = lays_name
        # Layers properties
        self.lays_propty = self.check_layers(
            layers=self.lays_propty,
            not_opt=('networks',),
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
            if identifier in ('input', 'output'):
                lays[identifier] = self.init_inout_lay(propty)
            elif identifier == 'networks':
                lays[identifier] = self.init_nets_lay(identifier, propty)
        return lays

    def init_nets_lay(self, identifier, propty):
        if not isinstance(propty, dict):
            utils.raise_value_err(
                "The properties for 'networks' layer has to be provided "
                "as a dictionary of (net_name, net_args) pairs."
            )
        return {name: self.get_net(net_lays_propty, name) \
            for name, net_lays_propty in propty.items()}

    def get_net(self, propty={}, name=''):
        net = 'FFNN' if 'lay_id' not in propty else propty['lay_id']
        net = net if callable(net) else utils.get_class(networks, net)
        return net(
            dim=propty['dim'],
            lays_propty=propty['lays_propty'],
            name='_'.join((self.name, name))
        )

    # Calling
    ###########################################################################
    def call_net(self, x, training=None):
        y = x
        # Inputs manipulation
        y = self._manipulate_inout(
            inputs=dict(x=x, y=y), identifier='input', training=training
        )
        # Kernel - Networks
        y = [net(y, training=training) \
             for net in self.active_lays['networks'].values()]
        # Outputs manipulation
        y = self._manipulate_inout(
            inputs=dict(x=x, y=y), identifier='output', training=training
        )
        return y
