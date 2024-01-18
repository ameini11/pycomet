import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet import networks
from pycomet.networks.basenet import BaseNet
from pycomet.networks.flexopnn import _eval_prenet
from pycomet.networks.opnn import _init_input_lay, _manipulate_input


class MMOpNN(BaseNet):
    """Multi-Physics Multi-Scale Deep Operator neural network."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        lays_name=None,
        lays_propty=None,
        noisy_io=None,
        inference=None,
        masking=False,
        force_siso=False,
        *args,
        **kwargs
    ):
        super(MMOpNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            noisy_io=noisy_io,
            inference=inference,
            masking=masking,
            force_siso=force_siso,
            *args,
            **kwargs
        )
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
            self.lays_name = self.lays_name + ('prenet', 'opnns')
        else:
            self.lays_name = lays_name
        # Layers properties
        self.lays_propty = self.check_layers(
            layers=self.lays_propty,
            not_opt=('opnns',),
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
            if identifier == 'input':
                lays[identifier] = _init_input_lay(self, identifier, propty)
            elif identifier == 'prenet':
                lays[identifier] = {
                    'from': propty['from'],
                    'to': propty['to'],
                    'block': self.get_lay(
                        propty={
                            'lay_id': 'PreNetBlock',
                            'kwargs': propty['kwargs']
                        }
                    )
                }
            elif identifier == 'opnns':
                lays[identifier] = self.init_opnns_lay(identifier, propty)
            elif identifier == 'output':
                lays[identifier] = self.init_inout_lay(propty)
        return lays

    def init_opnns_lay(self, identifier, propty):
        if not isinstance(propty, dict):
            utils.raise_value_err(
                "The properties for 'opnns' layer has to be provided "
                "as a dictionary of (opnn_name, opnn_args) pairs."
            )
        return {name: self.get_opnn(opnn_lays_propty) \
            for name, opnn_lays_propty in propty.items()}

    def get_opnn(self, propty={}):
        opnn = 'OpNN' if 'lay_id' not in propty else propty['lay_id']
        opnn = opnn if callable(opnn) else utils.get_class(networks, opnn)
        return opnn(
            dim=propty['dim'],
            lays_propty=propty['lays_propty']
        )

    # Calling
    ###########################################################################
    def call_net(self, x, training=None):
        y = x
        # Inputs manipulation
        y = _manipulate_input(self, y)
        # Global preNet
        y = _eval_prenet(self, y, training)
        # Kernel - OpNNs
        y_ = []
        for name, opnn in self.active_lays['opnns'].items():
            y_.append(opnn(y, training=training))
        # Outputs manipulation
        y = self._manipulate_inout(
            inputs=dict(x=x, y=y_), identifier='output', training=training
        )
        return y
