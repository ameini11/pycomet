import tensorflow as tf

from pycomet.networks.chemreacnn.mt import layers as mt_layers
from pycomet.networks.mmopnn import MMOpNN
from pycomet import config as cfg
from pycomet import utils


@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='TwoTempOneElemNN'
)
class TwoTempOneElemNN(MMOpNN):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        lays_name=None,
        lays_propty=None,
        *args,
        **kwargs
    ):
        super(TwoTempOneElemNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            *args,
            **kwargs
        )

    # Layers initialization
    ###########################################################################
    def init_output_lay(self, identifier, propty):
        th_lay = propty['lay_id'] if callable(propty['lay_id']) \
            else utils.get_class(mt_layers, propty['lay_id'])
        return {
            'xform': {
                'fn': th_lay(**propty['kwargs']), 'args': ('x', 'y')
            }
        }
