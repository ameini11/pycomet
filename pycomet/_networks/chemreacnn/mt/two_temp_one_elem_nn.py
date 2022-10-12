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
        _propty = {}
        if 'xform' in propty:
            if propty['xform'] is not None:
                if not isinstance(propty[k], tf_layers.Lambda):
                    args = utils.get_fun_args(propty[k])
                    fn = tf_layers.Lambda(
                        function=utils.lambda_lay_dec(propty[k])
                    )
                else:
                    args = utils.get_fun_args(propty[k].function)
                    propty[k].function = utils.lambda_lay_dec(
                        propty[k].function
                    )
                    fn = propty[k]
                _propty[k] = {'fn': fn, 'args': args}
                if any([k not in ('x', 'y') for k in _propty[k]['args']]):
                    utils.raise_value_err(
                        "The outputs transformation function, 'xform', "
                        "accept only ('x', 'y') as function signature, "
                        "where 'x' <- inputs and 'y' <- model ouputs."
                    )
            else:
                _propty['xform'] = propty['xform']
        else:
            th_lay = propty['lay_id'] if callable(propty['lay_id']) \
                else utils.get_class(mt_layers, propty['lay_id'])
            _propty = {
                'xform': {
                    'fn': th_lay(**propty['kwargs']), 'args': ('x', 'y')
                }
            }
        return _propty